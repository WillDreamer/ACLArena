"""Domain-balanced global data source for multi-domain SFT.

The default `RolloutDataSourceWithBuffer` draws each rollout batch as a
contiguous slice of one globally-shuffled pool. When the pool mixes domains of
very different sizes (oracle_sft: if=225k, search=157k, math=156k, tau=19k), a
uniformly-shuffled batch is dominated by the big domains and the small domain
(tau) is seen ~11x less often per step — and whole early batches can contain
almost no tau at all.

`BalancedRolloutDataSource` instead buckets the dataset by a per-record domain
key (default: `metadata["data_source"]`, which slime's Dataset lifts from a
top-level `data_source` field in the jsonl) and draws an EQUAL share from every
domain in each `get_samples(batch)` call. With 4 domains and batch=128 that is
32 records/domain/batch, so every step trains on all four domains evenly
regardless of their pool sizes.

Design mirrors RolloutDataSource so it is a drop-in for `--data-source-path`:
  * Each domain bucket is an independent shuffled list with its own offset.
  * A bucket that runs out is reshuffled and cycled (so the small domain simply
    repeats within an epoch — this is oversampling-by-repetition, the intended
    balancing behaviour). `epoch_id` advances when the LARGEST bucket wraps, so
    `--num-epoch` still means "one pass over the biggest domain".
  * save()/load() persist every bucket offset + epoch_id for exact resume.

Only the SFT path is targeted (n_samples_per_prompt=1, read-only), but the class
also carries the buffer plumbing (add_samples/buffer_filter) so it can stand in
for RolloutDataSourceWithBuffer without breaking RL callers.

Env knobs (read at construction, all optional):
  BALANCED_DOMAIN_KEY      metadata key holding the domain (default "data_source")
  BALANCED_DOMAINS         comma-separated explicit domain order (default: sorted
                           set discovered in the data). Fixing the order makes the
                           per-batch composition deterministic across resumes.
  BALANCED_DROP_UNKNOWN    if "1", drop records whose domain key is missing/empty
                           instead of bucketing them under "unknown" (default "0").
"""
import copy
import logging
import os
from collections import OrderedDict

import torch

from slime.rollout.data_source import RolloutDataSourceWithBuffer
from slime.utils.data import Dataset
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class BalancedRolloutDataSource(RolloutDataSourceWithBuffer):
    def __init__(self, args):
        # Deliberately skip RolloutDataSourceWithBuffer/RolloutDataSource __init__
        # data-loading (we build per-domain buckets instead), but reuse its buffer
        # machinery so RL callers still work.
        self.args = args

        # ── buffer plumbing (copied from RolloutDataSourceWithBuffer.__init__) ──
        from slime.rollout.data_source import pop_first
        from slime.utils.misc import load_function

        self.buffer = []
        if getattr(args, "buffer_filter_path", None) is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(args.buffer_filter_path)

        # ── bookkeeping shared with RolloutDataSource ──
        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.metadata = {}

        assert args.rollout_global_dataset, (
            "BalancedRolloutDataSource requires --rollout-global-dataset "
            "(remove --disable-rollout-global-dataset)."
        )

        self.domain_key = os.environ.get("BALANCED_DOMAIN_KEY", "data_source")
        self.drop_unknown = os.environ.get("BALANCED_DROP_UNKNOWN", "0") == "1"

        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        self.dataset = Dataset(
            args.prompt_data,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.rollout_max_prompt_len,
            prompt_key=args.input_key,
            multimodal_keys=args.multimodal_keys,
            label_key=args.label_key,
            metadata_key=args.metadata_key,
            tool_key=args.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
            seed=args.rollout_seed,
        )

        # ── bucket sample indices by domain ──
        # buckets: domain -> list[int] (indices into self.dataset.origin_samples)
        buckets = OrderedDict()
        n_unknown = 0
        for idx, sample in enumerate(self.dataset.origin_samples):
            dom = (sample.metadata or {}).get(self.domain_key)
            if dom is None or dom == "":
                if self.drop_unknown:
                    n_unknown += 1
                    continue
                dom = "unknown"
                n_unknown += 1
            buckets.setdefault(dom, []).append(idx)

        if not buckets:
            raise ValueError(
                f"BalancedRolloutDataSource found no records with domain key "
                f"'{self.domain_key}'. Check the data has a top-level "
                f"'{self.domain_key}' field (or set BALANCED_DOMAIN_KEY)."
            )

        # Fix the domain order: explicit env override, else sorted for determinism.
        explicit = os.environ.get("BALANCED_DOMAINS", "").strip()
        if explicit:
            order = [d.strip() for d in explicit.split(",") if d.strip()]
            missing = [d for d in order if d not in buckets]
            if missing:
                raise ValueError(
                    f"BALANCED_DOMAINS lists {missing} not present in the data. "
                    f"Found domains: {sorted(buckets)}"
                )
            extra = [d for d in buckets if d not in order]
            if extra:
                raise ValueError(
                    f"Data contains domains {extra} not listed in BALANCED_DOMAINS={order}. "
                    f"List all domains explicitly or unset BALANCED_DOMAINS."
                )
            self.domains = order
        else:
            self.domains = sorted(buckets)

        self.origin_buckets = {d: buckets[d] for d in self.domains}
        # per-domain live (shuffled) index order + read offset
        self.bucket_orders = {}
        self.bucket_offsets = {d: 0 for d in self.domains}
        self.bucket_epoch = {d: 0 for d in self.domains}
        for d in self.domains:
            self._shuffle_bucket(d, epoch=0)

        self._largest_domain = max(self.domains, key=lambda d: len(self.origin_buckets[d]))

        counts = {d: len(self.origin_buckets[d]) for d in self.domains}
        logger.info(
            f"[BalancedRolloutDataSource] domains={self.domains} counts={counts} "
            f"unknown={n_unknown} drop_unknown={self.drop_unknown} "
            f"largest={self._largest_domain} "
            f"(each get_samples draws an equal share per domain)"
        )

    # ── per-domain shuffling ──────────────────────────────────────────────
    def _shuffle_bucket(self, domain, epoch):
        import random

        idxs = list(self.origin_buckets[domain])
        if self.args.rollout_shuffle:
            # Distinct seed per (domain, epoch) so buckets don't move in lockstep.
            # Use the domain's fixed position (NOT builtin hash(): that is salted by
            # PYTHONHASHSEED and would reshuffle differently after a restart, so a
            # resumed offset would point into a different order and serve wrong data).
            domain_salt = self.domains.index(domain) + 1
            rng = random.Random(self.args.rollout_seed + epoch * 1000003 + domain_salt * 7919)
            rng.shuffle(idxs)
        self.bucket_orders[domain] = idxs
        self.bucket_offsets[domain] = 0
        self.bucket_epoch[domain] = epoch

    def _next_from_bucket(self, domain, k):
        """Pop k indices from a domain bucket, reshuffling+cycling on exhaustion."""
        out = []
        while k > 0:
            order = self.bucket_orders[domain]
            off = self.bucket_offsets[domain]
            take = order[off : off + k]
            out.extend(take)
            self.bucket_offsets[domain] = off + len(take)
            k -= len(take)
            if k > 0:
                # bucket exhausted -> next epoch for this bucket
                new_epoch = self.bucket_epoch[domain] + 1
                self._shuffle_bucket(domain, epoch=new_epoch)
                if domain == self._largest_domain:
                    # one full pass over the biggest domain == one global epoch
                    self.epoch_id = new_epoch
        return out

    # ── even per-domain quotas that sum to num_samples ────────────────────
    def _domain_quotas(self, num_samples):
        n = len(self.domains)
        base = num_samples // n
        rem = num_samples - base * n
        quotas = {d: base for d in self.domains}
        # Distribute the remainder deterministically (round-robin over fixed order),
        # rotating by how many groups we've already served so the same domains
        # don't always eat the remainder.
        start = self.sample_group_index % n if n else 0
        for j in range(rem):
            quotas[self.domains[(start + j) % n]] += 1
        return quotas

    def get_samples(self, num_samples):
        # Serve from the RL buffer first (no-op for SFT where buffer stays empty),
        # then top up with balanced draws from the domain buckets.
        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)
        if num_samples == 0:
            return samples

        quotas = self._domain_quotas(num_samples)
        chosen_idxs = []
        for d in self.domains:
            chosen_idxs.extend(self._next_from_bucket(d, quotas[d]))

        # Interleave across domains so a batch isn't [all-if, all-search, ...];
        # keeps per-micro-batch domain mixing when the trainer splits the batch.
        chosen_idxs = self._interleave(chosen_idxs, quotas)

        for idx in chosen_idxs:
            prompt_sample = self.dataset.origin_samples[idx]
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def _interleave(self, chosen_idxs, quotas):
        # chosen_idxs is domain-blocked in self.domains order; rebuild as a
        # round-robin so consecutive samples come from different domains.
        blocks = {}
        pos = 0
        for d in self.domains:
            q = quotas[d]
            blocks[d] = chosen_idxs[pos : pos + q]
            pos += q
        out = []
        i = 0
        remaining = True
        while remaining:
            remaining = False
            for d in self.domains:
                if i < len(blocks[d]):
                    out.append(blocks[d][i])
                    remaining = True
            i += 1
        return out

    # ── resume: persist every bucket offset + epoch ───────────────────────
    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return
        state_dict = {
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "bucket_offsets": self.bucket_offsets,
            "bucket_epoch": self.bucket_epoch,
            "domains": self.domains,
            "metadata": self.metadata,
        }
        path = os.path.join(self.args.save, f"rollout/balanced_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return
        if self.args.load is None:
            return
        path = os.path.join(self.args.load, f"rollout/balanced_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist.")
            return
        logger.info(f"load balanced dataset metadata from {path}")
        state_dict = torch.load(path)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})

        saved_domains = state_dict.get("domains", self.domains)
        if saved_domains != self.domains:
            logger.warning(
                f"[BalancedRolloutDataSource] resumed domain order {saved_domains} "
                f"differs from current {self.domains}; not restoring bucket offsets."
            )
            return
        # Reshuffle each bucket to its saved epoch, then restore its offset, so
        # the exact same records are served next as before the checkpoint.
        for d in self.domains:
            ep = state_dict.get("bucket_epoch", {}).get(d, 0)
            self._shuffle_bucket(d, epoch=ep)
            self.bucket_offsets[d] = state_dict.get("bucket_offsets", {}).get(d, 0)
            self.bucket_epoch[d] = ep

    def __len__(self):
        return len(self.dataset)
