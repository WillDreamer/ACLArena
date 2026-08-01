"""Multi-teacher on-policy distillation reward module (SGLang teacher mode).

Also exports the EOPD (Entropy-Aware OPD, arXiv:2603.07079) variants
``reward_func_eopd`` / ``post_process_rewards_eopd``. Same domain-routed teacher
scoring, but additionally requests each teacher's per-position top-k logprobs so
the forward-KL term (loss.compute_eopd_forward_kl) has the teacher top-k
distribution + per-token entropy it consumes. Mirrors the single-teacher pair in
slime.rollout.on_policy_distillation; keeps the multi-teacher aborted-sample
fail-soft guard so a bad teacher call drops the sample instead of killing the run.


Pure distillation: the task reward is always 0.0; the only learning signal is the
OPD KL penalty applied in slime/backends/megatron_utils/loss.py:

    advantage -= opd_kl_coef * (student_log_prob - teacher_log_prob)   # per token

Each student rollout is scored by the teacher that OWNS its domain (domain-routing,
*not* averaging) — a Math teacher gives meaningless logprobs on a Tau trajectory,
a Search teacher on a Math trajectory, etc. The three teachers are the specialists
from the math2sea / sea2tau / tau2if experiments:

    sample.metadata["domain"] == "math"    ->  Math   teacher  (Qwen3-8B-Base-Math)
    sample.metadata["domain"] == "search"  ->  Search teacher  (Qwen3-8B-Base-Math-SeaSFT-Search)
    sample.metadata["domain"] == "tau"     ->  Tau    teacher  (Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau)

Distilling all three back into ONE student (the end-of-chain
Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau-IF, which has drifted from every earlier
specialty) recovers math + search + tau simultaneously.

The three teachers are served IN-CLUSTER as frozen (update_weights:false) models in the
run's --sglang-config, alongside the actor (student rollout) and the tau user_sim. Each
frozen model gets its OWN sglang router; we resolve its /generate URL by model name via
slime's get_model_url(args, name, "/generate") — no hardcoded IPs, works on Cluster
where there is no standalone service node:

    sample.metadata["domain"] == "math"    ->  model "teacher_math"
    sample.metadata["domain"] == "search"  ->  model "teacher_search"
    sample.metadata["domain"] == "tau"     ->  model "teacher_tau"

This mirrors slime/rollout/on_policy_distillation.py (the single-teacher example);
the only addition is per-sample teacher selection.
"""

import os

import aiohttp
import torch

from slime.rollout.sglang_rollout import get_model_url
from slime.utils.types import Sample

# domain -> the frozen teacher model NAME in the --sglang-config (resolved to a
# router URL at call time via get_model_url; see reward_func).
_DOMAIN_TO_MODEL = {
    "math": "teacher_math",
    "search": "teacher_search",
    "tau": "teacher_tau",
}

# Teacher prefill over a long multi-turn trajectory can be slow; give it room.
_TIMEOUT = aiohttp.ClientTimeout(total=float(os.environ.get("OPD_TEACHER_TIMEOUT", "600")))


def _domain_of(sample: Sample) -> str:
    domain = (sample.metadata or {}).get("domain")
    if domain not in _DOMAIN_TO_MODEL:
        raise ValueError(
            f"sample.metadata['domain'] must be one of {list(_DOMAIN_TO_MODEL)}, got {domain!r}. "
            "Every row of the mixed dataset must be tagged with metadata.domain "
            "(see prepare_opd_mixed_data.py)."
        )
    return domain


async def _score_trajectory(args, sample, extra_payload=None):
    """Route to the domain-owning teacher and score the student's own trajectory.

    ``max_new_tokens=0`` means we only *score* the given ``input_ids`` — no generation
    happens on the teacher. Returns the teacher's JSON (stored on ``sample.reward`` and
    consumed in post_process), or ``None`` on any teacher error (fail-soft). Extra sglang
    request fields (e.g. ``top_logprobs_num`` for EOPD) go in ``extra_payload``.
    """
    model_name = _DOMAIN_TO_MODEL[_domain_of(sample)]
    # get_model_url SILENTLY falls back to the default (actor) router when the name is
    # absent — that would score the trajectory with the STUDENT, not the teacher. Guard.
    routers = getattr(args, "sglang_model_routers", None) or {}
    assert model_name in routers, (
        f"'{model_name}' not in --sglang-config routers {sorted(routers)}; teacher scoring "
        "would silently fall back to the actor. Add the frozen teacher models to the config."
    )
    url = get_model_url(args, model_name, "/generate")
    payload = {
        "input_ids": sample.tokens,  # full prompt + multi-turn response (exact student token ids)
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    if extra_payload:
        payload.update(extra_payload)
    # RESILIENT teacher scoring (job <JOB_ID> fix). A single bad scoring call must NOT kill
    # the whole job. The observed failure: a long search/tau trajectory (prompt + up to 8192
    # response) exceeded the teacher's context_length, so sglang returned HTTP 400
    # ("input (19931 tokens) is longer than the model's context length") and the old
    # raise_for_status propagated up through async_rm -> generate_and_rm_group ->
    # train.py and crashed at step 4. context_length is now sized to the student's (32768),
    # but we ALSO fail-soft here: on ANY teacher error (4xx/5xx/timeout) return None. The
    # post_process guard then treats this sample as unscored -> zero teacher_log_probs +
    # remove_sample (loss_mask zeroed), so it contributes nothing instead of killing the run.
    try:
        async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status >= 400:
                    body = (await resp.text())[:300]
                    print(
                        f"[opd_multi_teacher] teacher '{model_name}' scoring HTTP {resp.status} "
                        f"for a {len(sample.tokens)}-token trajectory (likely over context_length); "
                        f"dropping this sample from the batch. body={body!r}",
                        flush=True,
                    )
                    return None
                return await resp.json()
    except Exception as e:  # noqa: BLE001 — network/timeout: drop the sample, don't crash the job
        print(
            f"[opd_multi_teacher] teacher '{model_name}' scoring failed "
            f"({type(e).__name__}: {e}) for a {len(sample.tokens)}-token trajectory; "
            f"dropping this sample from the batch.",
            flush=True,
        )
        return None


async def reward_func(args, sample, **kwargs):
    """Fetch the domain teacher's scalar token-level logprobs (reverse-KL OPD path)."""
    return await _score_trajectory(args, sample)


async def reward_func_eopd(args, sample, **kwargs):
    """EOPD variant: same domain-routed scoring as reward_func, but also request the
    teacher's per-position top-k logprobs (``top_logprobs_num``) so post_process can build
    the teacher top-k distribution + entropy for the forward-KL term. See
    slime.rollout.on_policy_distillation.reward_func_eopd (single-teacher original)."""
    topk = getattr(args, "eopd_topk", 16)
    return await _score_trajectory(args, sample, {"top_logprobs_num": topk})


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Write teacher logprobs onto each sample; return 0.0 task reward (pure OPD).

    The teacher scores the *entire* trajectory; we keep the response span (the last
    ``response_length`` tokens), which aligns 1:1 with ``sample.loss_mask``. Tokens
    that are masked out (tool/observation turns) still carry a teacher logprob here,
    but contribute nothing because the policy loss multiplies by loss_mask.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    for sample, reward, response_length in zip(samples, raw_rewards, response_lengths, strict=False):
        # ABORTED samples were NEVER scored by the teacher: generate_and_rm early-returns
        # on status==ABORTED (sglang_rollout.py) BEFORE calling reward_func, so sample.reward
        # is still None here (the dispatcher cleared the tau task reward). Indexing
        # reward["meta_info"] on that None crashed job <JOB_ID> with
        # "TypeError: 'NoneType' object is not subscriptable" once the local GLM user-sim
        # started failing (empty completions -> ABORTED tau trajectories). An unscored
        # sample carries no teacher signal, so give it a zero-length teacher_log_probs and
        # flag remove_sample: rollout._convert_samples_to_train_data then zeroes its
        # loss_mask, so it contributes NOTHING to the OPD KL / policy loss — the same
        # slime-native "drop without breaking the GRPO group" mechanism the tau
        # degenerate-no-op path uses. (This is domain-agnostic: a math/search abort would
        # hit the identical crash.)
        if reward is None or not (isinstance(reward, dict) and reward.get("meta_info")):
            sample.teacher_log_probs = torch.zeros(response_length, dtype=torch.float32)
            sample.remove_sample = True
            continue
        # sglang: meta_info["input_token_logprobs"] = [[logprob, token_id, ...], ...];
        # the first entry has no logprob (no preceding context), so skip [1:].
        t_log_probs = torch.tensor(
            [item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]],
            dtype=torch.float32,
        )
        sample.teacher_log_probs = t_log_probs[-response_length:]

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards


def post_process_rewards_eopd(args, samples: list[Sample], **kwargs):
    """EOPD post-process: everything post_process_rewards does (scalar teacher log-probs
    -> reverse-KL advantage path, task reward 0.0) PLUS the teacher's per-token top-k
    distribution + entropy for the forward-KL term:

      sample.teacher_topk_ids       [response_len, K]  int token ids
      sample.teacher_topk_log_probs [response_len, K]  teacher log-probs (raw)
      sample.teacher_entropy        [response_len]     approx teacher entropy (top-k mass)

    Mirrors slime.rollout.on_policy_distillation.post_process_rewards_eopd, but keeps the
    multi-teacher fail-soft guard: an ABORTED/dropped sample (reward None, or reward with
    no meta_info) gets a zero-length teacher_log_probs + remove_sample and NO top-k fields,
    so it contributes nothing instead of crashing on reward["meta_info"] indexing.
    """
    topk = getattr(args, "eopd_topk", 16)
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]
    pad_lp = -1e9  # padded slot: exp(-1e9)=0 -> zero weight in teacher dist / entropy / gather

    for sample, reward, rl in zip(samples, raw_rewards, response_lengths, strict=False):
        # Same drop-without-crash guard as post_process_rewards (aborted tau/search/math).
        # For EOPD we can't leave the top-k fields as None here: actor.process_rollout_data
        # iterates the whole per-sample list unconditionally (t.to(device, dtype=long)) and
        # would AttributeError on None. Provide zero-mass placeholders — very-negative
        # log-probs so exp()->0 contributes zero weight to the forward-KL term, and entropy
        # 0 so the gate (H > tau) never fires. loss_mask is zeroed elsewhere via
        # remove_sample so these tokens contribute nothing to training either way.
        if reward is None or not (isinstance(reward, dict) and reward.get("meta_info")):
            sample.teacher_log_probs = torch.zeros(rl, dtype=torch.float32)
            sample.teacher_topk_ids = torch.zeros(max(rl, 0), topk, dtype=torch.long)
            sample.teacher_topk_log_probs = torch.full(
                (max(rl, 0), topk), pad_lp, dtype=torch.float32
            )
            sample.teacher_entropy = torch.zeros(max(rl, 0), dtype=torch.float32)
            sample.remove_sample = True
            continue

        # scalar teacher log-probs for the reverse-KL advantage path (skip [0]: no context).
        t_log_probs = torch.tensor(
            [item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]],
            dtype=torch.float32,
        )
        sample.teacher_log_probs = t_log_probs[-rl:]

        # per-token top-k distribution + entropy for the forward-KL term.
        top = reward["meta_info"].get("input_top_logprobs")
        top = (top[1:] if top else [])
        top = top[-rl:] if rl > 0 else []
        ids_rows, lp_rows, ent_rows = [], [], []
        for entry in top:
            entry = entry or []
            lps = [e[0] for e in entry][:topk]
            ids = [int(e[1]) for e in entry][:topk]
            if len(lps) < topk:  # pad to exactly K so the batch is rectangular
                pad_n = topk - len(lps)
                lps = lps + [pad_lp] * pad_n
                ids = ids + [0] * pad_n
            lp_t = torch.tensor(lps, dtype=torch.float32)
            p_t = lp_t.exp()  # raw teacher probs of the top-k (do NOT sum to 1)
            ent_rows.append(float(-(p_t * lp_t).sum().item()))  # H over captured top-k mass
            ids_rows.append(ids)
            lp_rows.append(lps)

        if rl > 0 and ids_rows:
            sample.teacher_topk_ids = torch.tensor(ids_rows, dtype=torch.long)          # [R, K]
            sample.teacher_topk_log_probs = torch.tensor(lp_rows, dtype=torch.float32)  # [R, K]
            sample.teacher_entropy = torch.tensor(ent_rows, dtype=torch.float32)        # [R]
        else:
            # rl==0 (or teacher returned no top-k rows). Set zero-shape tensors — NOT None —
            # so downstream .to(device, dtype=long) doesn't AttributeError (samples[0]
            # gates whether the batch has top-k, but per-sample iteration follows unconditionally).
            sample.teacher_topk_ids = torch.zeros(0, topk, dtype=torch.long)
            sample.teacher_topk_log_probs = torch.zeros(0, topk, dtype=torch.float32)
            sample.teacher_entropy = torch.zeros(0, dtype=torch.float32)

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
