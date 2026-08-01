#!/usr/bin/env python3
"""Merge a LoRA sidecar adapter into the frozen SDFT base and export HuggingFace format.

Recipe (same one used to publish willamazon1/sdft-search-lora-iter20):
merge in MEGATRON space, THEN convert. This avoids having to un-fuse the
GQA-interleaved qkv and the gate/up split -- slime's convert_to_hf applies those
transforms uniformly to (base + delta) afterwards.

  base   : SDFT torch_dist ckpt  (frozen base, byte-identical across LoRA iters)
  adapter: <lora iter dir>/lora_adapter.safetensors  (+ lora_meta.json)
  delta  : W += scaling * (B @ A)   per LoRA'd module, added into the STACKED
           per-layer weight tensor sd["decoder.layers.<mod>.weight"][L]
  out    : HF safetensors dir (+ assets copied from --origin-hf-dir)

LayerNorm weights are deliberately untouched: the LoRA delta applies to the
post-LN matmul only.

Modes
-----
  --verify-only : only run the adapter health gate (no load of the 16GB base).
  (default)     : verify -> merge -> convert -> optional --upload-repo push.

The verify gate is NOT optional decoration. Every pre-2026-07-27 iter of this run
saved lora_B == exactly 0 (untrained; see memory lora-ckpt-save-untrained-bug),
and old pre-fix iter dirs live in the SAME S3 prefix as good ones. Publishing one
would mislabel the pre-RL model as an RL result.
"""

import argparse
import json
import os
import sys
import time

import safetensors.torch
import torch

# LoRA'd module suffixes -> they sit under decoder.layers.<L>.<suffix>
LORA_MODULE_SUFFIXES = (
    "self_attention.linear_qkv",
    "self_attention.linear_proj",
    "mlp.linear_fc1",
    "mlp.linear_fc2",
)


def load_adapter(iter_dir):
    """Load lora_adapter.safetensors + lora_meta.json from a LoRA iter dir."""
    apath = os.path.join(iter_dir, "lora_adapter.safetensors")
    mpath = os.path.join(iter_dir, "lora_meta.json")
    if not os.path.isfile(apath):
        sys.exit(
            f"FATAL: no lora_adapter.safetensors in {iter_dir}\n"
            "  This iter is from the PRE-FIX lineage (only *.distcp shards, whose LoRA\n"
            "  factors are mis-sharded to 1/4 shape and unrecoverable). NOT publishable.\n"
            "  Pick an iter dir that contains the sidecar."
        )
    adapter = safetensors.torch.load_file(apath)
    meta = {}
    if os.path.isfile(mpath):
        with open(mpath) as f:
            meta = json.load(f)
    else:
        print(f"WARN: {mpath} missing; will fall back to --default-scaling")
    return adapter, meta


def verify_adapter(adapter, meta, strict=True):
    """Fail-fast gate: is this adapter actually TRAINED?

    lora_B is initialised to exactly 0, and d(loss)/d(lora_A) is proportional to
    lora_B, so an untrained adapter has lora_B == 0 bit-exactly and lora_A still
    at its kaiming init. Adam normalises each step to ~lr, so ANY real gradient
    over even a handful of steps moves lora_B off zero. lora_B == 0 therefore
    means zero gradient ever reached the adapter.
    """
    pairs = {}
    for k in adapter:
        if k.endswith(".lora_A"):
            pairs.setdefault(k[: -len(".lora_A")], {})["A"] = adapter[k]
        elif k.endswith(".lora_B"):
            pairs.setdefault(k[: -len(".lora_B")], {})["B"] = adapter[k]

    print(f"\n== adapter health gate ==")
    print(f"tensors={len(adapter)}  modules={len(pairs)}  meta_iter={meta.get('iteration', '?')}")

    n_zero_B, n_total, b_amax_global = 0, 0, 0.0
    bad_shape = []
    for name, d in sorted(pairs.items()):
        if "A" not in d or "B" not in d:
            bad_shape.append(f"{name}: missing {'A' if 'A' not in d else 'B'}")
            continue
        A, B = d["A"].float(), d["B"].float()
        # B (out, r) @ A (r, in) must be a valid matmul
        if A.ndim != 2 or B.ndim != 2 or B.shape[1] != A.shape[0]:
            bad_shape.append(f"{name}: B{tuple(B.shape)} @ A{tuple(A.shape)} not conformable")
            continue
        n_total += 1
        b_amax = B.abs().max().item()
        b_amax_global = max(b_amax_global, b_amax)
        if b_amax == 0.0:
            n_zero_B += 1
        if not (torch.isfinite(A).all() and torch.isfinite(B).all()):
            bad_shape.append(f"{name}: non-finite values")

    print(f"modules with conformable A/B : {n_total}")
    print(f"lora_B all-zero (UNTRAINED)  : {n_zero_B}/{n_total}")
    print(f"max |lora_B| over all modules: {b_amax_global:.3e}")
    if bad_shape:
        print("SHAPE/FINITENESS PROBLEMS:")
        for s in bad_shape[:20]:
            print("  " + s)

    if bad_shape:
        sys.exit("FATAL: adapter has malformed tensors (see above).")
    if n_total == 0:
        sys.exit("FATAL: no usable LoRA module pairs found in adapter.")
    if n_zero_B == n_total:
        sys.exit(
            "FATAL: EVERY lora_B is bit-exactly 0 => adapter is UNTRAINED.\n"
            "  Merging would reproduce the SDFT base exactly and publishing it would\n"
            "  mislabel the pre-RL model as an RL result. Do NOT publish.\n"
            "  (Root cause history: TIS rejection-sampling veto zeroed the whole batch;\n"
            "   only the TIS-OFF lineage produces nonzero lora_B.)"
        )
    if n_zero_B > 0 and strict:
        sys.exit(
            f"FATAL: {n_zero_B}/{n_total} modules have lora_B == 0 (partially untrained).\n"
            "  Pass --allow-partial-zero if this is genuinely expected."
        )
    print("GATE PASSED: adapter is trained (lora_B nonzero).\n")
    return b_amax_global


def merge_into_base(state_dict, adapter, meta, default_scaling, num_layers):
    """Add scaling*(B@A) into the stacked base weights, in-place."""
    scaling_map = meta.get("scaling", {})
    merged, deltas = 0, []
    for suffix in LORA_MODULE_SUFFIXES:
        wkey = f"decoder.layers.{suffix}.weight"
        if wkey not in state_dict:
            sys.exit(f"FATAL: base state_dict missing {wkey}. Keys look like:\n  " +
                     "\n  ".join(list(state_dict)[:20]))
        W = state_dict[wkey]
        if W.shape[0] != num_layers:
            sys.exit(f"FATAL: {wkey} first dim {W.shape[0]} != num_layers {num_layers}")
        for L in range(num_layers):
            base_key = f"decoder.layers.{L}.{suffix}"
            A = adapter.get(base_key + ".lora_A")
            B = adapter.get(base_key + ".lora_B")
            if A is None or B is None:
                sys.exit(f"FATAL: adapter missing {base_key}.lora_A/.lora_B")
            scaling = float(scaling_map.get(base_key, default_scaling))
            delta = (B.float() @ A.float() * scaling)
            if delta.shape != W[L].shape:
                sys.exit(
                    f"FATAL: delta {tuple(delta.shape)} != base slice {tuple(W[L].shape)} "
                    f"at {base_key}. A partial/mis-sharded adapter would look like this."
                )
            if not torch.isfinite(delta).all():
                sys.exit(f"FATAL: non-finite delta at {base_key}")
            wnorm = W[L].float().norm().item()
            dnorm = delta.norm().item()
            deltas.append((base_key, dnorm / wnorm if wnorm else float("inf")))
            W[L] = (W[L].float() + delta).to(W.dtype)
            merged += 1
        state_dict[wkey] = W
    rels = sorted(d for _, d in deltas)
    print(f"merged {merged} LoRA deltas into {len(LORA_MODULE_SUFFIXES)} stacked tensors")
    print(f"relative ||delta||/||W||  min={rels[0]:.3e}  median={rels[len(rels)//2]:.3e}  max={rels[-1]:.3e}")
    if rels[-1] == 0.0:
        sys.exit("FATAL: all deltas are zero after merge -- refusing to publish a no-op.")
    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-iter-dir", required=True,
                   help="LoRA iter dir containing lora_adapter.safetensors (e.g. .../iter_0000160)")
    p.add_argument("--base-torch-dist-dir",
                   help="SDFT base torch_dist iter dir (e.g. .../sdft_init/iter_0001000)")
    p.add_argument("--origin-hf-dir", help="HF dir for config/tokenizer assets")
    p.add_argument("--output-dir", help="where to write merged HF model")
    p.add_argument("--vocab-size", type=int, default=151936)
    p.add_argument("--chunk-size", type=int, default=5 * 1024**3)
    p.add_argument("--default-scaling", type=float, default=2.0,
                   help="alpha/r fallback if lora_meta.json absent (alpha=32,r=16 -> 2.0)")
    p.add_argument("--verify-only", action="store_true", help="run the adapter gate then exit")
    p.add_argument("--allow-partial-zero", action="store_true")
    p.add_argument("--upload-repo", default=None, help="HF repo id to push to, e.g. willamazon1/foo")
    p.add_argument("--private", action="store_true", help="create the HF repo private")
    p.add_argument("--force", action="store_true", help="overwrite output dir")
    args = p.parse_args()

    adapter, meta = load_adapter(args.adapter_iter_dir)
    verify_adapter(adapter, meta, strict=not args.allow_partial_zero)
    if args.verify_only:
        print("verify-only: done.")
        return

    for req in ("base_torch_dist_dir", "origin_hf_dir", "output_dir"):
        if not getattr(args, req):
            sys.exit(f"FATAL: --{req.replace('_', '-')} is required unless --verify-only")
    if os.path.exists(args.output_dir) and not args.force:
        sys.exit(f"FATAL: {args.output_dir} exists; pass --force")

    # Imported here so --verify-only works without megatron/slime on PYTHONPATH.
    import torch.distributed.checkpoint as dist_cp
    from transformers import AutoConfig
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    import tools.convert_torch_dist_to_hf as cvt

    print(f"== loading base torch_dist: {args.base_torch_dist_dir} ==")
    t = time.time()
    common = torch.load(os.path.join(args.base_torch_dist_dir, "common.pt"), weights_only=False)
    megatron_args = common["args"]
    state_dict = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=cvt.WrappedStorageReader(args.base_torch_dist_dir),
        planner=cvt.EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    print(f"base loaded in {time.time()-t:.1f}s, {len(state_dict)} tensors, "
          f"num_layers={megatron_args.num_layers}")

    # If the base dir IS a LoRA iter dir (recommended: its frozen base weights are
    # byte-identical to the SDFT init, which removes any doubt about WHICH SDFT iter
    # the run loaded), its .distcp also carries the stale in-band lora_A/lora_B keys.
    # Those are the mis-sharded/zero ones -- we merge from the sidecar instead, and
    # convert_to_hf has no mapping for them, so drop them before conversion.
    stale = [k for k in state_dict if ".lora_A" in k or ".lora_B" in k]
    for k in stale:
        del state_dict[k]
    if stale:
        print(f"stripped {len(stale)} stale in-band lora_* keys from the base state_dict "
              f"(merging from the sidecar instead)")

    print("== merging LoRA delta in Megatron space ==")
    merge_into_base(state_dict, adapter, meta, args.default_scaling, megatron_args.num_layers)

    hf_config = AutoConfig.from_pretrained(args.origin_hf_dir, trust_remote_code=True)
    model_name = type(hf_config).__name__.lower()
    print(f"== converting to HF ({model_name}) -> {args.output_dir} ==")
    cvt.save_tensors(megatron_args, model_name, state_dict, args.output_dir,
                     args.chunk_size, args.vocab_size)
    cvt.copy_assets(args.origin_hf_dir, args.output_dir)

    # GOTCHA seen last time: a stale partial-download shard in origin_hf_dir
    # (e.g. model-00004-of-00005.safetensors.7Ec15561) gets blindly copied by
    # copy_assets and is not in the index. Drop anything not in the fresh index.
    idx_path = os.path.join(args.output_dir, "model.safetensors.index.json")
    with open(idx_path) as f:
        wanted = set(json.load(f)["weight_map"].values())
    for fn in sorted(os.listdir(args.output_dir)):
        if ".safetensors" in fn and fn not in wanted and fn != "model.safetensors.index.json":
            full = os.path.join(args.output_dir, fn)
            print(f"removing stray shard not in index: {fn}")
            os.remove(full)

    print("== sanity check merged output ==")
    n_nan = 0
    for fn in sorted(wanted):
        tensors = safetensors.torch.load_file(os.path.join(args.output_dir, fn))
        for k, v in tensors.items():
            if not torch.isfinite(v).all():
                n_nan += 1
                print(f"  NON-FINITE: {fn}:{k}")
        del tensors
    if n_nan:
        sys.exit(f"FATAL: {n_nan} non-finite tensors in output; refusing to upload.")
    print(f"0 non-finite tensors across {len(wanted)} shards")

    if args.upload_repo:
        from huggingface_hub import HfApi, create_repo
        tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not tok:
            sys.exit("FATAL: HF_TOKEN not set")
        api = HfApi(token=tok)
        print("HF user:", api.whoami().get("name"))
        create_repo(args.upload_repo, token=tok, private=args.private,
                    exist_ok=True, repo_type="model")
        print(f"uploading {args.output_dir} -> {args.upload_repo} "
              f"({'private' if args.private else 'PUBLIC'})")
        api.upload_folder(
            folder_path=args.output_dir,
            repo_id=args.upload_repo,
            repo_type="model",
            commit_message=f"SDFT base + merged search-R1 LoRA adapter "
                           f"(iter {meta.get('iteration', '?')})",
        )
        print(f"UPLOAD DONE: https://huggingface.co/{args.upload_repo}")
    else:
        print("no --upload-repo given; merged HF model left at", args.output_dir)

    print("ALL DONE")


if __name__ == "__main__":
    main()
