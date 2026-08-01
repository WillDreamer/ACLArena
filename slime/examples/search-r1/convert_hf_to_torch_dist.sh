#!/bin/bash
# HF -> Megatron torch_dist conversion, CLUSTER / single-node / 1 GPU.
#
# Downloads a HuggingFace model, converts it to the Megatron `torch_dist` format
# slime's --load/--ref-load require, and leaves the result under $MODEL_ROOT so
# the BatchService bootstrap's output-sync uploads it to S3.
#
# Defaults are baked for the search RL init:
#     HF repo  : willhx/Qwen3-8B-Base-Math-SeaSFT           (public safetensors)
#     out (S3) : s3://YOUR_BUCKET/ACLArena/Qwen3-8B-Base-Math-SeaSFT_torch_dist/
#
# Submit (single node, 1 GPU is enough for an 8B convert — ~30s once downloaded):
#     python3 cluster_cli_seq_search.py batch \
#         --script examples/search-r1/convert_hf_to_torch_dist.sh \
#         --num-nodes 1 --rollout-nodes 0 \
#         --stage-model none --stage-data none \
#         --output-prefix ACLArena/Qwen3-8B-Base-Math-SeaSFT_torch_dist/
#
# The conversion pulls the HF weights over the internet (snapshot_download), so it
# assumes the BatchService node has outbound access to huggingface.co. If it does not, set
# CONVERT_HF_S3 to an S3 prefix holding the raw HF checkpoint and it will pull from
# there instead (stage it via --stage-model <that-prefix> so it's on local NVMe).
set -ex

# ── Overridable roots (BatchService bootstrap sets these to the NVMe mirror; dev defaults kept) ──
ROOT_DIR=${ROOT_DIR:-/data/user}
MODEL_ROOT=${MODEL_ROOT:-/shared/user}
DATA_ROOT=${DATA_ROOT:-${ROOT_DIR}}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SLIME_DIR:-${ROOT_DIR}/slime}"

export PYTHONBUFFERED=16

# ── What to convert ──
HF_REPO=${CONVERT_HF_REPO:-willhx/Qwen3-8B-Base-Math-SeaSFT}   # HF hub id (or a local path)
MODEL_NAME=${CONVERT_MODEL_NAME:-Qwen3-8B-Base-Math-SeaSFT}    # local dir name for the raw HF copy
OUT_SUBPATH=${CONVERT_OUT_SUBPATH:-ACLArena/Qwen3-8B-Base-Math-SeaSFT_torch_dist}  # relative to MODEL_ROOT
MODEL_SH=${CONVERT_MODEL_SH:-qwen3-8B.sh}                      # scripts/models/<...> providing MODEL_ARGS

HF_LOCAL="${MODEL_ROOT}/${MODEL_NAME}"
OUT_DIR="${MODEL_ROOT}/${OUT_SUBPATH}"
mkdir -p "${MODEL_ROOT}" "${OUT_DIR%/*}"

# ── 1. Obtain the raw HF checkpoint on local NVMe ──
# Priority: (a) already present locally (e.g. staged from S3 to $HF_LOCAL),
#           (b) CONVERT_HF_S3 given -> already staged by the CLI to $HF_LOCAL,
#           (c) HF hub download via huggingface_hub.snapshot_download.
if [ -f "${HF_LOCAL}/config.json" ]; then
  echo "[convert] raw HF checkpoint already present at ${HF_LOCAL}; skipping download"
else
  echo "[convert] downloading ${HF_REPO} -> ${HF_LOCAL} via huggingface_hub.snapshot_download"
  HF_REPO="${HF_REPO}" HF_LOCAL="${HF_LOCAL}" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
repo = os.environ["HF_REPO"]
local = os.environ["HF_LOCAL"]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
# Skip *.bin when a safetensors index exists (halves egress); allow_patterns kept broad
# so tokenizer/config/merges/vocab always come along.
snapshot_download(
    repo_id=repo,
    local_dir=local,
    local_dir_use_symlinks=False,
    token=token,
    ignore_patterns=["*.pth", "*.msgpack", "*.h5", "original/*"],
)
print("[convert] downloaded:", sorted(os.listdir(local))[:40])
PY
  [ -f "${HF_LOCAL}/config.json" ] || { echo "FATAL: HF download did not produce ${HF_LOCAL}/config.json" >&2; exit 1; }
fi

# ── 2. Convert HF -> torch_dist ──
# MODEL_ARGS for the 8B arch (36 layers, hidden 4096, ffn 12288, 32 heads/8 KV,
# kv-channels 128, vocab 151936, qk-layernorm, untied). Same recipe used for the
# other Qwen3-8B converts; ~30s on one H200.
source "${SCRIPT_DIR}/../../scripts/models/${MODEL_SH}"

# Fresh output dir so a partial prior attempt can't confuse the release-rename step.
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

echo "[convert] converting ${HF_LOCAL} -> ${OUT_DIR}"
PYTHONPATH="${MEGATRON_DIR:-${ROOT_DIR}/Megatron-LM}:${PYTHONPATH:-}" \
python3 tools/convert_hf_to_torch_dist.py \
   "${MODEL_ARGS[@]}" \
   --hf-checkpoint "${HF_LOCAL}" \
   --save "${OUT_DIR}"

# ── 3. Sanity-check the produced checkpoint ──
# Expected layout mirrors the other *_torch_dist dirs: a "release" tracker +
# release/{*.distcp, common.pt, metadata.json, .metadata}.
if [ ! -s "${OUT_DIR}/latest_checkpointed_iteration.txt" ] || [ ! -d "${OUT_DIR}/release" ]; then
  echo "FATAL: conversion output looks wrong (no release/ or tracker) at ${OUT_DIR}" >&2
  ls -la "${OUT_DIR}" || true
  exit 1
fi
echo "[convert] DONE. torch_dist checkpoint at ${OUT_DIR}:"
ls -la "${OUT_DIR}" "${OUT_DIR}/release" || true
echo "[convert] the BatchService bootstrap output-sync will upload ${OUT_SUBPATH}/ to S3 (final sync on exit)."
