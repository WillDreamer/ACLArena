#!/bin/bash
# Merge the search-R1 LoRA sidecar at iter_0000160 into the frozen SDFT base,
# export HuggingFace format, and push to the Hub. Runs ON THE SDB (needs the
# slime + Megatron env and ~40GB free disk; no GPU required -- the torch_dist
# loader uses no_dist=True).
#
# Prereqs on the sdb:
#   - AWS creds exported (sdb has NO aws CLI -> S3 via boto3 / s3sync_boto.py)
#   - HF_TOKEN exported
#   - slime repo at $SLIME_DIR, Megatron at $MEGATRON_DIR
#
# STEP 0 (from the laptop) open the tunnel, then ssh in:
#   aws ssm start-session --target i-INSTANCE_ID \
#     --document-name AWS-StartPortForwardingSessionToRemoteHost \
#     --parameters '{"portNumber":["22"],"localPortNumber":["1053"],"host":["10.2.83.121"]}' \
#     --profile cluster --region ap-south-1
#   ssh -p 1053 root@localhost
set -euxo pipefail
export PYTHONUNBUFFERED=1

SLIME_DIR="${SLIME_DIR:-/root/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-/root/Megatron-LM}"
WORK="${WORK:-/tmp/instance_storage/user/lora160}"

ITER="${ITER:-160}"
ITER_DIR_NAME="$(printf 'iter_%07d' "${ITER}")"
S3_BUCKET="${S3_BUCKET:-YOUR_BUCKET}"
S3_CKPT_PREFIX="${S3_CKPT_PREFIX:-MultiStageRL-LoRA-single/Qwen3-8B-SDFT-Search-LoRA/${ITER_DIR_NAME}}"
# HF assets (config.json / tokenizer / generation_config) -- weights are ignored.
S3_ORIGIN_HF="${S3_ORIGIN_HF:-Qwen3/Qwen3-8B-Base}"

CKPT_DIR="${WORK}/${ITER_DIR_NAME}"
ORIGIN_HF_DIR="${WORK}/origin_hf"
OUT_DIR="${WORK}/hf_export/sdft-search-lora-iter${ITER}"
REPO_ID="${HF_REPO_ID:-willamazon1/sdft-search-lora-iter${ITER}}"

mkdir -p "${WORK}"

echo "== 1. pull the checkpoint iter dir from S3 (sidecar + base .distcp) =="
python3 "${SYNC:-/root/s3sync_boto.py}" download "${S3_BUCKET}" "${S3_CKPT_PREFIX}" "${CKPT_DIR}"
ls -la "${CKPT_DIR}"

echo "== 2. GATE: is the adapter actually TRAINED? =="
# Hard-stops if lora_B is bit-exactly zero (every pre-TIS-off iter was) or if the
# sidecar is absent (pre-fix .distcp-only iters). Cheap: no base load.
python3 "${SLIME_DIR}/examples/search-r1/merge_lora_and_convert.py" \
  --adapter-iter-dir "${CKPT_DIR}" --verify-only

echo "== 3. pull HF assets for config/tokenizer =="
python3 "${SYNC:-/root/s3sync_boto.py}" download "${S3_BUCKET}" "${S3_ORIGIN_HF}" "${ORIGIN_HF_DIR}"
test -f "${ORIGIN_HF_DIR}/config.json"

echo "== 4. merge + convert to HF =="
# --base-torch-dist-dir = the SAME iter dir: its frozen base weights are
# byte-identical to the SDFT init (verified previously, max|diff|=0), which avoids
# depending on which SDFT iter (500 vs 1000) the run actually loaded. The stale
# in-band lora_* keys in that ckpt are stripped automatically.
cd "${SLIME_DIR}"
PYTHONPATH="${SLIME_DIR}:${MEGATRON_DIR}:${PYTHONPATH:-}" \
  python3 examples/search-r1/merge_lora_and_convert.py \
    --adapter-iter-dir "${CKPT_DIR}" \
    --base-torch-dist-dir "${CKPT_DIR}" \
    --origin-hf-dir "${ORIGIN_HF_DIR}" \
    --output-dir "${OUT_DIR}" \
    --vocab-size "${VOCAB_SIZE:-151936}" \
    --force

echo "== 5. smoke test the merged model =="
export OUT_DIR
python3 - <<'PY'
import os
from transformers import AutoConfig, AutoTokenizer
out = os.environ["OUT_DIR"]
c = AutoConfig.from_pretrained(out, trust_remote_code=True)
print("config OK:", type(c).__name__, "vocab", c.vocab_size, "layers", c.num_hidden_layers)
t = AutoTokenizer.from_pretrained(out, trust_remote_code=True)
print("tokenizer OK, vocab", len(t))
PY

echo "== 6. upload =="
# NOTE: PUBLIC, matching the prior sdft-search-lora-iter20 publish. Change
# private=False below to keep it private.
export OUT_DIR REPO_ID ITER
python3 - <<'PY'
import os
from huggingface_hub import HfApi, create_repo
tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
assert tok, "HF_TOKEN not set"
repo, out = os.environ["REPO_ID"], os.environ["OUT_DIR"]
api = HfApi(token=tok)
print("HF user:", api.whoami().get("name"))
create_repo(repo, token=tok, private=False, exist_ok=True, repo_type="model")
api.upload_folder(folder_path=out, repo_id=repo, repo_type="model",
                  commit_message=f"SDFT base + merged search-R1 LoRA (iter {os.environ['ITER']})")
print("UPLOAD DONE:", f"https://huggingface.co/{repo}")
PY

echo "ALL DONE"
