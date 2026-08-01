#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Multi-task SFT on Qwen3-8B — oracle_sft mixture (search + IF + math + tau).
#
# Data: oracle_sft_mix_capped.jsonl — 559,075 records, each {"messages": [...]}.
#   Built by merging + shuffling 4 oracle-SFT sources and normalizing them to a
#   single messages schema (search's rendered <|im_start|> prompt/response were
#   re-parsed into real turns). Length-capped to drop a ~0.04% >~20k-token tail
#   that would otherwise OOM a single GPU under first-fit micro-batch packing
#   (messages-format input skips slime's rollout_max_prompt_len filter).
#
# slime path: slime.rollout.sft_rollout.generate_rollout +
#   MultiTurnLossMaskGenerator(loss_mask_type=qwen3) tokenize + build the loss
#   mask at train time — masks system/user/tool tokens, trains only assistant
#   turns (multi-turn tool trajectories included). Plain single-node train-only
#   SFT: no rollout / no SGLang (--debug-train-only).
#
# Starting model: Qwen3-8B-Base (raw base, NOT the post-IF ckpt)
#   (staged from S3 as HF weights + a torch_dist ckpt for the Megatron load).
#
# Submit (single node, 8xH100/P5EN is enough for 8B train-only SFT):
#   cd /workspace/ACLArena
#   python3 cluster_cli_sdft.py batch \
#     --script examples/SDFT/run_qwen3_8b_oracle_mix_sft.sh \
#     --num-nodes 1 --rollout-nodes 0 \
#     --stage-model Qwen3/Qwen3-8B-Base/ \
#     --stage-model Qwen3/Qwen3-8B-Base_torch_dist/ \
#     --stage-data oracle_sft/
#   (SFT is train-only. The jsonl must land at ${SDFT_DATA_DIR}/oracle_sft_mix_capped.jsonl.)
# ─────────────────────────────────────────────────────────────────────────────

set -ex
export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi

# ROOT_DIR / MODEL_ROOT / DATA_ROOT are exported by the cluster bootstrap; the
# defaults here keep the script runnable on a dev box.
ROOT_DIR=${ROOT_DIR:-/data/user}
MODEL_ROOT=${MODEL_ROOT:-/data/user/models}
DATA_ROOT=${DATA_ROOT:-/data/user/data}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-8B.sh"   # MODEL_ARGS for Qwen3-8B
WANDB_API_KEY="${WANDB_API_KEY}"

GPU_LIST=(0 1 2 3 4 5 6 7)
CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
export CUDA_VISIBLE_DEVICES
NUM_GPUS=${#GPU_LIST[@]}

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Model name (staged under $MODEL_ROOT/Qwen3/); overridable via env.
# SFT starts from the raw Qwen3-8B-Base (not the post-IF ckpt).
BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen3-8B-Base}"
HF_CKPT="${MODEL_ROOT}/Qwen3/${BASE_MODEL_NAME}/"
TORCH_DIST="${MODEL_ROOT}/Qwen3/${BASE_MODEL_NAME}_torch_dist/"

SDFT_DATA_DIR="${SDFT_DATA_DIR:-${DATA_ROOT}/oracle_sft}"
SFT_DATA="${SFT_DATA:-${SDFT_DATA_DIR}/oracle_sft_mix_capped.jsonl}"
test -s "${SFT_DATA}" || { echo "FATAL: SFT data ${SFT_DATA} missing — stage oracle_sft/ to S3 first."; exit 1; }

CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --ref-load ${TORCH_DIST}
   # Fresh SFT: --load points at the same torch_dist base. On the first run the
   # dir has no latest_checkpointed_iteration.txt for THIS --save, so slime falls
   # back to finetune=True and loads weights (no optim/rng) from ref_load.
   --load ${TORCH_DIST}
   # MUST live under $MODEL_ROOT/ACLArena/ — the cluster bootstrap ONLY syncs
   # $MODEL_ROOT/ACLArena/ -> s3://YOUR_BUCKET/ACLArena/ (output_prefix). A --save
   # under any other subtree writes locally but is NEVER pushed to S3, so the
   # checkpoints are lost when the BatchService job ends.
   --save ${MODEL_ROOT}/ACLArena/Qwen3-8B-Base-oracle-mix-SFT/
   --save-interval 500
   --save-retain-interval 500
   --finetune
   --start-rollout-id 0
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data ${SFT_DATA}
   --input-key messages
   --rollout-global-dataset
   --rollout-shuffle
   --num-epoch 1
   --rollout-batch-size 128
   --global-batch-size 128
   --loss-type sft_loss
   --loss-mask-type qwen3
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 24576
   --log-probs-chunk-size 1024
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.03
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project ACLArena
   --wandb-group sdft_Qwen3-8B-Base_oracle_mix
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-bias-dropout-fusion
)

export PYTHONPATH="${SLIME_DIR:-${ROOT_DIR}/slime}:${MEGATRON_DIR:-${ROOT_DIR}/Megatron-LM}:${SCRIPT_DIR}:${PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"

if [ -z "${SKIP_SYS_LDPATH:-}" ]; then
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
fi

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats \
   --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir ${RAY_TEMP_DIR:-${ROOT_DIR}/ray_temp}

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"FI_EFA_FORK_SAFE\": \"1\"
  }
}"

# Train-only SFT on a single node: actor uses all GPUs, no rollout / no sglang.
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --num-gpus-per-node ${NUM_GPUS} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]}
