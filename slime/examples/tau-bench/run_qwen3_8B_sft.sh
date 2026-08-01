#!/bin/bash

# SFT training on high-quality tau-bench trajectories.
#
# Prerequisites:
#   1. Run run_qwen3_30B_rollout_only.sh to collect trajectories.
#   2. Run parse_rollout_to_sft.py to filter and convert to JSONL:
      #  python parse_rollout_to_sft.py \
      #    --input-dir  /shared/user/MultiStageRL/rollout_only_tau_Qwen3-32B \
      #    --output-file /shared/user/MultiStageRL/rollout_only_tau_Qwen3-32B/sft_data.jsonl \
      #    --reward-threshold 1.0
#   3. Then run this script.

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python

set -ex

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

ROOT_DIR=/data/user
MODEL_ROOT=/shared/user
WANDB_API_KEY="${WANDB_API_KEY}"
SFT_DATA_DIR="${MODEL_ROOT}/MultiStageRL/rollout_only_tau_Qwen3-32B"
WANDB_GROUP="tau-bench_Qwen3-8B_sft"

GPU_LIST=(0 1 2 3 4 5 6 7)
CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
export CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${#GPU_LIST[@]}
echo "Detected ${NUM_GPUS} GPUs for this run"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-8B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen3-8B-Base/
   --load ${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search
   --save ${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT
   --save-interval 20
   --save-retain-interval 20
   --finetune
   --start-rollout-id 0
)

SFT_ARGS=(
   # Custom rollout function: reads pre-tokenized trajectories, no generation
   --rollout-function-path sft_rollout_from_rollout.generate_rollout
   --prompt-data ${SFT_DATA_DIR}/sft_data.jsonl
   --input-key data
   --rollout-global-dataset
   --rollout-shuffle
   --num-epoch 1
   --rollout-batch-size 64
   --global-batch-size 64

   # SFT loss: plain cross-entropy on masked tokens, no RL advantage
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns

   # Skip SGLang (no generation needed)
   --debug-train-only
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project Seq-train-8B
   --wandb-group ${WANDB_GROUP}
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PYTHONPATH="${ROOT_DIR}/slime:${ROOT_DIR}/Megatron-LM:${SCRIPT_DIR}:${PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_TMPDIR="/data1/ray_out"
rm -rf "$RAY_TMPDIR"
mkdir -p "$RAY_TMPDIR"

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${MISC_ARGS[@]}
