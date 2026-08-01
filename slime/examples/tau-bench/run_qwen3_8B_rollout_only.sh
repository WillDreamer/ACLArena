#!/bin/bash

# Rollout-only script for collecting tau-bench trajectories for SFT.
# Runs full tau-bench episodes and saves trajectories to .pt files.
# Does NOT train — use run_qwen3_30B_sft.sh after converting data.

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python

set -ex

export PYTHONBUFFERED=16

export OPENAI_API_BASE=http://131.179.168.120:8098/v1
export OPENAI_API_KEY=dummy

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

ROOT_DIR=/data/user
MODEL_ROOT=/shared/user
SFT_DATA_DIR="${MODEL_ROOT}/MultiStageRL/rollout_only_tau_Qwen3-32B"
ROLLOUT_BATCH_SIZE=64

GPU_LIST=(0 1 2 3 4 5 6 7)
CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
export CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${#GPU_LIST[@]}
echo "Detected ${NUM_GPUS} GPUs for this run"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-32B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen3-32B/
   --load ${MODEL_ROOT}/Qwen3-30B-A3B_torch_dist/
   --finetune
   --start-rollout-id 0
   --skip-eval-before-train True
)

ROLLOUT_ARGS=(
   --prompt-data ${ROOT_DIR}/tau-bench/retail_train_tasks.jsonl
   --input-key index
   --rollout-shuffle
   --num-rollout 500
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt 8
   --rollout-max-response-len 2048
   --rollout-max-context-len 131072
   --rollout-temperature 1
   --save-debug-rollout-data "${SFT_DATA_DIR}/rollout_{rollout_id}.pt"
   --debug-rollout-only
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.8
   # --sglang-ep-size 4
   # --sglang-max-total-tokens 1500000
   # --sglang-context-length 131072
   # --sglang-json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_tau.generate
)

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PYTHONPATH="${ROOT_DIR}/slime:${ROOT_DIR}/Megatron-LM:${SCRIPT_DIR}:${PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_TMPDIR="/data1/ray_out"
rm -rf "$RAY_TMPDIR"
mkdir -p "$RAY_TMPDIR"
mkdir -p "${SFT_DATA_DIR}"

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
