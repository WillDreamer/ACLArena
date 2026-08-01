#!/bin/bash

# Pure evaluation script: loads a checkpoint, runs eval once, logs to wandb, exits.
# Mechanism: train.py special-cases `--num-rollout 0 && --eval-interval != None`
# and calls rollout_manager.eval(rollout_id=0) then exits (see train.py:36-37).

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3

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
MODEL_ROOT=/data/user
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-30B-A3B.sh"
WANDB_API_KEY="${WANDB_API_KEY}"


EVAL_CKPT="${EVAL_CKPT:-${ROOT_DIR}/Qwen3-30B-A3B_base_math_sft_60_gspo/}"
EVAL_TAG="${EVAL_TAG:-$(basename ${EVAL_CKPT})}"

WANDB_GROUP="eval_${EVAL_TAG}"
ROLLOUT_DEBUG_DIR="${ROOT_DIR}/multi_stage_rl_log/${WANDB_GROUP}"


GPU_LIST=(0 1 2 3 4 5 6 7)
CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
export CUDA_VISIBLE_DEVICES
NUM_GPUS=${#GPU_LIST[@]}
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  NUM_GPUS=${NUM_GPUS}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen/Qwen3-30B-A3B-Base/
   --ref-load ${EVAL_CKPT}
   --load ${EVAL_CKPT}
   --finetune
   --start-rollout-id 0
)

# Pure eval: num-rollout=0 triggers the eval-only branch in train.py.
# eval-interval must be non-None (any int is fine — it's only checked for presence here).
ROLLOUT_ARGS=(
   # dummy prompt-data is still required by the arg parser; use the eval file.
   --prompt-data ${ROOT_DIR}/ARLArena/datasets/data/searchR1_processed_direct/test_small.parquet
   --input-key prompt
   --label-key reward_model
   --apply-chat-template
   --num-rollout 0
   --rollout-batch-size 64
   --n-samples-per-prompt 1
   --rollout-max-response-len 2048
   --rollout-temperature 0

   # eval args — this is the interesting part
   --eval-interval 1
   --eval-prompt-data nq_test ${ROOT_DIR}/ARLArena/datasets/data/searchR1_processed_direct/test.parquet
   --eval-input-key prompt
   --eval-label-key reward_model
   --n-samples-per-eval-prompt 1

   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/eval_rollout_{rollout_id}.pt"

   --global-batch-size 320
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project Seq-eval
   --wandb-group ${WANDB_GROUP}
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${NUM_GPUS}
   --sglang-mem-fraction-static 0.6
   --sglang-ep-size 4
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --distributed-timeout-minutes 60
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search_tools_qwen_sft.generate
   --custom-rm-path generate_with_search_tools_qwen_sft.reward_func
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PYTHONPATH="${ROOT_DIR}/slime:${ROOT_DIR}/Megatron-LM:${SCRIPT_DIR}:${PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
export RAY_memory_usage_threshold=0.99
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
export RAY_TMPDIR="/data1/ray_out"
rm -rf "$RAY_TMPDIR"
mkdir -p "$RAY_TMPDIR"
mkdir -p "$ROLLOUT_DEBUG_DIR"

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --temp-dir ${MODEL_ROOT}/ray_temp

export RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=1

LOG_FILE="${ROOT_DIR}/${WANDB_GROUP}.log"
echo "Logging to ${LOG_FILE}"
python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   --no-offload-train \
   --no-offload-rollout \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   2>&1 | tee "${LOG_FILE}"
