#!/bin/bash

# Rollout-only script: use Qwen3-30B-A3B (instruct) to generate trajectories
# and save them for later filtering + SFT.
# No training is performed (--debug-rollout-only).

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3

set -ex

# will prevent ray from buffering stdout/stderr
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
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-32B.sh"

ROLLOUT_BATCH_SIZE=64
# rollout-only 模式下 global-batch-size 不影响训练，但框架仍需要此参数
GLOBAL_BATCH_SIZE=320

ROLLOUT_DEBUG_DIR="${MODEL_ROOT}/MultiStageRL/rollout_only_search_Qwen3-32B"

GPU_LIST=(0 1 2 3 4 5 6 7)  # <<<------  which GPUs to use, directly fill here
CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
export CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${#GPU_LIST[@]}
echo "Detected ${NUM_GPUS} GPUs for this run"

CKPT_ARGS=(
   # hf-checkpoint 用于获取 tokenizer 和模型 config
   --hf-checkpoint ${MODEL_ROOT}/Qwen3-32B/
   # ref-load: rollout 引擎加载的模型权重 (HF 格式)
   --ref-load ${MODEL_ROOT}/Qwen/Qwen3-30B-A3B/
   # debug-rollout-only 模式下 --load/--save 不需要，但框架可能仍需要 --load
   --load ${MODEL_ROOT}/Qwen/Qwen3-30B-A3B/
   --finetune
   --start-rollout-id 0
)

ROLLOUT_ARGS=(
   --prompt-data ${ROOT_DIR}/Search-R1/data/nq_hotpotqa_train/train.parquet
   --input-key prompt
   --label-key reward_model
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt 5
   --rollout-max-response-len 3072
   --rollout-temperature 1

   # 保存每个 iteration 的 rollout 数据
   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"

   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data

   # rollout-only 核心参数
   --debug-rollout-only
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
   --max-tokens-per-gpu 9216
)

SGLANG_ARGS=(
   # MoE related args
   --rollout-num-gpus-per-engine ${NUM_GPUS}
   --sglang-mem-fraction-static 0.8
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
   --custom-generate-function-path generate_with_search_tools_qwen_sft_no_drift.generate
   --custom-rm-path generate_with_search_tools_qwen_sft_no_drift.reward_func
)

# Keep <think> on EVERY turn (do NOT strip) when generating SFT data, so the
# distilled trajectories teach "think before every action" and the SFT format
# stays consistent with what RL rollout produces. OFF here. Must be exported
# BEFORE `ray start` so the rollout actors inherit it.
# Read by generate_with_search_tools_qwen_sft_no_drift._should_strip_think.
export SEARCH_R1_STRIP_THINK=0

# launch the master node of ray in container
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

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --temp-dir ${MODEL_ROOT}/ray_temp

export RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=1
python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   --no-offload-rollout \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
