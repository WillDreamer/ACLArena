#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
# pkill -9 ray
# pkill -9 python

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
MODEL_ROOT=/data/user
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-30B-A3B.sh"
WANDB_API_KEY="${WANDB_API_KEY}"
ROLLOUT_BATCH_SIZE=64
GLOBAL_BATCH_SIZE=320

## v3 加大kl 0.1,答对但没有toolcall降到0.2
WANDB_GROUP="search_Qwen3-30B-A3B_tis_bs_${ROLLOUT_BATCH_SIZE}_memory_qwen_gspo_sft_light_80_mask_4k"
ROLLOUT_DEBUG_DIR="${ROOT_DIR}/multi_stage_rl_log/${WANDB_GROUP}"


GPU_LIST=(0 1 2 3 4 5 6 7)  
CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
export CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${#GPU_LIST[@]}
echo "Detected ${NUM_GPUS} GPUs for this run"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen/Qwen3-30B-A3B-Base/
   --ref-load ${ROOT_DIR}/Qwen3-30B-A3B_base_math_sft_search_light/
   --load ${ROOT_DIR}/Qwen3-30B-A3B_base_math_sft_search_light/
   --save ${ROOT_DIR}/Qwen3-30B-A3B_base_math_sft_80_gspo_4k/
   --save-interval 20
   --save-retain-interval 60
   --finetune
   --start-rollout-id 0
)

# --finetune 的效果（参见 Megatron 的 checkpointing.py 第 1711 行）：
# 只加载模型权重，跳过 optimizer 和 lr scheduler 的状态恢复
# iteration 从 0 重新开始，不会接着 math 的 iteration 继续计数

ROLLOUT_ARGS=(
   --prompt-data ${ROOT_DIR}/Search-R1/data/nq_hotpotqa_train/train.parquet
   --input-key prompt
   --label-key reward_model
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 500
   # --override-opt-param-scheduler
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt 5
   --rollout-max-response-len 4096
   --rollout-temperature 1

   # eval args
   --eval-interval 25
   # --eval-prompt-data nq_test ${ROOT_DIR}/Search-R1/data/nq_hotpotqa_train/test.parquet@[0:3000]
   --eval-prompt-data nq_test ${ROOT_DIR}/ARLArena/datasets/data/searchR1_processed_direct/test_small.parquet
   --eval-input-key prompt
   --eval-label-key reward_model
   --n-samples-per-eval-prompt 1

   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"

   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data
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

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192 
)

GRPO_ARGS=(
   --advantage-estimator gspo
   --use-kl-loss
   --kl-loss-coef 0.15
   --kl-loss-type low_var_kl
   --entropy-coef 0.01
   --eps-clip 0.2
   --eps-clip-high 0.28

   # whether enabling TIS
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project Seq-train
   --wandb-group ${WANDB_GROUP}
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
)

SGLANG_ARGS=(
   # MoE related args
   --rollout-num-gpus-per-engine ${NUM_GPUS}
   --sglang-mem-fraction-static 0.6
   --sglang-ep-size 4
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   # --sglang-enable-dp-attention
   # --sglang-dp-size 8
)

FAULT_TOLERANCE_ARGS=(
   --use-fault-tolerance
   --rollout-health-check-interval 30
   --rollout-health-check-timeout 60
   --rollout-health-check-first-wait 120
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash

   --distributed-timeout-minutes 60 
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search_tools_qwen_sft.generate
   --custom-rm-path generate_with_search_tools_qwen_sft.reward_func
#    --custom-generate-function-path generate_with_search_tools_qwen_sft_no_drift.generate
# --custom-rm-path generate_with_search_tools_qwen_sft_no_drift.reward_func
# --custom-tis-function-path generate_with_search_tools_qwen_sft_no_drift.compute_mis_weights_with_cp_no_drift

   # Group-gated loss-mask: when >50% of a group has no real tool execution,
   # zero out the loss_mask of the no-tool samples (they still count toward
   # group-relative advantage). See no_tool_loss_mask_filter.py.
   --dynamic-sampling-filter-path no_tool_loss_mask_filter.no_tool_loss_mask_filter

   # TIS-related args, recommended to enable when using TIS
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# export PYTHONPATH="${ROOT_DIR}/slime:${ROOT_DIR}/Megatron-LM:${PYTHONPATH}"
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

# RUNTIME_ENV_JSON="{
#   \"env_vars\": {
#     \"PYTHONPATH\": \"${ROOT_DIR}/Megatron-LM/:${SCRIPT_DIR}\",
#     \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
#   }
# }"

# ray job submit --address="http://127.0.0.1:8265" \
#    --runtime-env-json="${RUNTIME_ENV_JSON}" \
#    -- python3 train.py \

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
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${FAULT_TOLERANCE_ARGS[@]} \
   2>&1 | tee "${LOG_FILE}"
