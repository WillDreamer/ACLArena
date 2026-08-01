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
MODEL_ROOT=/shared/user
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-8B.sh"
WANDB_API_KEY="${WANDB_API_KEY}"
ROLLOUT_BATCH_SIZE=64
GLOBAL_BATCH_SIZE=512


WANDB_GROUP="search_bs_${ROLLOUT_BATCH_SIZE}"
ROLLOUT_DEBUG_DIR="${MODEL_ROOT}/MultiStageRL/${WANDB_GROUP}"


GPU_LIST=(0 1 2 3 4 5 6 7)  
CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
export CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${#GPU_LIST[@]}
echo "Detected ${NUM_GPUS} GPUs for this run"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen3-8B-Base/
   --ref-load ${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT
   --load ${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT
   --save ${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search
   --save-interval 20
   --save-retain-interval 60
   --finetune
   --start-rollout-id 0
)


ROLLOUT_ARGS=(
   --prompt-data ${ROOT_DIR}/Search-R1/data/nq_hotpotqa_train/train.parquet
   --input-key prompt
   --label-key reward_model
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 500
   # --override-opt-param-scheduler
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt 8
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
   --context-parallel-size 2

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
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
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
   # --optimizer-cpu-offload
   # --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project Seq-train-8B
   --wandb-group ${WANDB_GROUP}
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
)

SGLANG_ARGS=(
   # MoE related args
   --rollout-num-gpus-per-engine ${NUM_GPUS}
   --sglang-mem-fraction-static 0.7
   # --sglang-ep-size 4
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
   --custom-generate-function-path generate_with_search_tools_qwen_sft_no_drift.generate
   --custom-rm-path generate_with_search_tools_qwen_sft_no_drift.reward_func

   # Group-gated loss-mask: when >50% of a group has no real tool execution,
   # zero out the loss_mask of the no-tool samples (they still count toward
   # group-relative advantage). See no_tool_loss_mask_filter.py.
   --dynamic-sampling-filter-path no_tool_loss_mask_filter.no_tool_loss_mask_filter

   # TIS-related args, recommended to enable when using TIS.
   # 8B-specific copy so its thresholds can be tuned without touching the 30B run.
   --custom-config-path examples/train_infer_mismatch_helper/mis_8b.yaml
   # NOTE: this run uses the *_no_drift* generate fn with --apply-chat-template,
   # which writes placeholder rollout_log_prob=0.0 on all chat-template glue
   # (<|im_end|>\n, <|im_start|>user\n<tool_response>..., <|im_start|>assistant\n)
   # and tool_response tokens. Use the matching _no_drift TIS wrapper (NOT the
   # generic mis.compute_mis_weights_with_cp): it masks every rollout_log_prob==0.0
   # position before computing IS weights, so exp(train_lp - 0.0) garbage never
   # enters the gradient. Must pair with generate_with_search_tools_qwen_sft_no_drift.generate above.
   --custom-tis-function-path generate_with_search_tools_qwen_sft_no_drift.compute_mis_weights_with_cp_no_drift
)

# Strip historical-turn <think> blocks during rollout + training (Qwen3
# multi-turn convention; compresses prefill context). Must be exported
# BEFORE `ray start` so the rollout actors inherit it.
# Read by generate_with_search_tools_qwen_sft_no_drift._should_strip_think.
#
# OFF: with strip_think the rollout prefill strips past-turn think, so the
# direct-concat build_training_data must slice past-turn bodies to the
# post-</think> tail (a token-count slice that can drift by ~1 BPE token).
# With it OFF every turn's full streamed body is concatenated verbatim, so
# the trained sequence is byte-exact to what sglang prefilled (zero IS bias)
# and matches the SFT distribution. Turn back ON only once the policy
# reliably emits canonical <think> and prefill length becomes a problem.
export SEARCH_R1_STRIP_THINK=0

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
