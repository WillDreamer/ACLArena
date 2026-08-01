#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python

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
WANDB_API_KEY="${WANDB_API_KEY}"
ROLLOUT_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=64
WANDB_GROUP="tau-bench_Qwen3-30B-A3B_async_bs_${ROLLOUT_BATCH_SIZE}"
ROLLOUT_DEBUG_DIR="${MODEL_ROOT}/multi_stage_rl_log/${WANDB_GROUP}"


ACTOR_GPUS=4
ROLLOUT_GPUS=4
TOTAL_GPUS=$((ACTOR_GPUS + ROLLOUT_GPUS))

GPU_LIST=(0 1 2 3 4 5 6 7)  # <<<------  which GPUs to use, directly fill here
CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
export CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Training GPUs: ${ACTOR_GPUS}, Rollout GPUs: ${ROLLOUT_GPUS}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TAU_BENCH_DIR="${ROOT_DIR}/slime/examples/tau-bench"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-30B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen/Qwen3-30B-A3B-Base/
   --ref-load ${ROOT_DIR}/Qwen3-30B-A3B_base_math_search_strict_v3_gspo_cold_then_mask/
   --load ${ROOT_DIR}/Qwen3-30B-A3B_base_math_search_strict_v3_gspo_cold_then_mask/
   --save ${ROOT_DIR}/Qwen3-30B-A3B_base_math_search_tau_async/
   --save-interval 20
   --save-retain-interval 40
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
   --rollout-max-response-len 1024
   --rollout-max-context-len 131072
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std 
   --rollout-temperature 1
   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data
   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"
)


EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data retail-dev ${ROOT_DIR}/tau-bench/retail_dev_tasks.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 1024
   --eval-top-k 1
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

GRPO_ARGS=(
   --advantage-estimator gspo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project Seq-train
   --wandb-group ${WANDB_GROUP}
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
)

SGLANG_ARGS=(
   # Reduced from 8 to 4 GPUs per engine to match the rollout GPU budget
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.8
   # ep-size=4 with 4 GPUs: each GPU holds different experts (TP=1, EP=4)
   --sglang-ep-size 4
   --sglang-max-total-tokens 1500000
   # YaRN: extend context to 131072 (4x the native 32768)
   --sglang-context-length 131072
   --sglang-json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   # Self-play: user simulator runs through the same SGLang engine, no external API
   --custom-generate-function-path generate_with_tau_selfplay.generate

   # TIS-related args
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_TMPDIR="/data1/ray_out"
rm -rf "$RAY_TMPDIR"
mkdir -p "$RAY_TMPDIR"

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${TOTAL_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# SELFPLAY_DIR is added so generate_with_tau_selfplay.py and trainable_agents_selfplay.py
# can be imported as modules.  TAU_BENCH_DIR is still needed for openai_tool_adapter.py.
SELFPLAY_DIR="${ROOT_DIR}/slime/examples/tau-bench-async"
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${ROOT_DIR}/slime:${ROOT_DIR}/Megatron-LM:${SELFPLAY_DIR}:${TAU_BENCH_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"CUDA_VISIBLE_DEVICES\": \"${CUDA_VISIBLE_DEVICES}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${ROOT_DIR}/slime/train_async.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${ACTOR_GPUS} \
   --rollout-num-gpus ${ROLLOUT_GPUS} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
