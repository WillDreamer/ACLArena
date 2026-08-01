#!/bin/bash
# Resume script for run-qwen3-8B-base.sh
# ----------------------------------------
# 训练在 iter_0000100 (rollout step 99) 后中断。
# slime 在 `--load` 指向的目录下检测到 `latest_checkpointed_iteration.txt` 时，
# 会自动恢复：
#   1. model 权重 + optimizer 状态 + RNG （来自 iter_0000100/*.distcp）
#   2. start_rollout_id（从 checkpoint 中读出，本次将从 rollout 100 继续）
#   3. rollout 数据集进度（从 rollout/global_dataset_state_dict_99.pt）
# 因此本脚本与原脚本相比不需要改动训练参数，只是显式表明这是 resume 入口。

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 redis

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

GPU_LIST=(0 1 2 3 4 5 6 7)  # <<<------  which GPUs to use, directly fill here
CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
export CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
# Automatically detect the number of n_gpus_per_node
NUM_GPUS=${#GPU_LIST[@]}
echo "Detected ${NUM_GPUS} GPUs for this run"

ROLLOUT_BATCH_SIZE=256
GLOBAL_BATCH_SIZE=2048
WANDB_GROUP="reason_rl_Qwen3-8B_bs_${ROLLOUT_BATCH_SIZE}_resume"


LOAD_DIR=${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math/
if [ ! -f "${LOAD_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: ${LOAD_DIR}/latest_checkpointed_iteration.txt not found, cannot resume." >&2
    exit 1
fi
LATEST_ITER=$(cat "${LOAD_DIR}/latest_checkpointed_iteration.txt")
echo "Resuming from iter_$(printf '%07d' ${LATEST_ITER}) under ${LOAD_DIR}"
if [ ! -d "${LOAD_DIR}/iter_$(printf '%07d' ${LATEST_ITER})" ]; then
    echo "ERROR: checkpoint dir iter_$(printf '%07d' ${LATEST_ITER}) missing." >&2
    exit 1
fi

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen3-8B-Base/
   --ref-load ${MODEL_ROOT}/Qwen3-8B-Base_torch_dist/
   --load ${LOAD_DIR}
   --save ${LOAD_DIR}
   --save-interval 20
)

# 将 rollout 数据全部保存到本地（每次 rollout 的 samples 会存为 .pt 文件）
# 路径中的 {rollout_id} 会被替换为实际 rollout 编号；评估数据会存为 eval_{rollout_id}.pt
ROLLOUT_DEBUG_DIR="${MODEL_ROOT}/MultiStageRL/reason_rl_math/rollout_debug"
# 二选一：
# 1) 只保存 rollout 数据：
#    --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"
# 2) 使用 dump-details 会同时保存 rollout_data、train_data 以及 tokenizer 等，便于事后分析：
#    --dump-details "${ROLLOUT_DEBUG_DIR}"

ROLLOUT_ARGS=(
   --prompt-data "${SCRIPT_DIR}/../../data/dapo-math-17k.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 300
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt 16
   --rollout-max-response-len 16384
   --rollout-temperature 1
   --over-sampling-batch-size 512
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data

   # wandb 记录 rollout 的 reward / pass@k（否则训练侧才有 rollout/raw_reward，且默认不记 passrate）
   --log-passrate

   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"
   # 若还需要 train_data、tokenizer 等，可改用：--dump-details "${ROLLOUT_DEBUG_DIR}"

   #eval args
   --eval-interval 10
   --eval-prompt-data aime "${SCRIPT_DIR}/../../data/aime-2024.jsonl" aime25 "${SCRIPT_DIR}/../../data/aime-2025.jsonl"
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 15360
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   # --optimizer-cpu-offload
   # --overlap-cpu-optimizer-d2h-h2d
   # --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project Seq-train-8B
   --wandb-group ${WANDB_GROUP}
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.8
   --sglang-server-concurrency 1024
   # --sglang-ep-size ${NUM_GPUS}
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
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
)

CUSTOM_ARGS=(

   # TIS-related args, recommended to enable when using TIS
   # --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   # --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PYTHONPATH="${ROOT_DIR}/slime:${ROOT_DIR}/Megatron-LM:${SCRIPT_DIR}:${PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --temp-dir /data/user/ray_temp


python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
