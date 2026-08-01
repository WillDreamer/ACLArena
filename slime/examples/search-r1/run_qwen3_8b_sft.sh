#!/bin/bash

# SFT script: train on filtered rollout trajectories (reward > 0.4)
# Uses --load-debug-rollout-data to load pre-computed tokens + loss_masks.
#
# Prerequisites:
#   1. Run run_qwen3_30b_a3b_rollout_only.sh to generate rollout data
#   2. Run filter_rollout_data.py to filter by reward
#
# Example:
#   python filter_rollout_data.py \
#       --input-dir /data/user/rollout_only_Qwen3-30B-A3B/ \
#       --output-dir /data/user/rollout_only_Qwen3-30B-A3B/filtered/ \
#       --min-reward 0.4 --batch-size 64

# for rerun the task
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
MODEL_ROOT=/shared/user
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-8B.sh"
WANDB_API_KEY="${WANDB_API_KEY}"


FILTERED_DATA_DIR="${MODEL_ROOT}/MultiStageRL/rollout_only_search_Qwen3-32B/filtered"
NUM_ROLLOUT_FILES=536   # filter_rollout_data.py 输出的文件数量，根据实际情况修改

GPU_LIST=(0 1 2 3 4 5 6 7)
CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
export CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${#GPU_LIST[@]}
echo "Detected ${NUM_GPUS} GPUs for this run"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen3-8B-Base/
   --ref-load ${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math/
   --load ${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math/
   --save ${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT
   --save-interval 50
   --finetune
)

SFT_ARGS=(
   --load-debug-rollout-data "${FILTERED_DATA_DIR}/rollout_{rollout_id}.pt"
   --num-rollout ${NUM_ROLLOUT_FILES}

   # SFT 损失 (交叉熵，仅在 loss_mask=1 的 token 上计算)
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns

   # 数据配置
   --num-epoch 2
   --rollout-batch-size 64
   --global-batch-size 32

   --prompt-data ${ROOT_DIR}/Search-R1/data/nq_hotpotqa_train/train_filtered_conf.parquet
   --input-key prompt
   --label-key reward_model
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
   --lr 1e-6
   --lr-decay-style cosine
   --min-lr 5e-7
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project Seq-train-8B
   --wandb-group Qwen3-8B_Math_SeaSFT
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --distributed-timeout-minutes 60
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PYTHONPATH="${ROOT_DIR}/slime:${ROOT_DIR}/Megatron-LM:${SCRIPT_DIR}:${PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
export RAY_memory_usage_threshold=0.99
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
   --no-offload-train \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]}
