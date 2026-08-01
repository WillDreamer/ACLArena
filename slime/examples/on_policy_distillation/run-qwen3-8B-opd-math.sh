#!/bin/bash

# usage: bash examples/on_policy_distillation/run-qwen3-8B-opd.sh

set -ex

Model_root="/shared/user"
Data_root="/data/user/"

# ---- GPU allocation: teacher is served remotely, so ALL GPUs go to training ----
GPU_LIST=(0 1 2 3 4 5 6 7)        # <<<------ all GPUs for training (Ray, colocate)
NUM_GPUS=${#GPU_LIST[@]}
TRAIN_CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
echo "Training GPUs: ${TRAIN_CUDA_VISIBLE_DEVICES} (${NUM_GPUS} GPUs); teacher served remotely"

# ---- Remote teacher model server (already deployed elsewhere) ----
# NOTE: the OPD reward func (slime.rollout.on_policy_distillation.reward_func)
# posts raw student token ids to sglang's NATIVE /generate endpoint (not the
# OpenAI-compatible /v1 path) and reads meta_info.input_token_logprobs.
TEACHER_URL="http://131.179.168.120:8098"

## Wait until the remote teacher is reachable, then show what it is serving.
until curl -sf ${TEACHER_URL}/health_generate > /dev/null; do
    echo "Waiting for the remote teacher model server at ${TEACHER_URL}..."
    sleep 5
done
curl -s ${TEACHER_URL}/get_model_info; echo
echo "Remote teacher model server is up at ${TEACHER_URL}."


export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "${Data_root}/slime/scripts/models/qwen3-8B.sh"


CKPT_ARGS=(
   --hf-checkpoint ${Model_root}/Qwen3/Qwen3-8B
   --ref-load ${Model_root}/Qwen3/Qwen3-8B_torch_dist
   --load ${Model_root}/MultiStageRL/Qwen3-8B-Base-Math
   --save ${Model_root}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau-OPD
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${Model_root}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size 256
   --n-samples-per-prompt 8
   --rollout-max-response-len 16384
   --rollout-temperature 1

   --global-batch-size 2048
   --balance-data
)

RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
   --rm-url ${TEACHER_URL}/generate
)

EVAL_ARGS=(
   # --eval-interval 20
   # --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   # --n-samples-per-eval-prompt 16
   # --eval-max-response-len 16384
   # --eval-top-p 1
)

PERF_ARGS=(
   # TP=1: the 0.6B student needs no tensor parallelism, and colocate on 3
   # GPUs requires NUM_GPUS divisible by TP. (--sequence-parallel needs TP>1, removed.)
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef 1.0
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project OPD-8B
   --wandb-group ${WANDB_GROUP}
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.8
)


MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)




# ---- launch training (tau-bench style: ray start + direct python3 train.py) ----
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PYTHONPATH="${Data_root}/slime:${Data_root}/Megatron-LM:${PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_TMPDIR="/data1/ray_out"
rm -rf "$RAY_TMPDIR"
mkdir -p "$RAY_TMPDIR"

# Restrict Ray to the training GPUs only (the teacher already holds ${TEACHER_GPU}).
export CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

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
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}



####clear after training
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python