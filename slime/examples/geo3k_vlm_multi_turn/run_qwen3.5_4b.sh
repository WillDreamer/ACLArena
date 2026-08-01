#!/bin/bash

# Run the geo3k VLM multi-turn RL task with Qwen3.5-4B-Base.
#
# This is a self-contained bash launcher modeled on
#   examples/search-r1/run_qwen3_8b_seq_gspo_sft.sh
# (ray bootstrap, sourced MODEL_ARGS, grouped *_ARGS, direct `python3 train.py`),
# but carrying the geo3k task config from
#   examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py
# (multimodal data, custom multi-turn generate fn, math RM), with the model
# swapped to Qwen3.5-4B-Base.

# for rerun the task
# pkill -9 sglang
# sleep 3
# ray stop --force
# pkill -9 ray
# pkill -9 python
# sleep 3

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

ROOT_DIR=/root
MODEL_ROOT=/shared/user
SLIME_DIR="${ROOT_DIR}/slime"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "SCRIPT_DIR=${SCRIPT_DIR}"

# train.py and the dotted custom paths
# (examples.geo3k_vlm_multi_turn.rollout / the config yaml below) are resolved
# relative to the slime repo root, so run from there.
cd "${SLIME_DIR}"

# Qwen3.5-9B-Base megatron MODEL_ARGS (hybrid linear/full-attention VLM spec).
source "${SLIME_DIR}/scripts/models/qwen3.5-4B.sh"

WANDB_API_KEY="${WANDB_API_KEY}"
ROLLOUT_BATCH_SIZE=64
GLOBAL_BATCH_SIZE=512

WANDB_GROUP="geo3k_vlm_qwen3.5_4b"
SAVE_DIR="${MODEL_ROOT}/MultiStageRL/Qwen3.5-4B-Base-geo3k"

# geo3k multimodal dataset (already downloaded locally).
# If missing, fetch with:
#   hf download --repo-type dataset VeraIsHere/geo3k_imgurl_processed \
#       --local-dir ${MODEL_ROOT}/geo3k_imgurl_processed
GEO3K_DATA="${MODEL_ROOT}/geo3k_imgurl_processed/train.parquet"
if [ ! -f "${GEO3K_DATA}" ]; then
    echo "ERROR: geo3k train parquet not found at ${GEO3K_DATA}" >&2
    exit 1
fi

GPU_LIST=(0 1 2 3 4 5 6 7)
CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
export CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${#GPU_LIST[@]}
echo "Detected ${NUM_GPUS} GPUs for this run"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen3.5/Qwen3.5-4B-Base
   # Qwen3.5 is a VLM: load policy + reference weights directly from the HF
   # checkpoint via the megatron.bridge path (--megatron-to-hf-mode bridge below),
   # exactly like the reference VLM scripts (examples/geo3k_vlm/run_geo3k_vlm.sh).
   # Do NOT point these at convert_hf_to_torch_dist output: that tool builds a
   # text-only GPTModel (top-level embedding.*/decoder.*, no vision_model.* and no
   # language_model.* prefix), which mismatches the VLM slime instantiates and
   # crashes the dist-ckpt loader with "_io.BytesIO has no len()".
   --ref-load ${MODEL_ROOT}/Qwen3.5/Qwen3.5-4B-Base
   --load ${MODEL_ROOT}/Qwen3.5/Qwen3.5-4B-Base
   --save ${SAVE_DIR}
   --save-interval 20
   --save-retain-interval 60
   --finetune
   --start-rollout-id 0
)

ROLLOUT_ARGS=(
   --prompt-data ${GEO3K_DATA}
   --input-key problem
   --label-key answer
   --multimodal-keys '{"image": "images"}'
   --rm-type math
   --apply-chat-template
   --custom-generate-function-path examples.geo3k_vlm_multi_turn.rollout.generate
   --custom-config-path examples/geo3k_vlm_multi_turn/geo3k_vlm_multi_turn_config.yaml
   --rollout-shuffle
   --num-rollout 3000
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1

   # eval args (optional; test split lives next to the train parquet)
   # --eval-interval 20
   # --eval-prompt-data geo3k_eval ${MODEL_ROOT}/geo3k_imgurl_processed/test.parquet@[0:64]
   # --n-samples-per-eval-prompt 1
   # --eval-max-response-len 4096
   # --eval-top-k 1

   --global-batch-size ${GLOBAL_BATCH_SIZE}
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # Qwen3.5 uses Gated DeltaNet (linear attention), whose megatron kernel does
   # NOT support packed/THD sequences ("GDN does not support packed sequence for
   # now."). slime defaults to qkv_format=thd (packing) and --use-dynamic-batch-size
   # relies on that packing, so both are incompatible with GDN. Mirror the GDN
   # reference recipe (examples/geo3k_vlm/run_geo3k_qwen35.sh): unpacked bshd +
   # fixed micro-batch, no dynamic batching.
   --qkv-format bshd
   --micro-batch-size 1
)

GRPO_ARGS=(
   --advantage-estimator gspo
   --use-kl-loss
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   # TIS is left off: the search-r1 reference paired --use-tis with a custom
   # _no_drift TIS fn for its placeholder log_probs. The geo3k generate fn has
   # no matching wrapper, so enable only after verifying its rollout_log_probs.
   # --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
   --use-precision-aware-optimizer
)

if [ -n "${WANDB_API_KEY}" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project geo3k-vlm-qwen3.5-4b
      --wandb-group ${WANDB_GROUP}
      --wandb-key ${WANDB_API_KEY}
      --disable-wandb-random-suffix
   )
else
   echo "WANDB_API_KEY not set; running without wandb logging"
   WANDB_ARGS=()
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   # convert megatron weights to HF via the bridge during weight sync (qwen3.5 spec)
   --megatron-to-hf-mode bridge

   --distributed-timeout-minutes 60
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PYTHONPATH="${SLIME_DIR}:${ROOT_DIR}/Megatron-LM:${SCRIPT_DIR}:${PYTHONPATH}"
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
   --rollout-num-gpus ${NUM_GPUS} \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   2>&1 | tee "${LOG_FILE}"
#
