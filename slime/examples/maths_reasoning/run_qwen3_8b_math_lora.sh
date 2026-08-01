#!/bin/bash
# Math-reasoning GRPO RL for Qwen3-8B with **LoRA** adapters — CLUSTER / BatchService.
#
# This is the Cluster-ready, LoRA-on-SDFT sibling of examples/maths_reasoning/run-qwen3-8B-base.sh
# (which is dev-box only: hardcoded /data/user + /shared, full fine-tune, no S3 staging).
# It fuses two references:
#   * examples/maths_reasoning/run-qwen3-8B-base.sh          -> the math recipe
#       (dapo-math-17k prompts, deepscaler rule reward, DAPO dynamic sampling, GRPO,
#        16384-token responses, AIME eval).
#   * examples/search-r1/run_qwen3_8b_seq_search_lora.sh -> the Cluster + LoRA scaffolding
#       (env-overridable roots, Ray head + runtime-env, colocate/disaggregated topology,
#        LoRA adapter injection via lora_model_provider, common.py sync-filter patch,
#        SGLang p5en safety flags). The retriever / user-sim machinery is DROPPED — math
#        needs no external tool service.
#
#   init model : ACLArena/SDFT/Qwen3-8B-Base-oracle-mix-SFT-balance   (Megatron torch_dist,
#                iter_0001000; --load AND --ref-load). This is the "SDFT" checkpoint the task
#                asks us to LoRA-RL on top of — identical init to the search-r1 LoRA run.
#                (--finetune loads weights only, resets iteration to 0; KL is against this init.)
#   base (hf)  : Qwen3-8B-Base                                        (tokenizer/config + rollout init)
#   train data : data/dapo-math-17k/dapo-math-17k.jsonl               (prompt + label keys)
#   reward     : deepscaler rule-based (--rm-type deepscaler; boxed-answer grading) — BUILT IN,
#                so NO --custom-generate-function-path / --custom-rm-path (unlike search/tau).
#   save       : MultiStageRL-LoRA-math/Qwen3-8B-SDFT-Math-LoRA
#
# ── Submit (DEFAULT: single node, 8 GPU, COLOCATE) ──────────────────────────────────────────
#   python3 cluster_cli_math_lora.py batch \
#       --script examples/maths_reasoning/run_qwen3_8b_math_lora.sh
#
# ── Submit (disaggregated: 1 train + N rollout) ─────────────────────────────────────────────
#   python3 cluster_cli_math_lora.py batch \
#       --script examples/maths_reasoning/run_qwen3_8b_math_lora.sh \
#       --num-nodes 3 --rollout-nodes 2
#
# IMPORTANT: disaggregated weight-sync (UpdateWeightFromDistributed) HARD-ASSERTS TP==1 for
# Qwen3-8B's fused GQA-QKV. This script FORCES TP=1 whenever ROLLOUT_NUM_GPUS>0 (which also
# sidesteps the LoRA sequence-parallel issue — SP is off here regardless). Under colocate
# (single-node) TP defaults to 4.
set -ex

# for rerun on a dev box (harmless on a fresh BatchService node)
pkill -9 sglang 2>/dev/null || true
sleep 1
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 1

# ── Overridable roots (BatchService bootstrap sets these to the NVMe mirror; dev defaults kept) ──
ROOT_DIR=${ROOT_DIR:-/data/user}
MODEL_ROOT=${MODEL_ROOT:-/shared/user}
DATA_ROOT=${DATA_ROOT:-${ROOT_DIR}}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SLIME_DIR:-${ROOT_DIR}/slime}"   # train.py / tools / examples paths resolve from the slime root

export PYTHONBUFFERED=16

# ── Multi-node hooks (BatchService bootstrap exports these; single-node/dev defaults preserved) ──
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-0}   # >0 => DISAGGREGATED; 0 => COLOCATE

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
[ "$NVLINK_COUNT" -gt 0 ] && HAS_NVLINK=1 || HAS_NVLINK=0
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

GPU_LIST=(0 1 2 3 4 5 6 7)
export CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
NUM_GPUS=${#GPU_LIST[@]}
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} GPUs/node)"

source "${SCRIPT_DIR}/../../scripts/models/qwen3-8B.sh"   # -> MODEL_ARGS

# ----------------------------------------------------------------------------
# Model / ckpt args
# ----------------------------------------------------------------------------
# NOTE: --load/--ref-load need Megatron torch_dist, NOT HF safetensors, and must point at the
# BASE dir (Megatron reads latest_checkpointed_iteration.txt then loads iter_XXXXXXX/). Init =
# the oracle-mix SFT-balance run's iter-1000 checkpoint
# (s3://YOUR_BUCKET/ACLArena/SDFT/Qwen3-8B-Base-oracle-mix-SFT-balance/iter_0001000/), already
# torch_dist; the base dir carries latest_checkpointed_iteration.txt=1000. --finetune loads
# weights only and resets the iteration to 0.
INIT_MODEL=${INIT_MODEL:-${MODEL_ROOT}/ACLArena/SDFT/Qwen3-8B-Base-oracle-mix-SFT-balance}
SAVE_DIR=${SAVE_DIR:-${MODEL_ROOT}/MultiStageRL-LoRA-math/Qwen3-8B-SDFT-Math-LoRA}
if [ ! -d "${INIT_MODEL}" ]; then
  echo "FATAL: init torch_dist model not found at ${INIT_MODEL}." >&2
  echo "       It must be a Megatron torch_dist checkpoint (HF safetensors won't load)." \
       "Stage it via --stage-model ACLArena/SDFT/Qwen3-8B-Base-oracle-mix-SFT-balance/ (already in S3)." >&2
  exit 1
fi

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-256}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2048}
WANDB_GROUP=${WANDB_GROUP:-math_rl_Qwen3-8B-SDFT-LoRA_bs_${ROLLOUT_BATCH_SIZE}}
ROLLOUT_DEBUG_DIR="${MODEL_ROOT}/MultiStageRL-LoRA-math/${WANDB_GROUP}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen3-8B-Base/
   --ref-load      ${INIT_MODEL}
   --load          ${INIT_MODEL}
   --save          ${SAVE_DIR}
   --save-interval ${SAVE_INTERVAL:-20}
   --finetune
   --start-rollout-id 0
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout ${NUM_ROLLOUT:-300}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT:-16}
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN:-16384}
   --rollout-temperature 1
   --over-sampling-batch-size ${OVER_SAMPLING_BATCH_SIZE:-512}
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data

   # wandb records rollout reward / pass@k (otherwise only train-side rollout/raw_reward is logged)
   --log-passrate
)
# Per-rollout debug .pt dumps are OFF by default (the dev script always dumped). On Cluster
# the dump dir sits under the S3-synced output prefix, so 300 rollouts of full-batch tensors
# would re-upload hundreds of GB. Turn on with MATH_SAVE_DEBUG_ROLLOUT=1.
if [ "${MATH_SAVE_DEBUG_ROLLOUT:-0}" = "1" ]; then
  ROLLOUT_ARGS+=(--save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt")
fi

# ----------------------------------------------------------------------------
# EVAL data. The dev math script evaluates on AIME-2024 / AIME-2025 jsonls, which are NOT
# staged to S3. So eval is file-existence-gated: enabled only if BOTH files are present
# under $DATA_ROOT (stage them + set MATH_ENABLE_EVAL=1 to turn on). Defaults to a no-op.
# ----------------------------------------------------------------------------
MATH_ENABLE_EVAL=${MATH_ENABLE_EVAL:-1}
AIME24_JSONL=${AIME24_JSONL:-${DATA_ROOT}/aime-2024/aime-2024.jsonl}
AIME25_JSONL=${AIME25_JSONL:-${DATA_ROOT}/aime-2025/aime-2025.jsonl}
EVAL_ARGS=()
if [ "${MATH_ENABLE_EVAL}" = "1" ]; then
  if [ -s "${AIME24_JSONL}" ] && [ -s "${AIME25_JSONL}" ]; then
    EVAL_ARGS=(
       --eval-interval ${EVAL_INTERVAL:-10}
       --eval-prompt-data aime "${AIME24_JSONL}" aime25 "${AIME25_JSONL}"
       --n-samples-per-eval-prompt ${N_SAMPLES_PER_EVAL_PROMPT:-16}
       --eval-max-response-len ${EVAL_MAX_RESPONSE_LEN:-16384}
       --eval-top-p 1
    )
  else
    echo "WARNING: MATH_ENABLE_EVAL=1 but AIME eval jsonls missing (${AIME24_JSONL} / ${AIME25_JSONL}); disabling eval." \
         "Stage them to S3 (e.g. --stage-data aime-2024/ --stage-data aime-2025/) to enable." >&2
  fi
fi

# ----------------------------------------------------------------------------
# Parallelism. Colocate default TP=4 x CP=2 (LoRA-safe). Disaggregated FORCES TP=1
# (Qwen3-8B GQA-QKV disagg weight-sync assert; see header).
# ----------------------------------------------------------------------------
if [ "${ROLLOUT_NUM_GPUS:-0}" -gt 0 ]; then
  TP_SIZE=1
  echo "DISAGGREGATED (ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS}) -> forcing TP=1 (GQA-QKV weight-sync assert)."
else
  TP_SIZE=${TP_SIZE:-4}
fi
CP_SIZE=${CP_SIZE:-2}

# ── Sequence parallel: OFF by default for LoRA (SEQ_PARALLEL=0) ──────────────
# With --sequence-parallel + TP>1 the TE linears receive a SEQUENCE-SCATTERED input and
# all-gather internally, so their output is full-seq while our adapter computes on the raw
# (scattered) input -> shape mismatch. Making the adapter SP-aware is fragile; LoRA freezes
# the base (tiny optimizer state) + we keep full recompute, so H200 memory is ample without SP.
# ── Activation recompute (RECOMPUTE=full|selective|none) ─────────────────────
# LoRA + FULL recompute needs lora_model_provider's enable_input_require_grads hook, else the
# frozen embedding gives hidden_states requires_grad=False and the checkpointed decoder layers
# are never backwarded -> lora_A/lora_B get NO gradient (train/grad_norm==0). The provider
# installs that hook; set RECOMPUTE=none to A/B-test (LoRA frees enough memory to skip recompute).
RECOMPUTE=${RECOMPUTE:-full}
PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --pipeline-model-parallel-size 1
   --context-parallel-size ${CP_SIZE}

   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-15360}
)
case "${RECOMPUTE}" in
  full)
    PERF_ARGS+=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1)
    ;;
  selective)
    PERF_ARGS+=(--recompute-granularity selective)
    ;;
  none)
    echo "RECOMPUTE=none -> NO activation checkpointing (isolates the LoRA/recompute gradient break)."
    ;;
  *)
    echo "FATAL: RECOMPUTE must be full|selective|none (got '${RECOMPUTE}')." >&2; exit 1
    ;;
esac
# Opt back into sequence-parallel only if explicitly requested (adapter is NOT SP-aware).
if [ "${SEQ_PARALLEL:-0}" = "1" ]; then
  echo "SEQ_PARALLEL=1 -> enabling --sequence-parallel (WARNING: LoRA adapter is NOT SP-aware; will mismatch unless the adapter is fixed)."
  PERF_ARGS+=(--sequence-parallel)
fi

GRPO_ARGS=(
   --advantage-estimator ${ADV_ESTIMATOR:-grpo}
   --use-kl-loss
   --kl-loss-coef ${KL_LOSS_COEF:-0.01}
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# NOTE on LR: the full-FT math dev run used 1e-6. LoRA adapters are tiny and typically want a
# HIGHER LR (try LR=1e-5). Default kept at 1e-6 (search-r1 LoRA precedent); override via LR=.
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${LR:-1e-6}
   --lr-decay-style constant
   --weight-decay ${WEIGHT_DECAY:-0.01}
   --adam-beta1 0.9
   --adam-beta2 0.98
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project ${WANDB_PROJECT:-Seq-train-8B}
   --wandb-group ${WANDB_GROUP}
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
)

# SGLang rollout engine. On p5en the custom all-reduce kernel SIGSEGVs during cuda-graph
# capture (search-r1 job <JOB_ID>), so disable custom all-reduce by DEFAULT — it only
# accelerates the in-graph all-reduce, so on 8-GPU TP the speed hit is small and cuda graph
# is kept. Full cuda-graph disable is a fallback (SGLANG_DISABLE_CUDA_GRAPH=1).
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}
   --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION:-0.7}
   --sglang-server-concurrency 1024
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 ${SGLANG_CUDA_GRAPH_MAX_BS:-64})
)
[ "${SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-1}" = "1" ] && SGLANG_ARGS+=(--sglang-disable-custom-all-reduce)
[ "${SGLANG_DISABLE_CUDA_GRAPH:-0}" = "1" ]        && SGLANG_ARGS+=(--sglang-disable-cuda-graph)

FAULT_TOLERANCE_ARGS=(
   --use-fault-tolerance
   --rollout-health-check-interval 30
   --rollout-health-check-timeout 60
   --rollout-health-check-first-wait 120
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --distributed-timeout-minutes 60
)

# ── LoRA configuration ──
export LORA_RANK=${LORA_RANK:-16}
export LORA_ALPHA=${LORA_ALPHA:-32}
export LORA_DROPOUT=${LORA_DROPOUT:-0.0}
export LORA_TARGETS=${LORA_TARGETS:-"linear_qkv,linear_proj,linear_fc1,linear_fc2"}
echo "LoRA: rank=${LORA_RANK} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT} targets=${LORA_TARGETS}"

# Apply the common.py sync-filter patch (idempotent — skips lora_A/lora_B params in the
# actor->rollout weight-sync loop, which otherwise chokes on the extra adapter tensors).
COMMON_PY="${SLIME_DIR:-${ROOT_DIR}/slime}/slime/backends/megatron_utils/update_weight/common.py"
python3 - "$COMMON_PY" <<'PY'
import sys
f = sys.argv[1]
c = open(f).read()
if 'lora_A' in c:
    print('common.py already patched for LoRA filter'); sys.exit(0)
# vanilla path
old = '        for name, param in model_module.named_parameters():\n            yield _compute_fqn(name), param'
new = ('        for name, param in model_module.named_parameters():\n'
       '            if "lora_A" in name or "lora_B" in name:\n'
       '                continue\n'
       '            yield _compute_fqn(name), param')
c = c.replace(old, new, 1)
# global path
old2 = '        for name, param in model_module.named_parameters():\n            # for model without ddp wrap\n            if not name.startswith("module.module."):'
new2 = ('        for name, param in model_module.named_parameters():\n'
        '            if "lora_A" in name or "lora_B" in name:\n'
        '                continue\n'
        '            # for model without ddp wrap\n'
        '            if not name.startswith("module.module."):')
c = c.replace(old2, new2, 1)
open(f, 'w').write(c)
print('Patched common.py for LoRA filter')
PY

CUSTOM_ARGS=(
   --custom-model-provider-path lora_model_provider.custom_model_provider
   --only-train-params-name-list 'lora_A' 'lora_B'
)

# ----------------------------------------------------------------------------
# Ray + env. PYTHONPATH must include the search-r1 example dir (the LoRA modules
# lora_model_provider + lora_hooks live there, reused as-is), the maths_reasoning
# example dir, and the slime/Megatron roots. Worker-node rollout actors don't inherit
# this shell env, so it's ALSO injected into the Ray job runtime-env below.
# ----------------------------------------------------------------------------
_SLIME="${SLIME_DIR:-${ROOT_DIR}/slime}"
export PYTHONPATH="${_SLIME}/examples/search-r1:${_SLIME}/examples/maths_reasoning:${_SLIME}:${MEGATRON_DIR:-${ROOT_DIR}/Megatron-LM}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false
export RAY_memory_usage_threshold=0.99
RAY_TEMP_DIR=${RAY_TEMP_DIR:-${ROOT_DIR}/ray_temp}
rm -rf "$RAY_TEMP_DIR"; mkdir -p "$RAY_TEMP_DIR"

# Dev-box needs the system lib dir prepended; the Cluster image sets SKIP_SYS_LDPATH=1.
if [ -z "${SKIP_SYS_LDPATH:-}" ]; then
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
    export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
fi

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir "${RAY_TEMP_DIR}"
export RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=1

# ----------------------------------------------------------------------------
# Multi-node (disaggregated): wait until ALL GPUs (train + rollout) register with Ray.
# ----------------------------------------------------------------------------
EXPECTED_GPUS=$(( ACTOR_NUM_NODES * NUM_GPUS + ROLLOUT_NUM_GPUS ))
if [ "$EXPECTED_GPUS" -gt "$NUM_GPUS" ]; then
    echo "Waiting for ${EXPECTED_GPUS} GPUs (train $(( ACTOR_NUM_NODES * NUM_GPUS )) + rollout ${ROLLOUT_NUM_GPUS}) to register (up to 30 min)..."
    WAIT_OK=0; WAIT_START=$(date +%s)
    for i in $(seq 1 180); do
        READ=$(python3 -c "
import ray
ray.init(address='auto', logging_level='ERROR')
r = ray.cluster_resources()
alive = sorted(n['NodeManagerAddress'] for n in ray.nodes() if n.get('Alive'))
print(int(r.get('GPU', 0))); print(len(alive)); print(','.join(alive))
ray.shutdown()
" 2>/dev/null)
        GOT=$(echo "$READ" | sed -n '1p'); GOT=${GOT:-0}
        NNODES=$(echo "$READ" | sed -n '2p'); NNODES=${NNODES:-0}
        ALIVE_IPS=$(echo "$READ" | sed -n '3p')
        ELAPSED=$(( $(date +%s) - WAIT_START ))
        echo "  [$(date -u +%H:%M:%S) | +${ELAPSED}s | iter $i] ${GOT}/${EXPECTED_GPUS} GPUs, ${NNODES} node(s) alive: [${ALIVE_IPS}]"
        [ "$GOT" -ge "$EXPECTED_GPUS" ] && { WAIT_OK=1; break; }
        sleep 10
    done
    if [ "$WAIT_OK" != "1" ]; then
        echo "FATAL: only ${GOT}/${EXPECTED_GPUS} GPUs registered after $(( $(date +%s) - WAIT_START ))s. Likely cross-subnet control-plane block or a child node never booted." >&2
        ray stop --force 2>/dev/null || true
        exit 1
    fi
    echo "All ${EXPECTED_GPUS} GPUs registered in $(( $(date +%s) - WAIT_START ))s; proceeding."
fi

# ── Resource / topology args: colocate (default) vs disaggregated ──
RESOURCE_ARGS=(
   --actor-num-nodes ${ACTOR_NUM_NODES}
   --actor-num-gpus-per-node ${NUM_GPUS}
   --num-gpus-per-node ${NUM_GPUS}
)
if [ "${ROLLOUT_NUM_GPUS:-0}" -gt 0 ]; then
   RESOURCE_ARGS+=(--rollout-num-gpus ${ROLLOUT_NUM_GPUS})
else
   RESOURCE_ARGS+=(--colocate --no-offload-train --no-offload-rollout)
fi

# ── Runtime-env for worker-node rollout actors (they don't inherit this shell) ──
MULTINODE_ENV=""
if [ "${ACTOR_NUM_NODES:-1}" -gt 1 ] || [ "${ROLLOUT_NUM_GPUS:-0}" -gt 0 ]; then
    MULTINODE_ENV=",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME:-eth0}\",
    \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME:-eth0}\",
    \"TP_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME:-eth0}\",
    \"FI_PROVIDER\": \"${FI_PROVIDER:-efa}\",
    \"FI_EFA_USE_DEVICE_RDMA\": \"1\",
    \"NCCL_DEBUG\": \"${NCCL_DEBUG:-INFO}\",
    \"NCCL_DEBUG_SUBSYS\": \"INIT,NET\",
    \"TORCH_NCCL_BLOCKING_WAIT\": \"1\",
    \"TORCH_NCCL_TIMEOUT_MS\": \"${TORCH_NCCL_TIMEOUT_MS:-600000}\",
    \"NCCL_ASYNC_ERROR_HANDLING\": \"1\",
    \"NCCL_NET_PLUGIN\": \"${NCCL_NET_PLUGIN:-/opt/amazon/ofi-nccl/lib/libnccl-net.so}\",
    \"SLIME_WEIGHT_UPDATE_GROUP_TIMEOUT_S\": \"${SLIME_WEIGHT_UPDATE_GROUP_TIMEOUT_S:-600}\",
    \"SLIME_ROLLOUT_ENGINE_INIT_TIMEOUT_S\": \"${SLIME_ROLLOUT_ENGINE_INIT_TIMEOUT_S:-1800}\",
    \"LD_LIBRARY_PATH\": \"/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64}\""
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"FI_EFA_FORK_SAFE\": \"1\",
    \"LORA_RANK\": \"${LORA_RANK}\",
    \"LORA_ALPHA\": \"${LORA_ALPHA}\",
    \"LORA_DROPOUT\": \"${LORA_DROPOUT}\",
    \"LORA_TARGETS\": \"${LORA_TARGETS}\"${MULTINODE_ENV}
  }
}"

LOG_FILE="${ROOT_DIR}/${WANDB_GROUP}.log"
echo "Logging to ${LOG_FILE}"
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   ${RESOURCE_ARGS[@]} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${FAULT_TOLERANCE_ARGS[@]} \
   2>&1 | tee "${LOG_FILE}"

#### cleanup
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
