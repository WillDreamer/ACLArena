#!/bin/bash
# Search-R1 GSPO RL for Qwen3-8B — CLUSTER / BatchService (single-node colocate by default;
# multi-node disaggregated supported via the BatchService bootstrap env-gates).
#
#   init model : ACLArena/Qwen3-8B-Base-Math-SeaSFT_torch_dist   (Megatron torch_dist, --load/--ref-load)
#                (converted from HF willhx/Qwen3-8B-Base-Math-SeaSFT via
#                 examples/search-r1/convert_hf_to_torch_dist.sh — run that FIRST)
#   base (hf)  : Qwen3-8B-Base                                   (tokenizer/config + rollout init)
#   retriever  : in-cluster CPU-faiss dense retriever on the MAIN node (wiki-18 e5_Flat index),
#                reached by the rollout search tool via SEARCH_R1_SEARCH_URL=http://${MASTER_ADDR}:8000
#   save       : MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search
#
# This is the Cluster-ready sibling of run_qwen3_8b_seq_gspo_sft.sh (which is dev-box only:
# hardcoded /data/user + /shared, a fixed remote retriever URL, and NO retriever launch).
# Here every root is env-overridable (the BatchService bootstrap points them at local NVMe) and the
# retriever is launched + readiness-gated in-script, mirroring the multi-teacher OPD run script.
#
# ── Submit (single node, 8 GPU, COLOCATE — the common case) ──────────────────────────────
#   python3 cluster_cli_seq_search.py batch \
#       --script examples/search-r1/run_qwen3_8b_seq_search.sh \
#       --num-nodes 1 --rollout-nodes 0
#   (All --stage-model/--stage-data have search defaults baked into the CLI.)
#
# ── Submit (multi-node DISAGGREGATED) ────────────────────────────────────────────────────
#   ... --num-nodes 3 --rollout-nodes 2        # 1 train node + 2 rollout nodes (16 GPU actor pool)
#   IMPORTANT: disaggregated weight-sync (UpdateWeightFromDistributed) HARD-ASSERTS TP==1 for
#   Qwen3-8B's fused GQA-QKV (see the multi-teacher OPD run script's PERF_ARGS note). This script
#   FORCES TP=1 whenever ROLLOUT_NUM_GPUS>0. Under colocate (default) TP defaults to 4 (safe:
#   the colocate path de-strides via convert_qwen2_to_hf).
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
# Search retriever config (rollout search tool reads these from the env).
# ----------------------------------------------------------------------------
export SEARCH_R1_STRIP_THINK=${SEARCH_R1_STRIP_THINK:-0}
RETRIEVE_PORT=${RETRIEVE_PORT:-8000}
export SEARCH_R1_SEARCH_URL=${SEARCH_R1_SEARCH_URL:-http://${MASTER_ADDR}:${RETRIEVE_PORT}/retrieve}
export SEARCH_R1_CONCURRENCY=${SEARCH_R1_CONCURRENCY:-64}   # CPU-faiss can bottleneck; keep modest
export SEARCH_R1_TOPK=${SEARCH_R1_TOPK:-3}
export SEARCH_R1_MAX_TURNS=${SEARCH_R1_MAX_TURNS:-5}
WIKI_INDEX=${WIKI_INDEX:-${DATA_ROOT}/wiki-18/e5_Flat.index}
WIKI_CORPUS=${WIKI_CORPUS:-${DATA_ROOT}/wiki-18/wiki-18.jsonl}
E5_MODEL=${E5_MODEL:-${MODEL_ROOT}/e5-base-v2}

# ----------------------------------------------------------------------------
# Model / ckpt args
# ----------------------------------------------------------------------------
# NOTE: --load/--ref-load need Megatron torch_dist, NOT HF safetensors, and must
# point at the BASE dir (Megatron reads latest_checkpointed_iteration.txt then loads
# iter_XXXXXXX/). Init = the oracle-mix SFT-balance run's iter-1000 checkpoint, which
# is already torch_dist. The dir has latest_checkpointed_iteration.txt=1000 + iter_0001000/
# (--finetune loads weights only, resets iteration to 0).
INIT_MODEL=${INIT_MODEL:-${MODEL_ROOT}/ACLArena/SDFT/Qwen3-8B-Base-oracle-mix-SFT-balance}
SAVE_DIR=${SAVE_DIR:-${MODEL_ROOT}/MultiStageRL/Qwen3-8B-SDFT-Search}
if [ ! -d "${INIT_MODEL}" ]; then
  echo "FATAL: init torch_dist model not found at ${INIT_MODEL}." >&2
  echo "       It must be a Megatron torch_dist checkpoint (HF safetensors won't load)." \
       "Stage it via --stage-model ACLArena/Qwen3-8B-Base-SeaSFT_torch_dist/ (already in S3)." >&2
  exit 1
fi

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-64}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}
WANDB_GROUP=${WANDB_GROUP:-SDFT-Search_bs_${ROLLOUT_BATCH_SIZE}}
ROLLOUT_DEBUG_DIR="${MODEL_ROOT}/MultiStageRL/${WANDB_GROUP}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen3-8B-Base/
   --ref-load      ${INIT_MODEL}
   --load          ${INIT_MODEL}
   --save          ${SAVE_DIR}
   --save-interval ${SAVE_INTERVAL:-20}
   --finetune
   --start-rollout-id 0
)

# ----------------------------------------------------------------------------
# EVAL data. The dev script pointed at an ARLArena test_small.parquet that is NOT
# staged to S3; use a subsample of the search test set that IS in S3
# (data/nq_hotpotqa_train/test.parquet). Toggle off with SEARCH_ENABLE_EVAL=0.
# ----------------------------------------------------------------------------
SEARCH_ENABLE_EVAL=${SEARCH_ENABLE_EVAL:-1}
EVAL_TEST_PARQUET=${EVAL_TEST_PARQUET:-${DATA_ROOT}/nq_hotpotqa_train/test.parquet}
EVAL_N=${EVAL_N:-500}

ROLLOUT_ARGS=(
   --prompt-data ${DATA_ROOT}/nq_hotpotqa_train/train.parquet
   --input-key prompt
   --label-key reward_model
   --apply-chat-template
   --rollout-shuffle
   --num-rollout ${NUM_ROLLOUT:-500}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1

   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data
)
# Per-rollout debug .pt dumps are OFF by default here (the dev script always dumped).
# On Cluster the dump dir sits under the S3-synced output prefix, so 500 rollouts of
# full-batch tensors would re-upload hundreds of GB. Turn on with SEARCH_SAVE_DEBUG_ROLLOUT=1.
if [ "${SEARCH_SAVE_DEBUG_ROLLOUT:-0}" = "1" ]; then
  ROLLOUT_ARGS+=(--save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt")
fi
EVAL_ARGS=()
if [ "${SEARCH_ENABLE_EVAL}" = "1" ]; then
  if [ -s "${EVAL_TEST_PARQUET}" ]; then
    EVAL_ARGS=(
       --eval-interval ${EVAL_INTERVAL:-25}
       --eval-prompt-data nq_test "${EVAL_TEST_PARQUET}@[0:${EVAL_N}]"
       --eval-input-key prompt
       --eval-label-key reward_model
       --n-samples-per-eval-prompt 1
    )
  else
    echo "WARNING: eval enabled but ${EVAL_TEST_PARQUET} missing; disabling eval." >&2
  fi
fi

# ----------------------------------------------------------------------------
# Parallelism. Colocate default TP=4 x CP=2 (as the dev script). Disaggregated
# FORCES TP=1 (Qwen3-8B GQA-QKV disagg weight-sync assert; see header).
# ----------------------------------------------------------------------------
if [ "${ROLLOUT_NUM_GPUS:-0}" -gt 0 ]; then
  TP_SIZE=1
  echo "DISAGGREGATED (ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS}) -> forcing TP=1 (GQA-QKV weight-sync assert)."
else
  TP_SIZE=${TP_SIZE:-4}
fi
CP_SIZE=${CP_SIZE:-2}

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size ${CP_SIZE}

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-8192}
)

GRPO_ARGS=(
   --advantage-estimator gspo
   --use-kl-loss
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${LR:-1e-6}
   --lr-decay-style constant
   --weight-decay 0.01
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

# SGLang rollout engine. On p5en the custom all-reduce kernel SIGSEGVs during
# cuda-graph capture (job <JOB_ID>: custom_all_reduce.cuh:37 "CUDA error: invalid
# argument" on all 8 TP ranks -> SGLangEngine.init() -> RolloutManager death ->
# train.py exit 1, 0 steps). Disable custom all-reduce by DEFAULT — it only
# accelerates the in-graph all-reduce, so on 8-GPU TP the speed hit is small and
# cuda graph is kept. Full cuda-graph disable is a fallback (SGLANG_DISABLE_CUDA_GRAPH=1).
# Also cap cuda-graph-bs at 64 (rollout_batch_size*... never needs 256; smaller =
# faster capture + less capture-time VRAM).
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE:-${NUM_GPUS}}
   --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION:-0.7}
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

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search_tools_qwen_sft_no_drift.generate
   --custom-rm-path generate_with_search_tools_qwen_sft_no_drift.reward_func
   --dynamic-sampling-filter-path no_tool_loss_mask_filter.no_tool_loss_mask_filter
   --custom-config-path examples/train_infer_mismatch_helper/mis_8b.yaml
   --custom-tis-function-path generate_with_search_tools_qwen_sft_no_drift.compute_mis_weights_with_cp_no_drift
)

# ----------------------------------------------------------------------------
# Ray + env. PYTHONPATH must include the search-r1 example dir (custom fns) and
# the slime/Megatron roots. Rollout actors on worker nodes don't inherit this
# shell env, so it's ALSO injected into the Ray job runtime-env below.
# ----------------------------------------------------------------------------
export PYTHONPATH="${SLIME_DIR:-${ROOT_DIR}/slime}/examples/search-r1:${SLIME_DIR:-${ROOT_DIR}/slime}:${MEGATRON_DIR:-${ROOT_DIR}/Megatron-LM}:${PYTHONPATH:-}"
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
# Launch the in-cluster CPU-faiss retriever on the MAIN node (background), AFTER
# the Ray head is up (so worker nodes can join immediately, not wait for the 64 GB
# index load). Readiness is gated right before `ray job submit`.
# ----------------------------------------------------------------------------
if [ "${SEARCH_ENABLE_RETRIEVER:-1}" = "1" ]; then
  # The Cluster slime image lacks faiss/fastapi/uvicorn/pydantic; install on the main node.
  if [ "${SEARCH_RETRIEVER_PIP:-1}" = "1" ]; then
    echo "[retriever] ensuring python deps (faiss-cpu fastapi uvicorn pydantic)"
    python3 - <<'PYCHK' || pip install -q faiss-cpu fastapi uvicorn pydantic 2>&1 | tail -5
import importlib.util, sys
missing = [m for m in ("faiss", "fastapi", "uvicorn", "pydantic") if importlib.util.find_spec(m) is None]
sys.exit(1 if missing else 0)
PYCHK
    python3 -c "import faiss, fastapi, uvicorn, pydantic; print('[retriever] deps OK: faiss', faiss.__version__)" \
      || { echo "FATAL: retriever deps still missing after pip install." >&2; exit 1; }
  fi
  # Cap BLAS/OMP threads (job <JOB_ID> fix): on a 96-vCPU node unbounded OpenBLAS overflows
  # its memory-region buffers and SIGSEGVs during faiss index load.
  RETRIEVER_NUM_THREADS=${RETRIEVER_NUM_THREADS:-32}
  echo "[retriever] launching CPU-faiss dense retriever on main node :${RETRIEVE_PORT} (index=${WIKI_INDEX}, threads=${RETRIEVER_NUM_THREADS})"
  OMP_NUM_THREADS=${RETRIEVER_NUM_THREADS} \
  OPENBLAS_NUM_THREADS=${RETRIEVER_NUM_THREADS} \
  MKL_NUM_THREADS=${RETRIEVER_NUM_THREADS} \
  NUMEXPR_NUM_THREADS=${RETRIEVER_NUM_THREADS} \
  VECLIB_MAXIMUM_THREADS=${RETRIEVER_NUM_THREADS} \
  RETRIEVER_ENCODER_DEVICE=cpu python3 "${SLIME_DIR:-${ROOT_DIR}/slime}/examples/search-r1/local_dense_retriever/retrieval_server.py" \
    --index_path "${WIKI_INDEX}" \
    --corpus_path "${WIKI_CORPUS}" \
    --topk "${SEARCH_R1_TOPK}" \
    --retriever_name e5 \
    --retriever_model "${E5_MODEL}" \
    > "${ROOT_DIR}/retriever.log" 2>&1 &
  RETRIEVER_PID=$!
  echo "[retriever] pid=${RETRIEVER_PID}; loading index in background (readiness gated before submit)."
fi

# ----------------------------------------------------------------------------
# Multi-node: wait until ALL GPUs (train + rollout) register with Ray before submit.
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
    \"NCCL_NET_PLUGIN\": \"${NCCL_NET_PLUGIN:-/opt/amazon/ofi-nccl/lib/libnccl-net.so}\",
    \"LD_LIBRARY_PATH\": \"/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64}\""
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"FI_EFA_FORK_SAFE\": \"1\",
    \"SEARCH_R1_SEARCH_URL\": \"${SEARCH_R1_SEARCH_URL}\",
    \"SEARCH_R1_STRIP_THINK\": \"${SEARCH_R1_STRIP_THINK}\",
    \"SEARCH_R1_CONCURRENCY\": \"${SEARCH_R1_CONCURRENCY}\",
    \"SEARCH_R1_TOPK\": \"${SEARCH_R1_TOPK}\",
    \"SEARCH_R1_MAX_TURNS\": \"${SEARCH_R1_MAX_TURNS}\"${MULTINODE_ENV}
  }
}"

# ----------------------------------------------------------------------------
# Gate on retriever readiness (has been loading since ray start, overlapped with
# the GPU-registration wait). Only now — right before submit — require /retrieve.
# ----------------------------------------------------------------------------
if [ "${SEARCH_ENABLE_RETRIEVER:-1}" = "1" ]; then
  echo "[retriever] waiting for it to answer /retrieve (index load ~minutes) ..."
  RET_OK=0
  for i in $(seq 1 120); do
    if curl -sf -m 10 -X POST "http://127.0.0.1:${RETRIEVE_PORT}/retrieve" \
         -H 'Content-Type: application/json' \
         -d '{"queries":["ping"],"topk":1,"return_scores":false}' > /dev/null 2>&1; then
      echo "[retriever] ready after ${i} checks"; RET_OK=1; break
    fi
    kill -0 "${RETRIEVER_PID:-0}" 2>/dev/null || { echo "FATAL: retriever process died; see ${ROOT_DIR}/retriever.log" >&2; tail -n 40 "${ROOT_DIR}/retriever.log" || true; ray stop --force 2>/dev/null || true; exit 1; }
    [ $(( i % 6 )) -eq 1 ] && echo "  [retriever] not ready yet (check $i/120) — index still loading"
    sleep 10
  done
  [ "${RET_OK}" != "1" ] && { echo "FATAL: retriever not ready after 1200s; see ${ROOT_DIR}/retriever.log" >&2; ray stop --force 2>/dev/null || true; exit 1; }
fi

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
[ -n "${RETRIEVER_PID:-}" ] && kill "${RETRIEVER_PID}" 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
