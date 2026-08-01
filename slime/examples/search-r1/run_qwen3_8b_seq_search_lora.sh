#!/bin/bash
# Search-R1 GSPO RL for Qwen3-8B with **LoRA** adapters — CLUSTER / BatchService.
# Clone of run_qwen3_8b_seq_search.sh with:
#   - --custom-model-provider-path -> lora_model_provider.custom_model_provider
#   - --only-train-params-name-list 'lora_A' 'lora_B'
#   - LORA_RANK / LORA_ALPHA / LORA_TARGETS env vars
#   - common.py patch applied at startup (lora_A/lora_B sync filter)
#   - lora_hooks.patch_actor_for_lora injected via --custom-megatron-init-path (TBD)
#   - save/wandb names contain "-LoRA"
#
#   init model : ACLArena/Qwen3-8B-Base-Math-SeaSFT_torch_dist   (Megatron torch_dist, --load/--ref-load)
#                (converted from HF willhx/Qwen3-8B-Base-Math-SeaSFT via
#                 examples/search-r1/convert_hf_to_torch_dist.sh — run that FIRST)
#   base (hf)  : Qwen3-8B-Base                                   (tokenizer/config + rollout init)
#   retriever  : CPU-faiss dense retriever (wiki-18 e5_Flat index). In the DEFAULT 4-node
#                layout it runs on a DEDICATED node (0 GPU used) discovered via Ray; the
#                rollout search tool reaches it at SEARCH_R1_SEARCH_URL=http://<retriever-ip>:8000.
#                In single-node/colocate mode it launches locally on the main node.
#   save       : MultiStageRL-LoRA/Qwen3-8B-SDFT-Search-LoRA
#
# This is the Cluster-ready sibling of run_qwen3_8b_seq_gspo_sft.sh (which is dev-box only:
# hardcoded /data/user + /shared, a fixed remote retriever URL, and NO retriever launch).
# Here every root is env-overridable (the BatchService bootstrap points them at local NVMe) and the
# retriever is launched + readiness-gated in-script, mirroring the multi-teacher OPD run script.
#
# ── Submit (DEFAULT: 4 nodes = 1 policy + 2 rollout + 1 dedicated retriever) ───────────────
#   python3 cluster_cli_seq_search_lora.py batch \
#       --script examples/search-r1/run_qwen3_8b_seq_search_lora.sh
#   (num-nodes 4, rollout-nodes 2, retriever-nodes 1 are the CLI defaults. Per-role staging:
#    GPU nodes get model+prompts; the retriever node gets ONLY wiki-18 + e5.)
#
# ── Submit (single node, 8 GPU, COLOCATE — retriever local) ───────────────────────────────
#   python3 cluster_cli_seq_search_lora.py batch \
#       --script examples/search-r1/run_qwen3_8b_seq_search_lora.sh \
#       --num-nodes 1 --rollout-nodes 0 --retriever-nodes 0
#
# IMPORTANT: disaggregated weight-sync (UpdateWeightFromDistributed) HARD-ASSERTS TP==1 for
# Qwen3-8B's fused GQA-QKV. This script FORCES TP=1 whenever ROLLOUT_NUM_GPUS>0 — which also
# sidesteps the LoRA sequence-parallel issue (SP is off here regardless; see SEQ_PARALLEL note).
# Under colocate (single-node) TP defaults to 4.
set -ex

# ════════════════════════════════════════════════════════════════════════════
# DEDICATED RETRIEVER NODE (4-node disaggregated): bootstrap sets SLIME_NODE_ROLE
# =retriever, joins this node to Ray with 0 GPU + a 'retriever' resource label,
# then falls through to this script. We serve the CPU-faiss search server in the
# FOREGROUND (blocks for the job's lifetime) and NEVER touch the train/rollout
# path below — in particular we must NOT run the pkill/ray-head logic, which
# would kill the Ray worker bootstrap just started. All roots + retriever inputs
# were exported + staged by the bootstrap (ROOT_DIR/MODEL_ROOT/DATA_ROOT).
# ════════════════════════════════════════════════════════════════════════════
if [ "${SLIME_NODE_ROLE:-}" = "retriever" ]; then
  echo "[retriever-node] dedicated retriever role."
  R_ROOT_DIR=${ROOT_DIR:-/data/user}
  R_MODEL_ROOT=${MODEL_ROOT:-/shared/user}
  R_DATA_ROOT=${DATA_ROOT:-${R_ROOT_DIR}}
  R_PORT=${RETRIEVE_PORT:-8000}
  R_TOPK=${SEARCH_R1_TOPK:-3}
  R_WIKI_INDEX=${WIKI_INDEX:-${R_DATA_ROOT}/wiki-18/e5_Flat.index}
  R_WIKI_CORPUS=${WIKI_CORPUS:-${R_DATA_ROOT}/wiki-18/wiki-18.jsonl}
  R_E5_MODEL=${E5_MODEL:-${R_MODEL_ROOT}/e5-base-v2}
  R_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
  # ── GPU-faiss per README (Appendix Step 4: --faiss_gpu, GPU encoder), via a CONDA env. ──
  # This dedicated node has 8 IDLE GPUs. CPU-faiss (faiss-cpu + OpenBLAS) SIGSEGV'd mid-run on
  # the 96-vCPU box (OpenBLAS memory-region overflow) -> retriever died -> every search
  # 'Cannot connect :8000' -> retrieval~0 (jobs <JOB_ID>, e4c7e3ad, 8c2d4c54). pip faiss-gpu
  # on H200 fails with CUDA 209 (no sm_90 kernel). Use the EXACT README Step 2 conda recipe
  # (a single self-consistent env: conda pytorch==2.4.0 + pytorch-cuda=12.1 + faiss-gpu=1.8.0
  # from the SAME pytorch/nvidia channels so CUDA matches), VERIFIED on sdb H200: torch 2.4.0
  # cuda=True + faiss 1.8.0 GPU search coexist fine. Serve with --faiss_gpu + encoder on cuda.
  # Fall back to CPU-faiss if the conda setup fails.
  RETRIEVER_USE_GPU=${RETRIEVER_USE_GPU:-1}
  R_CONDA_PY=""
  if [ "${RETRIEVER_USE_GPU}" = "1" ]; then
    R_CONDA_DIR=${RETRIEVER_CONDA_DIR:-${R_ROOT_DIR}/miniconda3}
    R_CONDA_ENV_PY="${R_CONDA_DIR}/envs/retriever/bin/python"
    if [ ! -x "${R_CONDA_ENV_PY}" ]; then
      echo "[retriever-node] setting up conda GPU-faiss env (EXACT README Step 2; verified on H200)..."
      if [ ! -x "${R_CONDA_DIR}/bin/conda" ]; then
        curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh \
          && bash /tmp/mc.sh -b -p "${R_CONDA_DIR}" || echo "[retriever-node] WARNING: miniconda install failed"
      fi
      C="${R_CONDA_DIR}/bin/conda"
      "$C" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
      "$C" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
      # README Step 2, verbatim: fresh py3.10 env, then conda pytorch+cuda, then conda faiss-gpu,
      # then pip the pure-python server deps. All CUDA comes from conda (matches the 12.9 driver).
      "$C" create -n retriever -y python=3.10 \
        || echo "[retriever-node] WARNING: conda env create failed"
      "$C" install -n retriever -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia \
        || echo "[retriever-node] WARNING: conda pytorch install failed"
      "$C" install -n retriever -y faiss-gpu=1.8.0 -c pytorch -c nvidia \
        || echo "[retriever-node] WARNING: conda faiss-gpu install failed"
      "${R_CONDA_ENV_PY}" -m pip install -q transformers datasets pyserini huggingface_hub uvicorn fastapi pydantic 2>&1 | tail -3 || true
    fi
    # Verify the env actually has GPU-capable faiss + cuda torch before committing to GPU mode.
    if [ -x "${R_CONDA_ENV_PY}" ] && "${R_CONDA_ENV_PY}" - <<'PYCHK' 2>/dev/null
import sys
try:
    import faiss, torch
    ok = hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0 and torch.cuda.is_available()
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PYCHK
    then
      R_CONDA_PY="${R_CONDA_ENV_PY}"
      echo "[retriever-node] conda GPU-faiss env READY -> serving with --faiss_gpu + encoder on cuda."
    else
      echo "[retriever-node] WARNING: conda GPU-faiss env not usable; falling back to CPU faiss."
    fi
  fi

  if [ -n "${R_CONDA_PY}" ]; then
    # NOTE: the GPU keepalive for THIS retriever node is started earlier, in the bootstrap
    # (before the 64GB index stage), so it covers the entire idle-GPU window; it self-exits
    # when :${R_PORT}/retrieve serves. Nothing to launch here.
    echo "[retriever-node] launching GPU-faiss server on :${R_PORT} (index=${R_WIKI_INDEX}, sharded across $(nvidia-smi -L 2>/dev/null | wc -l) GPUs); foreground/blocking."
    exec env RETRIEVER_ENCODER_DEVICE=cuda \
      "${R_CONDA_PY}" "${R_SCRIPT_DIR}/local_dense_retriever/retrieval_server.py" \
        --index_path "${R_WIKI_INDEX}" \
        --corpus_path "${R_WIKI_CORPUS}" \
        --topk "${R_TOPK}" \
        --retriever_name e5 \
        --retriever_model "${R_E5_MODEL}" \
        --faiss_gpu
  else
    # CPU fallback: ensure deps in the system python, cap BLAS/OMP threads (job <JOB_ID> fix).
    python3 - <<'PYCHK' || pip install -q faiss-cpu fastapi uvicorn pydantic 2>&1 | tail -5
import importlib.util, sys
missing = [m for m in ("faiss", "fastapi", "uvicorn", "pydantic") if importlib.util.find_spec(m) is None]
sys.exit(1 if missing else 0)
PYCHK
    python3 -c "import faiss, fastapi, uvicorn, pydantic; print('[retriever-node] deps OK: faiss', faiss.__version__)" \
      || { echo "FATAL: retriever deps missing after pip install." >&2; exit 1; }
    R_THREADS=${RETRIEVER_NUM_THREADS:-32}
    echo "[retriever-node] launching CPU-faiss server on :${R_PORT} (index=${R_WIKI_INDEX}, threads=${R_THREADS}); foreground/blocking."
    exec env \
      OMP_NUM_THREADS=${R_THREADS} OPENBLAS_NUM_THREADS=${R_THREADS} MKL_NUM_THREADS=${R_THREADS} \
      NUMEXPR_NUM_THREADS=${R_THREADS} VECLIB_MAXIMUM_THREADS=${R_THREADS} \
      RETRIEVER_ENCODER_DEVICE=cpu \
      python3 "${R_SCRIPT_DIR}/local_dense_retriever/retrieval_server.py" \
        --index_path "${R_WIKI_INDEX}" \
        --corpus_path "${R_WIKI_CORPUS}" \
        --topk "${R_TOPK}" \
        --retriever_name e5 \
        --retriever_model "${R_E5_MODEL}"
  fi
  # exec replaces the shell; nothing below runs on the retriever node.
fi

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
# Provisional URL (points at THIS node). When RETRIEVER_REMOTE=1 the retriever lives
# on a dedicated node; we rediscover its IP via Ray after the head is up and OVERRIDE
# this below (see "remote retriever discovery"). For colocate/local it stays correct.
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
# iter_XXXXXXX/). Init = the oracle-mix SFT-balance run's iter-500 checkpoint
# (s3://YOUR_BUCKET/ACLArena/SDFT/Qwen3-8B-oracle-mix-SFT-balance/iter_0000500/), already
# torch_dist. Only that ONE iter dir is staged, so the bootstrap writes
# latest_checkpointed_iteration.txt=500 into this base dir after staging.
# (--finetune loads weights only, resets iteration to 0.)
INIT_MODEL=${INIT_MODEL:-${MODEL_ROOT}/ACLArena/SDFT/Qwen3-8B-oracle-mix-SFT-balance}
SAVE_DIR=${SAVE_DIR:-${MODEL_ROOT}/MultiStageRL-LoRA/Qwen3-8B-SDFT-Search-LoRA}
if [ ! -d "${INIT_MODEL}" ]; then
  echo "FATAL: init torch_dist model not found at ${INIT_MODEL}." >&2
  echo "       It must be a Megatron torch_dist checkpoint (HF safetensors won't load)." \
       "Stage it via --stage-model ACLArena/Qwen3-8B-Base-SeaSFT_torch_dist/ (already in S3)." >&2
  exit 1
fi

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-64}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}
WANDB_GROUP=${WANDB_GROUP:-SDFT-Search-LoRA_bs_${ROLLOUT_BATCH_SIZE}}
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

# ── Sequence parallel: OFF by default for LoRA (SEQ_PARALLEL=0) ──────────────
# With --sequence-parallel + TP>1, the TE Column/Row-parallel linears receive a
# SEQUENCE-SCATTERED input (seq/TP) and internally all-gather to full seq before
# the matmul, so their OUTPUT is full-seq. Our MergedLoRALinear adapter computes
# on the raw (scattered) input, giving seq/TP -> the base output (full seq) and
# adapter output (seq/TP) mismatch (observed: 8192 vs 2048 = TP4). Making the
# adapter SP-aware needs its own all-gather/reduce-scatter and is fragile; since
# LoRA freezes the base (tiny optimizer state) and we keep full recompute, memory
# is ample on H200 without SP. So default SP OFF here. CP=2 is unaffected (it
# slices sequence at the attention level, consistently for base AND adapter).
# ── Activation recompute (RECOMPUTE=full|selective|none) ─────────────────────
# LoRA + FULL recompute is the prime suspect for train/grad_norm==0: Megatron wraps each
# decoder layer in a custom autograd Function, whose output only gets a grad_fn if a
# TENSOR INPUT requires grad — parameters used inside do NOT count. With the embedding
# frozen (LoRA) hidden_states arrives with requires_grad=False, so the recompute segment
# is never backwarded and lora_A/lora_B get NO gradient (verified reproduction with
# torch.utils.checkpoint(use_reentrant=True): "None of the inputs have requires_grad=True.
# Gradients will be None"). lora_model_provider now installs the PEFT-style
# enable_input_require_grads hook to fix it; set RECOMPUTE=none to A/B-test that fix
# (LoRA freezes the base, so H200 memory is ample without recompute).
RECOMPUTE=${RECOMPUTE:-full}
PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --pipeline-model-parallel-size 1
   --context-parallel-size ${CP_SIZE}

   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-8192}
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
# Opt back into sequence-parallel only if explicitly requested (needs an SP-aware
# adapter, not implemented). Requires TP>1 to have any effect.
if [ "${SEQ_PARALLEL:-0}" = "1" ]; then
  echo "SEQ_PARALLEL=1 -> enabling --sequence-parallel (WARNING: LoRA adapter is NOT SP-aware; will mismatch unless the adapter is fixed)."
  PERF_ARGS+=(--sequence-parallel)
fi

GRPO_ARGS=(
   --advantage-estimator gspo
   --use-kl-loss
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)
# NOTE: --use-tis is added conditionally below (USE_TIS block, default OFF for LoRA).

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

# ── LoRA configuration ──
export LORA_RANK=${LORA_RANK:-16}
export LORA_ALPHA=${LORA_ALPHA:-32}
export LORA_DROPOUT=${LORA_DROPOUT:-0.0}
export LORA_TARGETS=${LORA_TARGETS:-"linear_qkv,linear_proj,linear_fc1,linear_fc2"}
echo "LoRA: rank=${LORA_RANK} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT} targets=${LORA_TARGETS}"

# Apply the common.py patch (idempotent — checks if already patched)
python3 "${SCRIPT_DIR}/../../slime/backends/megatron_utils/update_weight/_patch_lora_filter.py" 2>/dev/null \
  || python3 -c "
import sys; f='/root/slime/slime/backends/megatron_utils/update_weight/common.py'
c=open(f).read()
if 'lora_A' in c: print('common.py already patched'); sys.exit(0)
# vanilla patch
old='        for name, param in model_module.named_parameters():\n            yield _compute_fqn(name), param'
new='        for name, param in model_module.named_parameters():\n            if \"lora_A\" in name or \"lora_B\" in name:\n                continue\n            yield _compute_fqn(name), param'
c=c.replace(old,new,1)
# global patch
old2='        for name, param in model_module.named_parameters():\n            # for model without ddp wrap\n            if not name.startswith(\"module.module.\"):'
new2='        for name, param in model_module.named_parameters():\n            if \"lora_A\" in name or \"lora_B\" in name:\n                continue\n            # for model without ddp wrap\n            if not name.startswith(\"module.module.\"):'
c=c.replace(old2,new2,1)
open(f,'w').write(c)
print('Patched common.py for LoRA filter')
"

# ── TIS / train-infer mismatch correction (DEFAULT OFF for LoRA) ──────────────
# We disable TIS for the LoRA run by default (USE_TIS=0) as a gradient-flow
# diagnostic: the LoRA train-vs-rollout logprob mismatch is ~2x larger than the
# full-FT run, and the importance-sampling + rejection-sampling machinery
# (mis.py, mis_8b_lora.yaml, compute_mis_weights_with_cp_no_drift) is the prime
# suspect for train/grad_norm==0. With USE_TIS=0 we wire NONE of it — no
# --use-tis, no --custom-config-path (the YAML that re-enables use_tis AFTER
# arg-parse and thus overrides the CLI), no --custom-tis-function-path. The
# policy loss then uses the raw per-token loss_mask straight through the reducer,
# so grad flows iff the optimization chain itself is healthy. Re-enable the full
# TIS path (with the veto-relaxed config) by passing USE_TIS=1.
TIS_ARGS=()
if [ "${USE_TIS:-0}" = "1" ]; then
  echo "USE_TIS=1 -> enabling TIS (config=${MIS_CONFIG:-examples/train_infer_mismatch_helper/mis_8b_lora.yaml})."
  GRPO_ARGS+=(--use-tis)
  TIS_ARGS=(
     --custom-config-path ${MIS_CONFIG:-examples/train_infer_mismatch_helper/mis_8b_lora.yaml}
     --custom-tis-function-path generate_with_search_tools_qwen_sft_no_drift.compute_mis_weights_with_cp_no_drift
  )
else
  echo "USE_TIS=0 (default) -> TIS DISABLED for LoRA (no --use-tis / --custom-config-path / --custom-tis-function-path); raw loss_mask drives the gradient."
fi

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search_tools_qwen_sft_no_drift.generate
   --custom-rm-path generate_with_search_tools_qwen_sft_no_drift.reward_func
   --dynamic-sampling-filter-path no_tool_loss_mask_filter.no_tool_loss_mask_filter
   "${TIS_ARGS[@]}"
   --custom-model-provider-path lora_model_provider.custom_model_provider
   --only-train-params-name-list 'lora_A' 'lora_B'
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

# ── GPU keepalive during the retriever-warmup wait (2-node disaggregated only) ──
# In RETRIEVER_REMOTE mode the train node now BLOCKS below (retriever discovery loop +
# readiness gate) for minutes while the dedicated retriever node stages its 64GB index
# and loads GPU faiss. Its 8 GPUs are idle that whole time, which Cluster's
# GTLStuckGPULambda flags as a "stuck" (idle-GPU) job and kills at ~11-16min — the sole
# reason every 2-node run died regardless of the retriever fix (single-node runs, which
# start training immediately, are never hit). Run a light matmul loop to keep GPU util
# non-zero until we kill it right before `ray job submit`. Only in remote/disagg mode;
# harmless no-op elsewhere. Disable with GPU_KEEPALIVE=0.
KEEPALIVE_PID=""
if [ "${GPU_KEEPALIVE:-1}" = "1" ] && [ "${RETRIEVER_REMOTE:-0}" = "1" ]; then
  SCRIPT_DIR_KA="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
  python3 "${SCRIPT_DIR_KA}/gpu_keepalive.py" > "${ROOT_DIR}/gpu_keepalive.log" 2>&1 &
  KEEPALIVE_PID=$!
  echo "[gpu-keepalive] started pid=${KEEPALIVE_PID} (defeats stuck-detector during retriever warmup; killed before ray job submit)."
  # Safety net: reap the keepalive on ANY exit (incl. the FATAL retriever-timeout paths
  # below) so it never lingers holding GPU memory after the script dies.
  trap 'kill "${KEEPALIVE_PID}" 2>/dev/null || true' EXIT
fi

# ----------------------------------------------------------------------------
# Retriever provisioning. Two modes:
#   (A) RETRIEVER_REMOTE=1 : a DEDICATED retriever node is up (4-node disaggregated).
#       Don't launch locally — discover its IP from Ray (the node carrying the
#       'retriever' resource) and point SEARCH_R1_SEARCH_URL at it.
#   (B) otherwise           : launch the CPU-faiss retriever locally on THIS (main)
#       node in the background, as before (single-node / colocate).
# ----------------------------------------------------------------------------
if [ "${SEARCH_ENABLE_RETRIEVER:-1}" = "1" ] && [ "${RETRIEVER_REMOTE:-0}" = "1" ]; then
  echo "[retriever] REMOTE mode — discovering dedicated retriever node IP from Ray (resource 'retriever')."
  RETR_IP=""
  for i in $(seq 1 60); do
    RETR_IP=$(python3 -c "
import ray
ray.init(address='auto', logging_level='ERROR')
ip=''
for n in ray.nodes():
    if n.get('Alive') and float(n.get('Resources',{}).get('retriever',0))>0:
        ip=n['NodeManagerAddress']; break
print(ip)
ray.shutdown()
" 2>/dev/null | tail -1)
    [ -n "$RETR_IP" ] && break
    [ $(( i % 6 )) -eq 1 ] && echo "  [retriever] retriever node not registered in Ray yet (attempt $i/60); retrying"
    sleep 10
  done
  if [ -z "$RETR_IP" ]; then
    echo "FATAL: RETRIEVER_REMOTE=1 but no Ray node with resource 'retriever' after 600s." >&2
    ray stop --force 2>/dev/null || true; exit 1
  fi
  export SEARCH_R1_SEARCH_URL="http://${RETR_IP}:${RETRIEVE_PORT}/retrieve"
  echo "[retriever] remote retriever at ${RETR_IP}:${RETRIEVE_PORT} -> SEARCH_R1_SEARCH_URL=${SEARCH_R1_SEARCH_URL}"
  RETRIEVER_PID=""   # nothing local to reap

elif [ "${SEARCH_ENABLE_RETRIEVER:-1}" = "1" ]; then
  # SINGLE-NODE COLOCATE retriever. Prefer the baked conda GPU-faiss env (README/Search-R1
  # recipe: --faiss_gpu). GPU faiss does NOT SIGSEGV (unlike CPU faiss+OpenBLAS, which
  # crashed job <JOB_ID>) and on H200 the flat index costs only ~5-7GB/GPU out of 143GB,
  # so it comfortably coexists with training. Fall back to CPU faiss (+watchdog) only if
  # the conda env is absent. Detect the conda env once.
  R_CONDA_ENV_PY="${RETRIEVER_CONDA_DIR:-/opt/conda}/envs/retriever/bin/python"
  R_USE_GPU_FAISS=0
  if [ "${RETRIEVER_USE_GPU:-1}" = "1" ] && [ -x "${R_CONDA_ENV_PY}" ] \
     && "${R_CONDA_ENV_PY}" -c "import faiss,torch,sys; sys.exit(0 if hasattr(faiss,'StandardGpuResources') and torch.cuda.is_available() else 1)" 2>/dev/null; then
    R_USE_GPU_FAISS=1
    echo "[retriever] using baked conda GPU-faiss env (${R_CONDA_ENV_PY}); --faiss_gpu, encoder on cuda."
    # transformers>=5 needs torch>=2.5 (torch.distributed.tensor.DTensor); the conda env
    # has torch 2.4.0 (pinned for faiss-gpu=1.8.0), so a too-new transformers crashes the
    # server with `ImportError: cannot import name 'DTensor'` (job <JOB_ID>). e5 is a plain
    # BERT — transformers 4.x loads it fine. Pin to 4.x if the current one won't import.
    if ! "${R_CONDA_ENV_PY}" -c "import transformers, transformers.models.bert.modeling_bert" 2>/dev/null; then
      echo "[retriever] transformers incompatible with torch 2.4 (DTensor); pinning transformers<5 + tokenizers."
      "${R_CONDA_ENV_PY}" -m pip install -q "transformers==4.49.0" "tokenizers<0.22" 2>&1 | tail -3 || true
      "${R_CONDA_ENV_PY}" -c "import transformers, transformers.models.bert.modeling_bert; print('[retriever] transformers OK:', transformers.__version__)" \
        || echo "[retriever] WARNING: transformers still not importable after pin."
    fi
  else
    echo "[retriever] no conda GPU-faiss env; falling back to CPU faiss in system python."
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
  fi
  # CPU-faiss thread cap (job <JOB_ID> fix): unbounded OpenBLAS on 96 vCPU SIGSEGVs on load.
  RETRIEVER_NUM_THREADS=${RETRIEVER_NUM_THREADS:-32}

  # Launch (or relaunch) the retriever server in the background, setting RETRIEVER_PID.
  # Factored into a function so the watchdog below can respawn it after a crash.
  _launch_retriever() {
    if [ "${R_USE_GPU_FAISS}" = "1" ]; then
      RETRIEVER_ENCODER_DEVICE=cuda "${R_CONDA_ENV_PY}" \
        "${SLIME_DIR:-${ROOT_DIR}/slime}/examples/search-r1/local_dense_retriever/retrieval_server.py" \
        --index_path "${WIKI_INDEX}" \
        --corpus_path "${WIKI_CORPUS}" \
        --topk "${SEARCH_R1_TOPK}" \
        --retriever_name e5 \
        --retriever_model "${E5_MODEL}" \
        --faiss_gpu \
        >> "${ROOT_DIR}/retriever.log" 2>&1 &
    else
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
        >> "${ROOT_DIR}/retriever.log" 2>&1 &
    fi
    RETRIEVER_PID=$!
  }
  echo "[retriever] launching dense retriever on main node :${RETRIEVE_PORT} (index=${WIKI_INDEX}, gpu_faiss=${R_USE_GPU_FAISS})"
  _launch_retriever
  echo "[retriever] pid=${RETRIEVER_PID}; loading index in background (readiness gated before submit)."

  # Watchdog: ONLY for the CPU-faiss fallback, which can SIGSEGV mid-run under OpenBLAS/mem
  # contention (job <JOB_ID>). GPU-faiss does NOT crash, so we SKIP the watchdog when
  # R_USE_GPU_FAISS=1 — it was pointless there AND its `set -x` loop spammed
  # `+ kill -0 <pid> / sleep 15` every 15s, burying train.py's real logs (job <JOB_ID>).
  # `set +x` inside so even the CPU-fallback watchdog doesn't flood the trace.
  if [ "${R_USE_GPU_FAISS}" != "1" ] && [ "${RETRIEVER_WATCHDOG:-1}" = "1" ]; then
    set +x
    (
      while true; do
        sleep "${RETRIEVER_WATCHDOG_INTERVAL:-15}"
        if ! kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
          echo "[retriever-watchdog] $(date -u +%H:%M:%S) retriever pid=${RETRIEVER_PID} DEAD; last log:" >&2
          tail -n 15 "${ROOT_DIR}/retriever.log" >&2 2>/dev/null || true
          echo "[retriever-watchdog] relaunching CPU-faiss server (reloads index, ~minutes)..." >&2
          _launch_retriever
          echo "[retriever-watchdog] relaunched pid=${RETRIEVER_PID}" >&2
        fi
      done
    ) &
    RETRIEVER_WATCHDOG_PID=$!
    echo "[retriever] watchdog pid=${RETRIEVER_WATCHDOG_PID} (respawns retriever on crash; RETRIEVER_WATCHDOG=0 to disable)."
    # Stop the watchdog (and the retriever it guards) when this script exits, so the
    # respawn loop doesn't relaunch faiss during container teardown.
    trap 'kill "${RETRIEVER_WATCHDOG_PID}" 2>/dev/null || true; kill "${RETRIEVER_PID}" 2>/dev/null || true' EXIT
  fi
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
  # Poll the ACTUAL retriever endpoint (local 127.0.0.1 for colocate, or the discovered
  # remote node IP for RETRIEVER_REMOTE=1). SEARCH_R1_SEARCH_URL is ".../retrieve".
  GATE_URL="${SEARCH_R1_SEARCH_URL}"
  echo "[retriever] waiting for ${GATE_URL} to answer (index load ~minutes) ..."
  RET_OK=0
  for i in $(seq 1 120); do
    if curl -sf -m 10 -X POST "${GATE_URL}" \
         -H 'Content-Type: application/json' \
         -d '{"queries":["ping"],"topk":1,"return_scores":false}' > /dev/null 2>&1; then
      echo "[retriever] ready after ${i} checks"; RET_OK=1; break
    fi
    # Local-launch mode only: fail fast if the background server process died — UNLESS
    # the watchdog is active, in which case a crash is expected to self-heal (the watchdog
    # respawns it and reassigns RETRIEVER_PID in its own subshell, so this parent's PID may
    # be stale anyway). With the watchdog on we just keep polling until /retrieve answers.
    # Remote mode has no local PID (RETRIEVER_PID empty) — the retriever node fails
    # independently and would just make this gate time out.
    if [ -n "${RETRIEVER_PID:-}" ] && [ "${RETRIEVER_WATCHDOG:-1}" != "1" ]; then
      kill -0 "${RETRIEVER_PID}" 2>/dev/null || { echo "FATAL: local retriever process died; see ${ROOT_DIR}/retriever.log" >&2; tail -n 40 "${ROOT_DIR}/retriever.log" || true; ray stop --force 2>/dev/null || true; exit 1; }
    fi
    [ $(( i % 6 )) -eq 1 ] && echo "  [retriever] not ready yet (check $i/120) — index still loading"
    sleep 10
  done
  [ "${RET_OK}" != "1" ] && { echo "FATAL: retriever not ready after 1200s (url=${GATE_URL})." >&2; ray stop --force 2>/dev/null || true; exit 1; }
fi

# Stop the GPU keepalive NOW — retriever is ready, real training is about to start and
# needs all GPU memory. Kill + wait so its resident tensors are freed before train.py.
if [ -n "${KEEPALIVE_PID:-}" ]; then
  echo "[gpu-keepalive] stopping pid=${KEEPALIVE_PID} before training."
  kill "${KEEPALIVE_PID}" 2>/dev/null || true
  wait "${KEEPALIVE_PID}" 2>/dev/null || true
  sleep 2   # let CUDA free the context/memory
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
