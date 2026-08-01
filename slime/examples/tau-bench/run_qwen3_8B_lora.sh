#!/bin/bash
# Tau-bench GSPO RL for Qwen3-8B with **LoRA** adapters, trained ON TOP OF the SDFT
# (multi-stage SFT) weights — CLUSTER / BatchService.
#
# This is the tau-bench sibling of examples/search-r1/run_qwen3_8b_seq_search_lora.sh.
# It reuses the EXACT same LoRA machinery from the search-r1 example:
#   - --custom-model-provider-path lora_model_provider.custom_model_provider
#   - --only-train-params-name-list 'lora_A' 'lora_B'
#   - LORA_RANK / LORA_ALPHA / LORA_TARGETS env vars
#   - lora_hooks (installed at import by lora_model_provider) + common.py lora filter
#   - MERGE-around-sync / disable-during-ref semantics (base frozen; only adapters train)
# The base weights are the SDFT checkpoint (frozen); LoRA adapters are the only trainable
# params. This is a lightweight RL step on top of SDFT: tiny optimizer state, base untouched.
#
# LoRA-specific deviations from the full-FT tau run (run_qwen3_8B.sh) — all deliberate:
#   - TIS is OFF by default (USE_TIS=0). The full-FT run's mis.yaml has a SEQUENCE-level veto
#     (rs_veto_threshold=1e-4) that rejects ~99.7% of LoRA sequences -> grad_norm==0. USE_TIS=1
#     enables it with mis_8b_lora.yaml (veto disabled), never mis.yaml.
#   - --kl-loss-coef 0.01 (not 0.1) and --weight-decay 0.01 (not 0.1): both act directly on the
#     adapter, which is the only thing training.
#   - --sequence-parallel DROPPED (the adapter is not SP-aware).
#   - --optimizer-cpu-offload made opt-in (LoRA optimizer state is tiny).
#   - SGLang static mem fractions are now topology-derived + budget-checked: --mem-fraction-static
#     is a fraction of TOTAL GPU memory per server, so in single-node mode the actor engine's 0.8
#     plus the local GLM's 0.35 summed to 1.15 and OOM'd before step 1. Defaults are now
#     0.45/0.30 single-node, 0.8 (actor only) on the 2-node topology, with a fail-fast if the
#     sum exceeds GPU_FRACTION_BUDGET (0.85).
#   - Rollout fault tolerance ON by default (FAULT_TOLERANCE=0 to disable): long multi-turn
#     rollouts otherwise wedge the whole run on a stuck SGLang engine.
#
#   init model : ACLArena/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau-IF_torch_dist
#                (the end-of-chain multi-stage SFT ckpt; Megatron torch_dist, --load/--ref-load).
#                Override with INIT_MODEL. It has latest_checkpointed_iteration.txt + release/.
#   base (hf)  : Qwen3-8B-Base   (tokenizer/config + rollout init; --hf-checkpoint)
#   user-sim   : GLM-4.7-Flash served IN-CLUSTER as a standalone SGLang OpenAI server on a
#                DEDICATED node (0 training GPUs used by RL). In the DEFAULT 2-node layout it
#                runs on node 1 (discovered via Ray resource 'usersim'); the rollout's
#                LocalUserSimulationEnv reaches it at TAU_USER_SIM_URL=http://<ip>:<port>/v1/chat/completions.
#                In single-node mode the user-sim is served locally on this node.
#   save       : MultiStageRL-LoRA-tau/Qwen3-8B-SDFT-Tau-LoRA
#
# WHY the dedicated user-sim node (2-node topology, NOT slime --sglang-config):
#   The seq-search-LoRA-single run proved the most stable Cluster topology is ONE colocate
#   train+rollout node (all training NCCL intra-node over NVLink -> zero cross-subnet
#   weight-sync fragility) + ONE dedicated 0-GPU-for-RL aux node. Here the aux node runs the
#   GLM user-simulator instead of the faiss retriever. This is DELIBERATELY NOT the MOPD-style
#   `--sglang-config` multi-model split (actor + user_sim in one rollout pool): that path put
#   SGLang engines on separate nodes -> cross-/24 TCPStore rendezvous, and hit the DP
#   nccl_port collision. Serving GLM as a PLAIN standalone SGLang HTTP server (like the
#   retriever) and reaching it over HTTP keeps the actor a single colocated model on node 0.
#   The only cross-node traffic is the user-sim HTTP (:8099) + Ray control-plane join (:6379).
#
# ── Submit (DEFAULT: 2 nodes = 1 colocate train+rollout + 1 dedicated user-sim) ────────────
#   python3 cluster_cli_seq_tau_lora.py batch \
#       --script examples/tau-bench/run_qwen3_8B_lora.sh
#   (num-nodes 2, usersim-nodes 1 are the CLI defaults. Per-role staging: the GPU node gets
#    model+tau data; the user-sim node gets ONLY GLM-4.7-Flash.)
#
# ── Submit (single node, 8 GPU, COLOCATE — user-sim served locally) ────────────────────────
#   python3 cluster_cli_seq_tau_lora.py batch \
#       --script examples/tau-bench/run_qwen3_8B_lora.sh \
#       --num-nodes 1 --usersim-nodes 0
#   (GLM then shares the 8 GPUs with training: actor engine 0.45 + GLM 0.30 static, leaving
#    ~0.25 for the resident Megatron model + activations. Tight; prefer the 2-node topology.)
#
# NOTE: disaggregated weight-sync (UpdateWeightFromDistributed) HARD-ASSERTS TP==1 for
# Qwen3-8B's fused GQA-QKV. This run is COLOCATE (actor + rollout on node 0), which uses the
# UpdateWeightFromTensor path (de-strides QKV) and is fine at TP>1, so we keep TP=4 x CP=2.
set -ex

# ════════════════════════════════════════════════════════════════════════════
# DEDICATED USER-SIM NODE (2-node): the bootstrap sets SLIME_NODE_ROLE=usersim,
# joins this node to Ray with 0 GPU + a 'usersim' resource label, then falls
# through here. We serve GLM-4.7-Flash as a standalone SGLang OpenAI-compatible
# server in the FOREGROUND (blocks for the job's lifetime) and NEVER touch the
# train/rollout path below (in particular NOT the pkill/ray-head logic, which
# would kill the Ray worker bootstrap just started). All roots + the GLM path
# were exported + staged by the bootstrap (ROOT_DIR/MODEL_ROOT).
# ════════════════════════════════════════════════════════════════════════════
if [ "${SLIME_NODE_ROLE:-}" = "usersim" ]; then
  echo "[usersim-node] dedicated user-simulator role."
  U_ROOT_DIR=${ROOT_DIR:-/data/user}
  U_MODEL_ROOT=${MODEL_ROOT:-/shared/user}
  U_PORT=${USER_SIM_PORT:-8099}
  U_MODEL_PATH=${USER_SIM_MODEL_PATH:-${U_MODEL_ROOT}/GLM/GLM-4.7-Flash}
  U_SERVED_NAME=${TAU_USER_MODEL_ID:-user_sim}
  # GLM-4.7-Flash is a 30B-A3B MoE (arch glm4_moe_lite). Per its HF card the SGLang recipe is
  # --tp-size 4 --tool-call-parser glm47 --reasoning-parser glm45 --mem-fraction-static 0.8.
  # We serve it over the node's 8 GPUs (default TP=4 -> DP=2 = 2 engines behind one port).
  U_TP=${USER_SIM_TP:-4}
  U_MEM_FRAC=${USER_SIM_MEM_FRACTION:-0.85}
  U_NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); U_NGPU=${U_NGPU:-8}
  U_DP=$(( U_NGPU / U_TP )); [ "$U_DP" -lt 1 ] && U_DP=1

  if [ ! -d "${U_MODEL_PATH}" ]; then
    echo "FATAL: user-sim model not found at ${U_MODEL_PATH}." >&2
    echo "       Stage it via --stage-model GLM/GLM-4.7-Flash/ (already in s3://YOUR_BUCKET/GLM/)." >&2
    exit 1
  fi

  # Bind to the routable eth0 IP so the train node can reach us (0.0.0.0 also works, but the
  # train node discovers this specific IP via Ray). SGLang serves an OpenAI-compatible API at
  # /v1/chat/completions on --port.
  U_SELF_IP="$(ip -4 -o addr show eth0 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
  [ -z "$U_SELF_IP" ] && U_SELF_IP="0.0.0.0"
  echo "[usersim-node] launching SGLang GLM server: model=${U_MODEL_PATH} served_name=${U_SERVED_NAME}" \
       "tp=${U_TP} dp=${U_DP} port=${U_PORT} host=0.0.0.0 (self_ip=${U_SELF_IP})"

  # Foreground/blocking: the SGLang server owns this node for the whole job. The train node
  # gates on /health before submitting train.py, so it's fine for the load to take minutes.
  exec python3 -m sglang.launch_server \
      --model-path "${U_MODEL_PATH}" \
      --served-model-name "${U_SERVED_NAME}" \
      --host 0.0.0.0 \
      --port "${U_PORT}" \
      --tp-size "${U_TP}" \
      --dp-size "${U_DP}" \
      --tool-call-parser "${USER_SIM_TOOL_PARSER:-glm47}" \
      --reasoning-parser "${USER_SIM_REASONING_PARSER:-glm45}" \
      --mem-fraction-static "${U_MEM_FRAC}" \
      --trust-remote-code \
      ${USER_SIM_EXTRA_SGLANG_ARGS:-}
  # exec replaces the shell; nothing below runs on the user-sim node.
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
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-0}   # >0 => DISAGGREGATED; 0 => COLOCATE (default here)

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
[ "$NVLINK_COUNT" -gt 0 ] && HAS_NVLINK=1 || HAS_NVLINK=0
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

GPU_LIST=(0 1 2 3 4 5 6 7)
export CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
NUM_GPUS=${#GPU_LIST[@]}
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} GPUs/node)"

source "${SCRIPT_DIR}/../../scripts/models/qwen3-8B.sh"   # -> MODEL_ARGS

# ----------------------------------------------------------------------------
# Tau-bench user-sim config. LOCAL in-cluster GLM-4.7-Flash, reached over HTTP.
# The URL is resolved below: for a DEDICATED user-sim node (USER_SIM_REMOTE=1) we
# discover its IP via Ray and point TAU_USER_SIM_URL at it; for single-node we
# launch the server locally and point at 127.0.0.1. generate_with_tau._ensure_user_sim_url
# only fills TAU_USER_SIM_URL from the slime router when it's UNSET — we set it
# explicitly here (standalone server, not a slime --sglang-config model), so it wins.
# ----------------------------------------------------------------------------
export TAU_USER_STRATEGY=${TAU_USER_STRATEGY:-local}
export TAU_USER_MODEL_ID=${TAU_USER_MODEL_ID:-user_sim}    # served model name of the GLM server
export TAU_USER_SIM_MODEL=${TAU_USER_SIM_MODEL:-user_sim}
export TAU_ENV=${TAU_ENV:-retail}
export TAU_TASK_SPLIT=${TAU_TASK_SPLIT:-train}
# no-think for the user-sim. GLM-4.7-Flash defaults thinking ON; leaving this "off" (send no
# kwarg) let GLM burn the whole max_tokens budget inside <think> and return an EMPTY content
# -> "empty completion from local user-sim" on ~60% of turns -> 8x-retry storm -> the first
# rollout batch never fills -> zero train steps (observed on job <JOB_ID>, 2h no step). GLM's
# chat_template.jinja DOES honor enable_thinking ("'</think>' if (enable_thinking is defined
# and not enable_thinking) else '<think>'"), so send {"enable_thinking": false} to force
# no-think -> the user turn is generated directly within budget. (Overridable; "off" reverts.)
export TAU_USER_THINK_KWARG=${TAU_USER_THINK_KWARG:-enable_thinking}
export TAU_USER_MAX_TOKENS=${TAU_USER_MAX_TOKENS:-1024}
export TAU_USER_TEMP=${TAU_USER_TEMP:-0.7}                 # avoid temp-0 echo death-loop
export TAU_ENV_THREAD_WORKERS=${TAU_ENV_THREAD_WORKERS:-256}
USER_SIM_PORT=${USER_SIM_PORT:-8099}

# tau task jsonl (idempotent; the CLI stages tau-bench/ into DATA_ROOT/tau-bench).
TAU_DATA_DIR=${TAU_DATA_DIR:-${DATA_ROOT}/tau-bench}
mkdir -p "${TAU_DATA_DIR}"
if [ ! -s "${TAU_DATA_DIR}/retail_${TAU_TASK_SPLIT}_tasks.jsonl" ]; then
  echo "[tau data] generating retail_{train,test,dev}_tasks.jsonl via tau1_mock.py -> ${TAU_DATA_DIR}"
  ( cd "${SLIME_DIR:-${ROOT_DIR}/slime}/examples/tau-bench" && \
    PYTHONPATH="${SLIME_DIR:-${ROOT_DIR}/slime}/examples/tau-bench:${PYTHONPATH:-}" \
    python3 tau1_mock.py --local_dir "${TAU_DATA_DIR}" )
fi

# ----------------------------------------------------------------------------
# Model / ckpt args
# ----------------------------------------------------------------------------
# --load/--ref-load need Megatron torch_dist (NOT HF safetensors) at the BASE dir.
# Init = the end-of-chain multi-stage SFT checkpoint (Math->Sea->Search->Tau->IF), torch_dist.
# --finetune loads weights only + resets iteration to 0. The LoRA adapters are added on top;
# the base is frozen (only lora_A/lora_B train).
INIT_MODEL=${INIT_MODEL:-${MODEL_ROOT}/ACLArena/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau-IF_torch_dist}
SAVE_DIR=${SAVE_DIR:-${MODEL_ROOT}/MultiStageRL-LoRA-tau/Qwen3-8B-SDFT-Tau-LoRA}
if [ ! -d "${INIT_MODEL}" ]; then
  echo "FATAL: init torch_dist model not found at ${INIT_MODEL}." >&2
  echo "       It must be a Megatron torch_dist checkpoint (HF safetensors won't load)." \
       "Stage it via --stage-model ACLArena/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau-IF_torch_dist/ (already in S3)." >&2
  exit 1
fi

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-64}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}
WANDB_GROUP=${WANDB_GROUP:-tau_SDFT-Tau-LoRA_bs_${ROLLOUT_BATCH_SIZE}}
# Keep the debug dumps inside the S3-synced output subtree (the BatchService bootstrap exports
# TAU_ROLLOUT_DEBUG_ROOT alongside SAVE_DIR so both follow --output-prefix).
ROLLOUT_DEBUG_DIR="${TAU_ROLLOUT_DEBUG_ROOT:-${MODEL_ROOT}/MultiStageRL-LoRA-tau/}${WANDB_GROUP}"

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
   --prompt-data ${TAU_DATA_DIR}/retail_train_tasks.jsonl
   --input-key index
   --rollout-shuffle
   --num-rollout ${NUM_ROLLOUT:-300}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt 8
   --rollout-max-response-len 2048
   --rollout-temperature 1
   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_raw_task_reward_nonzero_std
   --balance-data
)
# Per-rollout debug .pt dumps OFF by default (on Cluster the dump dir is S3-synced, so 300
# rollouts of full-batch tensors re-upload hundreds of GB). Turn on with TAU_SAVE_DEBUG_ROLLOUT=1.
if [ "${TAU_SAVE_DEBUG_ROLLOUT:-0}" = "1" ]; then
  ROLLOUT_ARGS+=(--save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt")
fi

EVAL_ARGS=(
   --eval-interval ${EVAL_INTERVAL:-20}
   --eval-prompt-data retail-dev ${TAU_DATA_DIR}/retail_test_tasks.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 2048
   --eval-temperature 0.7
)

# ----------------------------------------------------------------------------
# Parallelism. Colocate default TP=4 x CP=2 (the original tau run). Disaggregated
# would FORCE TP=1 (Qwen3-8B GQA-QKV disagg weight-sync assert), but this run is
# colocate by default (ROLLOUT_NUM_GPUS=0), which uses UpdateWeightFromTensor and
# handles TP>1 fine.
# ----------------------------------------------------------------------------
if [ "${ROLLOUT_NUM_GPUS:-0}" -gt 0 ]; then
  TP_SIZE=1
  echo "DISAGGREGATED (ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS}) -> forcing TP=1 (GQA-QKV weight-sync assert)."
else
  TP_SIZE=${TP_SIZE:-4}
fi
CP_SIZE=${CP_SIZE:-2}

# ── Sequence parallel: OFF by default for LoRA (SEQ_PARALLEL=0) ──────────────
# With --sequence-parallel + TP>1, TE Column/Row-parallel linears receive a
# SEQUENCE-SCATTERED input (seq/TP) and internally all-gather to full seq before
# the matmul, so their OUTPUT is full-seq. Our MergedLoRALinear adapter computes on
# the raw (scattered) input, giving seq/TP -> base output (full seq) and adapter
# output (seq/TP) mismatch. Making the adapter SP-aware is fragile; since LoRA
# freezes the base (tiny optimizer state) and we keep full recompute, memory is
# ample on H200 without SP. So default SP OFF. CP=2 is unaffected (it slices seq at
# the attention level, consistently for base AND adapter). Original tau run used SP;
# we DROP it here for the LoRA adapter's correctness (same as the search LoRA run).
PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --pipeline-model-parallel-size 1
   --context-parallel-size ${CP_SIZE}
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-9216}
)
if [ "${SEQ_PARALLEL:-0}" = "1" ]; then
  echo "SEQ_PARALLEL=1 -> enabling --sequence-parallel (WARNING: LoRA adapter is NOT SP-aware; will mismatch unless the adapter is fixed)."
  PERF_ARGS+=(--sequence-parallel)
fi

# KL coef: 0.01, NOT the full-FT tau run's 0.1. Under LoRA the base is frozen and
# ref == base, so the KL term pulls the adapter straight back to zero; with only ~0.1%
# of params trainable at lr 1e-6 a coef of 0.1 dominates the policy gradient and pins
# the adapter at its zero init. The working search LoRA run uses 0.01. Overridable.
GRPO_ARGS=(
   --advantage-estimator gspo
   --use-kl-loss
   --kl-loss-coef ${KL_LOSS_COEF:-0.01}
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.25
)
# NOTE: --use-tis is added conditionally in the USE_TIS block below (default OFF for LoRA).

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${LR:-1e-6}
   --lr-decay-style constant
   # 0.01, not the full-FT run's 0.1: weight decay acts directly on lora_A/lora_B (the
   # ONLY trainable params here) and lora_B starts at exactly 0, so a large decay just
   # shrinks the adapter. Matches the search LoRA run.
   --weight-decay ${WEIGHT_DECAY:-0.01}
   --adam-beta1 0.9
   --adam-beta2 0.98
   --use-precision-aware-optimizer
)
# NOTE: LoRA freezes the base, so optimizer state is tiny (only adapters). Unlike the
# original full-FT tau run we do NOT need --optimizer-cpu-offload (kept overridable).
if [ "${OPTIMIZER_CPU_OFFLOAD:-0}" = "1" ]; then
  OPTIMIZER_ARGS+=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d)
fi

WANDB_ARGS=(
   --use-wandb
   --wandb-project ${WANDB_PROJECT:-Seq-train-8B}
   --wandb-group ${WANDB_GROUP}
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
)

# ── GPU memory budget (COLOCATE) ─────────────────────────────────────────────
# In colocate mode (--no-offload-train --no-offload-rollout) the actor's SGLang engine AND
# the resident Megatron train model live on the same 8 GPUs — and in SINGLE-NODE mode the
# LOCAL GLM user-sim server does too. --mem-fraction-static is a fraction of TOTAL GPU
# memory for EACH server, so the fractions ADD UP. The inherited full-FT defaults (actor
# 0.8) plus the local user-sim's 0.35 summed to 1.15 -> guaranteed OOM before step 1, i.e.
# the documented single-node fallback was unusable. Pick defaults from the topology and
# hard-fail if the sum leaves no headroom for Megatron weights/activations.
USER_SIM_LOCAL=0
if [ "${TAU_USER_STRATEGY}" = "local" ] && [ "${USER_SIM_REMOTE:-0}" != "1" ]; then
  USER_SIM_LOCAL=1
fi
if [ "${USER_SIM_LOCAL}" = "1" ]; then
  # Single node: actor engine + GLM + train model all share these GPUs.
  SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.45}
  USER_SIM_MEM_FRACTION=${USER_SIM_MEM_FRACTION:-0.30}
else
  # 2-node: GLM is on the dedicated node, so only the actor engine + train model are here.
  SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.8}
  USER_SIM_MEM_FRACTION=${USER_SIM_MEM_FRACTION:-0.85}
fi
GPU_FRACTION_BUDGET=${GPU_FRACTION_BUDGET:-0.85}
MEM_SUM=$(awk -v a="${SGLANG_MEM_FRACTION}" -v b="${USER_SIM_MEM_FRACTION}" -v loc="${USER_SIM_LOCAL}" \
            'BEGIN{printf "%.3f", a + (loc==1 ? b : 0)}')
echo "GPU mem fractions on this node: actor_sglang=${SGLANG_MEM_FRACTION}" \
     "local_user_sim=$([ "${USER_SIM_LOCAL}" = "1" ] && echo "${USER_SIM_MEM_FRACTION}" || echo "0 (remote node)")" \
     "sum=${MEM_SUM} budget=${GPU_FRACTION_BUDGET}"
if awk -v s="${MEM_SUM}" -v m="${GPU_FRACTION_BUDGET}" 'BEGIN{exit !(s > m)}'; then
  echo "FATAL: static SGLang memory fractions on this node sum to ${MEM_SUM} > ${GPU_FRACTION_BUDGET}." >&2
  echo "       Nothing would be left for the resident Megatron model + activations (colocate)." >&2
  echo "       Lower SGLANG_MEM_FRACTION / USER_SIM_MEM_FRACTION, or use the 2-node topology" >&2
  echo "       (--usersim-nodes 1) so GLM does not share these GPUs. Raise GPU_FRACTION_BUDGET" >&2
  echo "       only if you know the model actually fits." >&2
  exit 1
fi

# SGLang rollout engine. On p5en the custom all-reduce kernel SIGSEGVs during cuda-graph
# capture; disable custom all-reduce by DEFAULT (small speed hit on 8-GPU TP, keeps cuda graph).
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE:-${NUM_GPUS}}
   --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION}
)
[ "${SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-1}" = "1" ] && SGLANG_ARGS+=(--sglang-disable-custom-all-reduce)
[ "${SGLANG_DISABLE_CUDA_GRAPH:-0}" = "1" ]        && SGLANG_ARGS+=(--sglang-disable-cuda-graph)
# Cap the cuda-graph batch sizes (smaller = faster capture + less capture-time VRAM). OPT-IN:
# --sglang-cuda-graph-bs is auto-derived from the installed SGLang's ServerArgs, so it only
# exists if that build exposes --cuda-graph-bs. Set SGLANG_CUDA_GRAPH_MAX_BS to enable.
if [ -n "${SGLANG_CUDA_GRAPH_MAX_BS:-}" ]; then
  SGLANG_ARGS+=(--sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 ${SGLANG_CUDA_GRAPH_MAX_BS}))
fi

# ── Rollout fault tolerance ──────────────────────────────────────────────────
# Multi-turn tau rollouts are long; a wedged SGLang engine otherwise hangs the whole run
# until the Ray/NCCL timeout. Periodic /health_generate probes kill + restart it instead.
# first-wait covers model compilation / cuda-graph capture. Disable with FAULT_TOLERANCE=0.
FAULT_TOLERANCE_ARGS=()
if [ "${FAULT_TOLERANCE:-1}" = "1" ]; then
  FAULT_TOLERANCE_ARGS=(
     --use-fault-tolerance
     --rollout-health-check-interval ${ROLLOUT_HEALTH_CHECK_INTERVAL:-30}
     --rollout-health-check-timeout ${ROLLOUT_HEALTH_CHECK_TIMEOUT:-60}
     --rollout-health-check-first-wait ${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-120}
  )
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --distributed-timeout-minutes 60
)

# ── LoRA configuration (identical mechanism to the search-r1 LoRA run) ──
export LORA_RANK=${LORA_RANK:-16}
export LORA_ALPHA=${LORA_ALPHA:-32}
export LORA_DROPOUT=${LORA_DROPOUT:-0.0}
export LORA_TARGETS=${LORA_TARGETS:-"linear_qkv,linear_proj,linear_fc1,linear_fc2"}
echo "LoRA: rank=${LORA_RANK} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT} targets=${LORA_TARGETS}"

# Apply the common.py LoRA sync-filter patch (idempotent — checks if already patched). The
# ACLArena common.py already carries the filter, but a fresh image copy might not, so guard.
python3 -c "
import sys
for f in ('/root/slime/slime/backends/megatron_utils/update_weight/common.py',
          '${SLIME_DIR:-${ROOT_DIR}/slime}/slime/backends/megatron_utils/update_weight/common.py'):
    try:
        c=open(f).read()
    except FileNotFoundError:
        continue
    if 'lora_A' in c:
        print('common.py already patched for LoRA:', f); continue
    old='        for name, param in model_module.named_parameters():\n            yield _compute_fqn(name), param'
    new='        for name, param in model_module.named_parameters():\n            if \"lora_A\" in name or \"lora_B\" in name:\n                continue\n            yield _compute_fqn(name), param'
    c=c.replace(old,new,1)
    old2='        for name, param in model_module.named_parameters():\n            # for model without ddp wrap\n            if not name.startswith(\"module.module.\"):'
    new2='        for name, param in model_module.named_parameters():\n            if \"lora_A\" in name or \"lora_B\" in name:\n                continue\n            # for model without ddp wrap\n            if not name.startswith(\"module.module.\"):'
    c=c.replace(old2,new2,1)
    open(f,'w').write(c)
    print('Patched common.py for LoRA filter:', f)
"

# ── TIS / train-infer mismatch correction (DEFAULT OFF for LoRA) ──────────────
# The full-FT tau run used mis.yaml (use_rs=true, rs_veto_threshold=1.0e-4). That config
# is a LoRA GRADIENT KILLER: calculate_veto_mask (mis.py) is SEQUENCE-level — a single
# token with train_lp - rollout_lp < log(1e-4) zeroes the whole sequence's loss mask.
# The LoRA train-engine-vs-SGLang logprob mismatch is ~2x larger than full-FT
# (mis_ppl_ratio ~1.9 vs ~1.25), which made that veto reject ~99.7% of sequences ->
# whole-batch mask == 0 -> train/grad_norm == 0 -> the adapter never trained. tau is
# multi-turn long-context, so the mismatch is at least as bad as the search run's.
#
# So: default USE_TIS=0 and wire NONE of it — no --use-tis, no --custom-config-path (the
# YAML re-enables use_tis AFTER arg-parse and would override the CLI), no
# --custom-tis-function-path. The policy loss then uses the raw per-token loss_mask.
# USE_TIS=1 enables it with mis_8b_lora.yaml (use_rs=false, veto=null, upper_bound=4.0) —
# NOT mis.yaml. Override the config with MIS_CONFIG if you really want the strict one.
#
# The plain mis.compute_mis_weights_with_cp (not the search run's _no_drift variant) is
# correct here: trainable_agents._build_training_tensor already sets loss_mask=0 and
# rollout_log_probs=0.0 for every env/user-injected token, so there is no drift to mask.
TIS_ARGS=()
if [ "${USE_TIS:-0}" = "1" ]; then
  MIS_CONFIG=${MIS_CONFIG:-examples/train_infer_mismatch_helper/mis_8b_lora.yaml}
  echo "USE_TIS=1 -> enabling TIS (config=${MIS_CONFIG})."
  case "${MIS_CONFIG}" in
    *mis.yaml|*mis_8b.yaml)
      echo "WARNING: ${MIS_CONFIG} has use_rs=true + rs_veto_threshold=1.0e-4, which zeroes" \
           "essentially every LoRA sequence (grad_norm==0). Prefer mis_8b_lora.yaml." >&2
      ;;
  esac
  GRPO_ARGS+=(--use-tis)
  TIS_ARGS=(
     --custom-config-path ${MIS_CONFIG}
     --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
  )
else
  echo "USE_TIS=0 (default) -> TIS DISABLED for LoRA (no --use-tis / --custom-config-path /" \
       "--custom-tis-function-path); raw loss_mask drives the gradient."
fi

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_tau.generate
   # TIS-related args (only present when USE_TIS=1; see the block above).
   "${TIS_ARGS[@]}"
   # LoRA: reuse the search-r1 provider + hooks (they are generic, not search-specific).
   --custom-model-provider-path lora_model_provider.custom_model_provider
   --only-train-params-name-list 'lora_A' 'lora_B'
)

# ----------------------------------------------------------------------------
# Ray + env. PYTHONPATH must include the tau-bench example dir (generate/agent/env),
# the search-r1 dir (lora_model_provider + lora_hooks live there), and slime/Megatron.
# Rollout actors on worker nodes don't inherit this shell env, so it's ALSO injected
# into the Ray job runtime-env below.
# ----------------------------------------------------------------------------
export PYTHONPATH="${SLIME_DIR:-${ROOT_DIR}/slime}/examples/tau-bench:${SLIME_DIR:-${ROOT_DIR}/slime}/examples/search-r1:${SLIME_DIR:-${ROOT_DIR}/slime}:${MEGATRON_DIR:-${ROOT_DIR}/Megatron-LM}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
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

# ── GPU keepalive during the user-sim-warmup wait (2-node only) ──
# When USER_SIM_REMOTE=1 the train node BLOCKS below (user-sim discovery + readiness gate)
# for minutes while the dedicated user-sim node loads GLM (~250GB). Its 8 GPUs are idle that
# whole time, which Cluster's GTLStuckGPULambda flags as "stuck" (idle-GPU) and kills at
# ~11-16min. Run a light matmul loop to keep GPU util non-zero until we kill it right before
# `ray job submit`. Only in remote mode; harmless no-op elsewhere. Disable with GPU_KEEPALIVE=0.
KEEPALIVE_PID=""
if [ "${GPU_KEEPALIVE:-1}" = "1" ] && [ "${USER_SIM_REMOTE:-0}" = "1" ]; then
  python3 "${SLIME_DIR:-${ROOT_DIR}/slime}/examples/search-r1/gpu_keepalive.py" > "${ROOT_DIR}/gpu_keepalive.log" 2>&1 &
  KEEPALIVE_PID=$!
  echo "[gpu-keepalive] started pid=${KEEPALIVE_PID} (defeats stuck-detector during user-sim warmup; killed before ray job submit)."
  trap 'kill "${KEEPALIVE_PID}" 2>/dev/null || true' EXIT
fi

# ----------------------------------------------------------------------------
# User-sim provisioning. Two modes:
#   (A) USER_SIM_REMOTE=1 : a DEDICATED user-sim node is up (2-node). Don't launch
#       locally — discover its IP from Ray (the node carrying the 'usersim' resource)
#       and point TAU_USER_SIM_URL at it.
#   (B) otherwise         : launch the GLM SGLang server LOCALLY on THIS (main) node
#       in the background (single-node). NOTE: single-node co-hosts GLM + training on
#       the same 8 GPUs — only viable if VRAM allows (H200 141GB). Prefer the 2-node
#       topology for real runs.
# ----------------------------------------------------------------------------
USER_SIM_PID=""
if [ "${TAU_USER_STRATEGY}" = "local" ] && [ "${USER_SIM_REMOTE:-0}" = "1" ]; then
  echo "[usersim] REMOTE mode — discovering dedicated user-sim node IP from Ray (resource 'usersim')."
  US_IP=""
  for i in $(seq 1 60); do
    US_IP=$(python3 -c "
import ray
ray.init(address='auto', logging_level='ERROR')
ip=''
for n in ray.nodes():
    if n.get('Alive') and float(n.get('Resources',{}).get('usersim',0))>0:
        ip=n['NodeManagerAddress']; break
print(ip)
ray.shutdown()
" 2>/dev/null | tail -1)
    [ -n "$US_IP" ] && break
    [ $(( i % 6 )) -eq 1 ] && echo "  [usersim] user-sim node not registered in Ray yet (attempt $i/60); retrying"
    sleep 10
  done
  if [ -z "$US_IP" ]; then
    echo "FATAL: USER_SIM_REMOTE=1 but no Ray node with resource 'usersim' after 600s." >&2
    ray stop --force 2>/dev/null || true; exit 1
  fi
  export TAU_USER_SIM_URL="http://${US_IP}:${USER_SIM_PORT}/v1/chat/completions"
  echo "[usersim] remote user-sim at ${US_IP}:${USER_SIM_PORT} -> TAU_USER_SIM_URL=${TAU_USER_SIM_URL}"

elif [ "${TAU_USER_STRATEGY}" = "local" ]; then
  # SINGLE-NODE: serve GLM locally in the background. It shares this node's 8 GPUs with
  # training, so mem-fraction is small; readiness gated before submit.
  U_MODEL_PATH=${USER_SIM_MODEL_PATH:-${MODEL_ROOT}/GLM/GLM-4.7-Flash}
  U_TP=${USER_SIM_TP:-4}
  U_DP=$(( NUM_GPUS / U_TP )); [ "$U_DP" -lt 1 ] && U_DP=1
  if [ ! -d "${U_MODEL_PATH}" ]; then
    echo "FATAL: user-sim model not found at ${U_MODEL_PATH}; stage GLM/GLM-4.7-Flash/ or set USER_SIM_MODEL_PATH." >&2
    ray stop --force 2>/dev/null || true; exit 1
  fi
  echo "[usersim] LOCAL mode — launching GLM SGLang server on 127.0.0.1:${USER_SIM_PORT} (tp=${U_TP} dp=${U_DP}; shares GPUs with training, mem-fraction=${USER_SIM_MEM_FRACTION})."
  python3 -m sglang.launch_server \
      --model-path "${U_MODEL_PATH}" \
      --served-model-name "${TAU_USER_MODEL_ID}" \
      --host 127.0.0.1 --port "${USER_SIM_PORT}" \
      --tp-size "${U_TP}" --dp-size "${U_DP}" \
      --tool-call-parser "${USER_SIM_TOOL_PARSER:-glm47}" \
      --reasoning-parser "${USER_SIM_REASONING_PARSER:-glm45}" \
      --mem-fraction-static "${USER_SIM_MEM_FRACTION}" \
      --trust-remote-code \
      > "${ROOT_DIR}/usersim.log" 2>&1 &
  USER_SIM_PID=$!
  export TAU_USER_SIM_URL="http://127.0.0.1:${USER_SIM_PORT}/v1/chat/completions"
  echo "[usersim] pid=${USER_SIM_PID}; loading GLM in background (readiness gated before submit). url=${TAU_USER_SIM_URL}"
  trap 'kill "${USER_SIM_PID}" 2>/dev/null || true; [ -n "${KEEPALIVE_PID:-}" ] && kill "${KEEPALIVE_PID}" 2>/dev/null || true' EXIT
fi

# ----------------------------------------------------------------------------
# Multi-node: wait until ALL training GPUs register with Ray before submit. In the
# 2-node topology the user-sim node joins with 0 GPU (resource 'usersim'), so we only
# expect ACTOR_NUM_NODES*NUM_GPUS training GPUs (the user-sim node contributes none).
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
# The rollout drives the tau env (LocalUserSimulationEnv reads TAU_USER_SIM_URL etc), so tau
# env vars + PYTHONPATH must be injected into the Ray job runtime-env.
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
    \"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN\": \"1\",
    \"TAU_USER_STRATEGY\": \"${TAU_USER_STRATEGY}\",
    \"TAU_USER_MODEL_ID\": \"${TAU_USER_MODEL_ID}\",
    \"TAU_USER_SIM_MODEL\": \"${TAU_USER_SIM_MODEL}\",
    \"TAU_USER_SIM_URL\": \"${TAU_USER_SIM_URL:-}\",
    \"TAU_ENV\": \"${TAU_ENV}\",
    \"TAU_TASK_SPLIT\": \"${TAU_TASK_SPLIT}\",
    \"TAU_USER_THINK_KWARG\": \"${TAU_USER_THINK_KWARG}\",
    \"TAU_USER_MAX_TOKENS\": \"${TAU_USER_MAX_TOKENS}\",
    \"TAU_USER_TEMP\": \"${TAU_USER_TEMP}\",
    \"TAU_ENV_THREAD_WORKERS\": \"${TAU_ENV_THREAD_WORKERS}\"${MULTINODE_ENV}
  }
}"

# ----------------------------------------------------------------------------
# Gate on user-sim readiness (it has been loading since ray start, overlapped with
# the GPU-registration wait). Only now — right before submit — require /health.
# ----------------------------------------------------------------------------
if [ "${TAU_USER_STRATEGY}" = "local" ]; then
  # Health endpoint = the server root (SGLang serves /health). Derive host:port from
  # TAU_USER_SIM_URL (strip the /v1/chat/completions suffix).
  GATE_BASE="${TAU_USER_SIM_URL%/v1/chat/completions}"
  echo "[usersim] waiting for ${GATE_BASE}/health to answer (GLM load ~minutes) ..."
  US_OK=0
  for i in $(seq 1 150); do
    if curl -sf -m 10 "${GATE_BASE}/health" > /dev/null 2>&1 \
       || curl -sf -m 10 "${GATE_BASE}/v1/models" > /dev/null 2>&1; then
      echo "[usersim] ready after ${i} checks"; US_OK=1; break
    fi
    # Local-launch mode: fail fast if the background server process died.
    if [ -n "${USER_SIM_PID:-}" ]; then
      kill -0 "${USER_SIM_PID}" 2>/dev/null || { echo "FATAL: local user-sim process died; see ${ROOT_DIR}/usersim.log" >&2; tail -n 40 "${ROOT_DIR}/usersim.log" || true; ray stop --force 2>/dev/null || true; exit 1; }
    fi
    [ $(( i % 6 )) -eq 1 ] && echo "  [usersim] not ready yet (check $i/150) — GLM still loading"
    sleep 10
  done
  [ "${US_OK}" != "1" ] && { echo "FATAL: user-sim not ready after 1500s (base=${GATE_BASE})." >&2; ray stop --force 2>/dev/null || true; exit 1; }
fi

# Stop the GPU keepalive NOW — user-sim is ready, real training is about to start and needs
# all GPU memory. Kill + wait so its resident tensors are freed before train.py.
if [ -n "${KEEPALIVE_PID:-}" ]; then
  echo "[gpu-keepalive] stopping pid=${KEEPALIVE_PID} before training."
  kill "${KEEPALIVE_PID}" 2>/dev/null || true
  wait "${KEEPALIVE_PID}" 2>/dev/null || true
  sleep 2
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
[ -n "${USER_SIM_PID:-}" ] && kill "${USER_SIM_PID}" 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
