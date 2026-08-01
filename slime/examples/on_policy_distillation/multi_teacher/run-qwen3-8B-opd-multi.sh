#!/bin/bash
# 3-teacher on-policy distillation (pure OPD, task reward = 0) — CLUSTER / multi-node.
#
#   student : ACLArena/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau-IF  (megatron torch_dist, --load)
#   teachers (the specialists from the math2sea / sea2tau / tau2if experiments), served IN-CLUSTER
#   as FROZEN (update_weights:false) models in the generated --sglang-config, resolved by name via
#   slime's get_model_url (NO hardcoded IPs — works on isolated Cluster batch nodes):
#     * math   : ACLArena/Qwen3-8B-Base-Math                     -> sglang model "teacher_math"
#     * search : ACLArena/Qwen3-8B-Base-SeaSFT-Search            -> sglang model "teacher_search"
#     * tau    : ACLArena/Qwen3-8B-Base-TauSFT-Tau               -> sglang model "teacher_tau"
#
# One student rolls out over a MIXED math+search+tau prompt stream; each trajectory is scored by the
# teacher that OWNS its domain (domain-routing, see opd_multi_teacher.py). Goal: distill all three
# specialties back into the end-of-chain IF model, which has drifted from math, search AND tau.
#
# ── Cluster topology (disaggregated; all teachers OFF the training GPUs) ──────────────────────
#   Submit:  python3 cluster_cli_sdft.py batch \
#              --script examples/on_policy_distillation/multi_teacher/run-qwen3-8B-opd-multi.sh \
#              --num-nodes 3 --rollout-nodes 2 --user-sim-nodes 1 \
#              --stage-model ACLArena/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau-IF_torch_dist/ \
#              --stage-model ACLArena/Qwen3-8B-Base-Math/ \
#              --stage-model ACLArena/Qwen3-8B-Base-SeaSFT-Search/ \
#              --stage-model ACLArena/Qwen3-8B-Base-TauSFT-Tau/ \
#              --stage-model Qwen3-8B-Base/ --stage-model GLM/GLM-4.7-Flash/ --stage-model e5-base-v2/ \
#              --stage-data dapo-math-17k/ --stage-data tau-bench/ \
#              --stage-data nq_hotpotqa_train/ --stage-data wiki-18/
#   Node 0 (train, 8 GPU): student TP4 x CP2 x PP1 = 8. Also runs the CPU-faiss search retriever.
#   Nodes 1-2 (rollout pool, ROLLOUT_NUM_GPUS=16): actor(5) + user_sim/GLM(8) + 3 teachers(1 each) = 16.
#     (The sdft CLI forces USER_SIM_NUM_GPUS = user_sim_nodes*8 = 8; teachers are carved out of the
#      actor share inside the generated YAML. sum(num_gpus) MUST equal ROLLOUT_NUM_GPUS — slime asserts.)
#
# The three teachers + the GLM user-sim are frozen extra models; only "actor" receives student weight
# updates (slime auto-excludes update_weights:false models from the actor->rollout weight sync).
set -ex

# ── Overridable roots (Cluster BatchService bootstrap sets these to the NVMe mirror; dev defaults kept) ──
ROOT_DIR=${ROOT_DIR:-/data/user}
MODEL_ROOT=${MODEL_ROOT:-/shared/user}
DATA_ROOT=${DATA_ROOT:-${ROOT_DIR}}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SLIME_DIR:-${ROOT_DIR}/slime}"   # train.py / tools / examples paths resolve from the slime root

export PYTHONBUFFERED=16

# ── Multi-node hooks (Cluster bootstrap exports these; single-node/dev defaults preserved) ──
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-0}
USER_SIM_NUM_GPUS=${USER_SIM_NUM_GPUS:-0}

# This run REQUIRES disaggregated mode: the frozen teachers + GLM user-sim live in the rollout GPU
# pool (there is no room for them under --colocate, which locks rollout to the actor GPUs).
if [ "${ROLLOUT_NUM_GPUS:-0}" -le 0 ]; then
  echo "FATAL: multi-teacher OPD needs DISAGGREGATED rollout (ROLLOUT_NUM_GPUS>0). Submit with" \
       "--rollout-nodes >= 2 --user-sim-nodes 1 (see the header)." >&2
  exit 1
fi
if [ "${USER_SIM_NUM_GPUS:-0}" -le 0 ]; then
  echo "FATAL: the tau user-sim (GLM) needs reserved GPUs (USER_SIM_NUM_GPUS>0). Submit with" \
       "--user-sim-nodes 1." >&2
  exit 1
fi

# ----------------------------------------------------------------------------
# Tau user-sim: LOCAL in-cluster GLM-4.7-Flash (resolved from the sglang router by generate_with_tau).
# ----------------------------------------------------------------------------
export TAU_USER_STRATEGY=${TAU_USER_STRATEGY:-local}
export TAU_USER_MODEL_ID=${TAU_USER_MODEL_ID:-user_sim}   # served model name of the frozen user_sim entry
export TAU_USER_SIM_MODEL=${TAU_USER_SIM_MODEL:-user_sim} # named model in --sglang-config to route to
export TAU_ENV=${TAU_ENV:-retail}
export TAU_TASK_SPLIT=${TAU_TASK_SPLIT:-train}
export TAU_ENABLE_THINKING=${TAU_ENABLE_THINKING:-0}
export TAU_STRIP_HISTORICAL_THINK=${TAU_STRIP_HISTORICAL_THINK:-1}
export TAU_USER_THINK_KWARG=${TAU_USER_THINK_KWARG:-off}
export TAU_USER_MAX_TOKENS=${TAU_USER_MAX_TOKENS:-1024}

# ----------------------------------------------------------------------------
# Search retriever: in-cluster CPU-faiss server on the MAIN node (rollout nodes reach it via MASTER_ADDR).
# ----------------------------------------------------------------------------
export SEARCH_R1_STRIP_THINK=${SEARCH_R1_STRIP_THINK:-0}
RETRIEVE_PORT=${RETRIEVE_PORT:-8000}
export SEARCH_R1_SEARCH_URL=${SEARCH_R1_SEARCH_URL:-http://${MASTER_ADDR}:${RETRIEVE_PORT}/retrieve}
# CPU-faiss brute-force + CPU e5 encoder can bottleneck rollout; tune down if search dominates.
export SEARCH_R1_CONCURRENCY=${SEARCH_R1_CONCURRENCY:-64}
WIKI_INDEX=${WIKI_INDEX:-${DATA_ROOT}/wiki-18/e5_Flat.index}
WIKI_CORPUS=${WIKI_CORPUS:-${DATA_ROOT}/wiki-18/wiki-18.jsonl}
E5_MODEL=${E5_MODEL:-${MODEL_ROOT}/e5-base-v2}
OPD_TEACHER_TIMEOUT=${OPD_TEACHER_TIMEOUT:-600}
export OPD_TEACHER_TIMEOUT

# ----------------------------------------------------------------------------
# Data: build the mixed (math + search pre-templated + tau index) dataset once (main node only).
# ----------------------------------------------------------------------------
# tau task jsonl (idempotent; skip if staged via --stage-data tau-bench/)
TAU_DATA_DIR="${DATA_ROOT}/tau-bench"
mkdir -p "${TAU_DATA_DIR}"
if [ ! -s "${TAU_DATA_DIR}/retail_${TAU_TASK_SPLIT}_tasks.jsonl" ]; then
  echo "[tau data] generating retail_{train,test,dev}_tasks.jsonl via tau1_mock.py -> ${TAU_DATA_DIR}"
  ( cd "${SLIME_DIR:-${ROOT_DIR}/slime}/examples/tau-bench" && \
    PYTHONPATH="${SLIME_DIR:-${ROOT_DIR}/slime}/examples/tau-bench:${PYTHONPATH}" \
    python3 tau1_mock.py --local_dir "${TAU_DATA_DIR}" )
fi

MIXED_DATA=${MODEL_ROOT}/ACLArena/opd_mixed3/train.jsonl
if [ ! -f "${MIXED_DATA}" ]; then
  PYTHONPATH="${SCRIPT_DIR}:${SLIME_DIR:-${ROOT_DIR}/slime}/examples/search-r1:${SLIME_DIR:-${ROOT_DIR}/slime}/examples/tau-bench:${SLIME_DIR:-${ROOT_DIR}/slime}:${PYTHONPATH}" \
  python3 "${SCRIPT_DIR}/prepare_opd_mixed_data.py" \
    --hf ${MODEL_ROOT}/Qwen3-8B-Base \
    --math-jsonl     ${DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl \
    --search-parquet ${DATA_ROOT}/nq_hotpotqa_train/train.parquet \
    --tau-jsonl      ${TAU_DATA_DIR}/retail_train_tasks.jsonl \
    --per-domain-target ${PER_DOMAIN_TARGET:-3000} \
    --out "${MIXED_DATA}"
fi

# ----------------------------------------------------------------------------
# EVAL data (per-domain, pre-staged): math=AIME-2024, search=NQ/HotpotQA test_small
# subsample, tau=retail test. Staged to ${DATA_ROOT}/opd_eval/ (see cluster_cli_mopd
# --stage-data opd_eval/). Prompts are ALREADY chat-templated offline by
# prepare_opd_eval_data.py (math+search) — this run has no global --apply-chat-template —
# and tau is a bare index. Toggle the whole eval off with OPD_ENABLE_EVAL=0.
# ----------------------------------------------------------------------------
OPD_ENABLE_EVAL=${OPD_ENABLE_EVAL:-1}
EVAL_DATA_DIR=${EVAL_DATA_DIR:-${DATA_ROOT}/opd_eval}
EVAL_CFG="${SCRIPT_DIR}/eval_multi.generated.yaml"
if [ "${OPD_ENABLE_EVAL}" = "1" ]; then
  # All three eval files must be present (staged). Fail loud if a path is missing so a
  # bad --stage-data doesn't silently disable a domain's eval.
  for f in eval_math.jsonl eval_search.jsonl eval_tau.jsonl; do
    if [ ! -s "${EVAL_DATA_DIR}/${f}" ]; then
      echo "FATAL: eval file ${EVAL_DATA_DIR}/${f} missing/empty. Stage it via" \
           "cluster_cli_mopd --stage-data opd_eval/ (or set OPD_ENABLE_EVAL=0 to disable eval)." >&2
      exit 1
    fi
  done
  # Generate the eval-config YAML with the resolved container paths (mirrors the
  # in-script sglang-config generation). n_samples/temp/max_len per domain match each
  # domain's native rollout; the per-dataset custom_generate_function_path routes eval
  # to generate_eval_mixed.* (real task reward, bypassing the OPD teacher RM).
  cat > "${EVAL_CFG}" <<EVALYAML
# AUTO-GENERATED by run-qwen3-8B-opd-multi.sh — do not edit by hand.
eval:
  defaults:
    n_samples_per_eval_prompt: 1
  datasets:
    - name: math-aime2024
      path: ${EVAL_DATA_DIR}/eval_math.jsonl
      input_key: prompt
      label_key: label
      n_samples_per_eval_prompt: ${EVAL_MATH_N:-4}
      temperature: 0.6
      max_response_len: 8192
      custom_generate_function_path: generate_eval_mixed.generate_eval_math
    - name: search-nq
      path: ${EVAL_DATA_DIR}/eval_search.jsonl
      input_key: prompt
      label_key: reward_model
      temperature: 1.0
      max_response_len: 4096
      custom_generate_function_path: generate_eval_mixed.generate_eval_search
    - name: tau-retail
      path: ${EVAL_DATA_DIR}/eval_tau.jsonl
      input_key: index
      temperature: 0.7
      max_response_len: 2048
      custom_generate_function_path: generate_eval_mixed.generate_eval_tau
EVALYAML
  echo "[eval-config] generated ${EVAL_CFG}:"
  cat "${EVAL_CFG}"
fi

# ----------------------------------------------------------------------------
# Model / parallelism args
# ----------------------------------------------------------------------------
GPU_LIST=(0 1 2 3 4 5 6 7)
export CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
NUM_GPUS=${#GPU_LIST[@]}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
[ "$NVLINK_COUNT" -gt 0 ] && HAS_NVLINK=1 || HAS_NVLINK=0

source "${SCRIPT_DIR}/../../../scripts/models/qwen3-8B.sh"   # -> MODEL_ARGS

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}

STUDENT=${MODEL_ROOT}/ACLArena/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau-IF_torch_dist

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen3-8B-Base                # tokenizer + config (+ actor rollout init)
   --ref-load      ${STUDENT}                                 # (KL-loss-coef=0, so ref is inert; mirror student)
   --load          ${STUDENT}                                 # STUDENT (init point, NOT a resume)
   --no-load-optim
   --no-load-rng
   --finetune
   --save          ${MODEL_ROOT}/ACLArena/OPD/Qwen3-8B-IF_OPD_MathSearchTau/
   --save-interval ${SAVE_INTERVAL:-5}
)

ROLLOUT_ARGS=(
   --prompt-data ${MIXED_DATA}
   --input-key prompt          # NOTE: NO --apply-chat-template (math+search pre-templated; tau=index)
   --rollout-shuffle           # mixes math+search+tau within every rollout batch -> all teachers active each step
   --num-rollout ${NUM_ROLLOUT:-300}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt 8    # reward=0 -> grouping is neutral; can lower to save rollout cost
   --rollout-max-response-len 8192   # max(math 8192, search 4096, tau 2048)
   --rollout-temperature 1
   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data
)

# ---- eval (per-domain REAL task metrics; every EVAL_INTERVAL rollouts) ----
# Each dataset in eval_multi.generated.yaml sets its own custom_generate_function_path
# -> generate_eval_mixed.generate_eval_{math,search,tau}, which runs the native rollout
# and fills the REAL task reward (\boxed grade / EM / tau success), bypassing the OPD
# teacher RM (async_rm prefers --custom-rm-path only when reward is still None). wandb:
# eval/math-aime2024, eval/search-nq, eval/tau-retail (+ per-metric sub-keys).
# NOTE eval is NOT free here: tau eval (115 tasks) is GLM-user-sim-bound like train
# rollout, so each eval adds real wall-clock. EVAL_INTERVAL=10 per request; raise it or
# set OPD_ENABLE_EVAL=0 if eval dominates. --skip-eval-before-train avoids a cold eval at
# step 0 (set OPD_SKIP_EVAL_BEFORE_TRAIN=0 to get a baseline eval first).
EVAL_ARGS=()
if [ "${OPD_ENABLE_EVAL}" = "1" ]; then
  EVAL_ARGS=(
     --eval-interval ${EVAL_INTERVAL:-10}
     --eval-config "${EVAL_CFG}"
  )
  [ "${OPD_SKIP_EVAL_BEFORE_TRAIN:-1}" = "1" ] && EVAL_ARGS+=(--skip-eval-before-train)
fi

# OPD reward: per-sample teacher logprobs (domain-routed to teacher_{math,search,tau}), task reward = 0.
RM_ARGS=(
   --custom-rm-path opd_multi_teacher.reward_func
   --custom-reward-post-process-path opd_multi_teacher.post_process_rewards
)
# IMPORTANT: do NOT add the tau reward-std dynamic-sampling filter
# (check_raw_task_reward_nonzero_std) — with reward=0 it would drop every group.

PERF_ARGS=(
   # TP=1 (job <JOB_ID> fix — the definitive one). The DISAGGREGATED weight-sync path
   # (UpdateWeightFromDistributed, chosen because we're NOT --colocate) calls all_gather_param()
   # which HARD-ASSERTS partition_stride == 1 (update_weight/common.py:37) and has NO de-interleave
   # logic. At ANY TP>1, Qwen3-8B's GQA fused-QKV (num_query_groups=8) is sharded with
   # partition_stride>1, so the assert fires on the QKV param. Empirically confirmed at BOTH TP=4
   # (job <JOB_ID>/694552) AND TP=2 (job <JOB_ID>: converted 1 param, then died on the QKV) — so TP=2
   # is NOT enough, and --sequence-parallel is irrelevant. The colocate path (UpdateWeightFromTensor
   # -> convert_qwen2_to_hf) de-strides via view(num_query_groups,...)/split, but we can't colocate
   # (no room for the 3 frozen teachers + GLM). At TP=1 there is NO tensor-parallel partitioning, so
   # partition_stride is trivially 1 and the assert cannot fire — zero-risk. Train node = 8 GPU:
   # TP1 x CP2 x PP1 -> DP=4 (fills 8). 8B at TP=1 fits comfortably on an H200 (141 GB).
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 12288        # >= longest single sequence / CP; math resp up to 8192 (+ short prompt)
)

# OPD on top of GRPO; the only signal is the OPD KL penalty.
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef ${OPD_KL_COEF:-0.5}
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${LR:-1e-6}
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# W&B: upload to the public cloud (api.wandb.ai) under a dedicated MOPD project, mirroring
# examples/SDFT. The bootstrap injects WANDB_API_KEY into the container env; slime only
# inits wandb when --use-wandb is passed (default off), so this block is what turns it on.
WANDB_ARGS=(
   --use-wandb
   --wandb-project MOPD
   --wandb-group ${WANDB_GROUP:-mopd_Qwen3-8B_math_search_tau}
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
)

# ── Generate the multi-model --sglang-config: actor + user_sim(GLM) + 3 frozen teachers ──
# sum(num_gpus) MUST equal ROLLOUT_NUM_GPUS (slime rollout.py asserts). Teachers are 8B TP1
# scoring-only = 1 GPU each; carved out of the actor share (invisible to the CLI).
N_TEACHER_GPUS=3
# Teacher context length (job <JOB_ID> fix). The teacher scores the FULL student trajectory
# (input_ids = prompt + multi-turn response). At context_length=16384 a long search/tau
# trajectory (retrieved wiki passages + up to 8192 response) overflowed — sglang returned
# HTTP 400 "input (19931 tokens) is longer than the model's context length (16384)" and
# reward_func's raise_for_status killed the whole job at step 4. The student engines run at
# 32768, so size the teachers to match: prompt(~12-14k) + response(8192) fits comfortably.
# (opd_multi_teacher.reward_func is ALSO hardened to not crash if a teacher score still 4xx's.)
TEACHER_CTX_LEN=${TEACHER_CTX_LEN:-32768}
ACTOR_ROLLOUT_GPUS=$(( ROLLOUT_NUM_GPUS - USER_SIM_NUM_GPUS - N_TEACHER_GPUS ))
if [ "${ACTOR_ROLLOUT_GPUS}" -le 0 ]; then
  echo "FATAL: no actor rollout GPUs left: ROLLOUT_NUM_GPUS(${ROLLOUT_NUM_GPUS}) - USER_SIM_NUM_GPUS" \
       "(${USER_SIM_NUM_GPUS}) - ${N_TEACHER_GPUS} teachers = ${ACTOR_ROLLOUT_GPUS}. Increase --rollout-nodes." >&2
  exit 1
fi
# NODE-ALIGNMENT GUARD (job <JOB_ID> fix). The generated YAML orders models so groups never
# straddle a node boundary: [actor + 3 teachers] must exactly fill one node, and user_sim
# must be a whole number of nodes. If not, some model group would start mid-node and slime
# would hand its cross-node engines the wrong node's dist_init_addr -> 600s TCPStore timeout
# (see the heredoc comment). Assert both here so a bad node/GPU split fails loud, not hangs.
if [ $(( ACTOR_ROLLOUT_GPUS + N_TEACHER_GPUS )) -ne "${NUM_GPUS}" ]; then
  echo "FATAL: actor(${ACTOR_ROLLOUT_GPUS}) + ${N_TEACHER_GPUS} teachers = $(( ACTOR_ROLLOUT_GPUS + N_TEACHER_GPUS )) must equal one node's GPUs (${NUM_GPUS}) so the actor+teachers group fills node 0 exactly. Adjust --rollout-nodes/--user-sim-nodes." >&2
  exit 1
fi
if [ $(( USER_SIM_NUM_GPUS % NUM_GPUS )) -ne 0 ]; then
  echo "FATAL: USER_SIM_NUM_GPUS(${USER_SIM_NUM_GPUS}) must be a whole multiple of a node's GPUs (${NUM_GPUS}) so user_sim occupies whole nodes (no cross-node straddle)." >&2
  exit 1
fi
USER_SIM_TP=${USER_SIM_TP:-1}
SGLANG_CFG="${SCRIPT_DIR}/opd_multi_sglang.generated.yaml"
cat > "${SGLANG_CFG}" <<YAMLEOF
# AUTO-GENERATED by run-qwen3-8B-opd-multi.sh — do not edit by hand.
# rollout split: actor=${ACTOR_ROLLOUT_GPUS} + 3 teachers + user_sim=${USER_SIM_NUM_GPUS} = ${ROLLOUT_NUM_GPUS} (= --rollout-num-gpus)
#
# MODEL ORDER IS LOAD-BEARING (job <JOB_ID> fix). slime packs rollout GPUs in this list
# order via a cumulative gpu_offset (rollout.py:1103-1154), and its per-engine
# dist_init_addr/TCPStore is grouped by node = local_rank // engines_per_node
# (rollout.py:936) — which assumes every model group STARTS on a node boundary. With
# 2 rollout nodes (8 GPU each) and the old order [actor=5, user_sim=8, teachers=3], the
# 8-GPU user_sim started at GPU offset 5 (mid-node) and straddled BOTH nodes; slime then
# handed all its engines the FIRST node's IP, so the user_sim engines physically on node 1
# tried to TCPStore-rendezvous cross-subnet to node 0 and timed out after 600s
# (DistNetworkError: client socket timed out connecting to <node0>:15296). FIX: order the
# 8 single-GPU models first (actor 5 + 3 teachers = 8 = exactly node 0), then user_sim (8
# = exactly node 1). Now every group is node-aligned; no model spans the node boundary.
# (ACTOR_ROLLOUT_GPUS + 3 == num_gpus_per_node is asserted below before this heredoc.)
sglang:
  - name: actor
    update_weights: true
    num_gpus_per_engine: 1
    server_groups:
      - worker_type: regular
        num_gpus: ${ACTOR_ROLLOUT_GPUS}
  - name: teacher_math
    model_path: ${MODEL_ROOT}/ACLArena/Qwen3-8B-Base-Math
    update_weights: false
    num_gpus_per_engine: 1
    server_groups:
      - worker_type: regular
        num_gpus: 1
        overrides:
          served_model_name: teacher_math
          context_length: ${TEACHER_CTX_LEN}
          mem_fraction_static: 0.8
          trust_remote_code: true
  - name: teacher_search
    model_path: ${MODEL_ROOT}/ACLArena/Qwen3-8B-Base-SeaSFT-Search
    update_weights: false
    num_gpus_per_engine: 1
    server_groups:
      - worker_type: regular
        num_gpus: 1
        overrides:
          served_model_name: teacher_search
          context_length: ${TEACHER_CTX_LEN}
          mem_fraction_static: 0.8
          trust_remote_code: true
  - name: teacher_tau
    model_path: ${MODEL_ROOT}/ACLArena/Qwen3-8B-Base-TauSFT-Tau
    update_weights: false
    num_gpus_per_engine: 1
    server_groups:
      - worker_type: regular
        num_gpus: 1
        overrides:
          served_model_name: teacher_tau
          context_length: ${TEACHER_CTX_LEN}
          mem_fraction_static: 0.8
          trust_remote_code: true
  - name: user_sim
    model_path: ${MODEL_ROOT}/GLM/GLM-4.7-Flash
    update_weights: false
    num_gpus_per_engine: ${USER_SIM_TP}
    server_groups:
      - worker_type: regular
        num_gpus: ${USER_SIM_NUM_GPUS}
        overrides:
          served_model_name: ${TAU_USER_MODEL_ID}
          tool_call_parser: glm47
          reasoning_parser: glm45
          mem_fraction_static: 0.8
          trust_remote_code: true
YAMLEOF
echo "[sglang-config] generated ${SGLANG_CFG} (actor=${ACTOR_ROLLOUT_GPUS} + user_sim=${USER_SIM_NUM_GPUS} + 3 teachers = ${ROLLOUT_NUM_GPUS}):"
cat "${SGLANG_CFG}"

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   --sglang-config "${SGLANG_CFG}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ----------------------------------------------------------------------------
# NOTE (job <JOB_ID> ordering fix): the CPU-faiss retriever is NOT launched here.
# It loads a 64 GB flat index + 14 GB corpus (minutes), and previously the head
# blocked on that load BEFORE `ray start --head` — so the head's GCS :6379 didn't
# exist while worker nodes probed it (their join budget is only 600s) and they
# timed out. Now: (1) `ray start --head` first, (2) launch the retriever in the
# background so it loads DURING the 30-min GPU-registration wait (free time), then
# (3) gate on retriever readiness right before `ray job submit`. See below.
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Ray + launch
# ----------------------------------------------------------------------------
# multi_teacher (custom fns) + both agentic example dirs (search/tau native generates) on PYTHONPATH.
export PYTHONPATH="${SCRIPT_DIR}:${SLIME_DIR:-${ROOT_DIR}/slime}/examples/search-r1:${SLIME_DIR:-${ROOT_DIR}/slime}/examples/tau-bench:${SLIME_DIR:-${ROOT_DIR}/slime}:${MEGATRON_DIR:-${ROOT_DIR}/Megatron-LM}:${PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export RAY_memory_usage_threshold=0.99
RAY_TEMP_DIR=${RAY_TEMP_DIR:-${ROOT_DIR}/ray_temp}
rm -rf "$RAY_TEMP_DIR"; mkdir -p "$RAY_TEMP_DIR"

# Dev-box needs the system lib dir prepended; the Cluster image sets SKIP_SYS_LDPATH=1.
if [ -z "${SKIP_SYS_LDPATH:-}" ]; then
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
fi

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir "${RAY_TEMP_DIR}"

# ----------------------------------------------------------------------------
# Launch the CPU-faiss search retriever on the MAIN node (background), AFTER the
# Ray head is up so worker nodes can join immediately (they don't have to wait for
# the 64 GB index load). It loads in parallel with the GPU-registration wait below;
# readiness is gated right before `ray job submit`. Rollout-node search actors reach
# it via SEARCH_R1_SEARCH_URL=http://${MASTER_ADDR}:${RETRIEVE_PORT}.
# ----------------------------------------------------------------------------
if [ "${OPD_ENABLE_RETRIEVER:-1}" = "1" ]; then
  # The Cluster slime image ships torch/numpy/transformers/datasets/tqdm but NOT
  # faiss/fastapi/uvicorn/pydantic — the retriever imports these (retrieval_server.py:23-31)
  # and a missing faiss crashed job <JOB_ID> (ModuleNotFoundError: faiss). Install the
  # missing deps on the MAIN node (only it runs the retriever). CPU faiss -> faiss-cpu.
  # Mirrors the bootstrap's runtime `pip install awscli`. Toggle: OPD_RETRIEVER_PIP=0.
  if [ "${OPD_RETRIEVER_PIP:-1}" = "1" ]; then
    echo "[retriever] ensuring python deps (faiss-cpu fastapi uvicorn pydantic) are present"
    python3 - <<'PYCHK' || pip install -q faiss-cpu fastapi uvicorn pydantic 2>&1 | tail -5
import importlib.util, sys
missing = [m for m in ("faiss", "fastapi", "uvicorn", "pydantic") if importlib.util.find_spec(m) is None]
sys.exit(1 if missing else 0)
PYCHK
    python3 -c "import faiss, fastapi, uvicorn, pydantic; print('[retriever] deps OK: faiss', faiss.__version__)" \
      || { echo "FATAL: retriever deps still missing after pip install; cannot start retriever." >&2; exit 1; }
  fi
  # THREAD CAP (job <JOB_ID> fix): on this 96-vCPU node faiss-cpu SIGSEGV'd during index load
  # with a flood of "BLAS : Program is Terminated. Because you tried to allocate too many
  # memory regions." OpenBLAS has a compile-time cap on concurrent memory-region buffers
  # (~NUM_THREADS*2); unbounded on 96 cores it overflows and aborts, crashing faiss. Cap the
  # BLAS/OMP thread count for the retriever process (it also shares this node with the 8-GPU
  # student). Flat-index faiss + a tiny e5 BERT encoder don't need 96 threads.
  # Tune via RETRIEVER_NUM_THREADS (default 32).
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
    --topk "${SEARCH_R1_TOPK:-3}" \
    --retriever_name e5 \
    --retriever_model "${E5_MODEL}" \
    > "${ROOT_DIR}/retriever.log" 2>&1 &
  RETRIEVER_PID=$!
  echo "[retriever] pid=${RETRIEVER_PID}; loading index in background (readiness gated before submit)."
fi

# Multi-node: wait until ALL GPUs (training + rollout) register with Ray before submitting.
EXPECTED_GPUS=$(( ACTOR_NUM_NODES * NUM_GPUS + ROLLOUT_NUM_GPUS ))
if [ "$EXPECTED_GPUS" -gt "$NUM_GPUS" ]; then
    echo "Waiting for ${EXPECTED_GPUS} GPUs (train $(( ACTOR_NUM_NODES * NUM_GPUS )) + rollout ${ROLLOUT_NUM_GPUS}) to register (head=${MASTER_ADDR}, up to 30 min for child-node boot-skew)..."
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
        echo "FATAL: only ${GOT}/${EXPECTED_GPUS} GPUs registered after $(( $(date +%s) - WAIT_START ))s (head=${MASTER_ADDR}, alive=[${ALIVE_IPS}]). Likely a cross-subnet control-plane block or the child node never booted." >&2
        ray stop --force 2>/dev/null || true
        exit 1
    fi
    echo "All ${EXPECTED_GPUS} GPUs registered in $(( $(date +%s) - WAIT_START ))s; proceeding to submit."
fi

RESOURCE_ARGS=(
   --actor-num-nodes ${ACTOR_NUM_NODES}
   --actor-num-gpus-per-node ${NUM_GPUS}
   --num-gpus-per-node ${NUM_GPUS}
   --rollout-num-gpus ${ROLLOUT_NUM_GPUS}
)

# Worker-node rollout actors do NOT inherit this shell env — they run the tau user-sim + search
# tool, so tau/search env + PYTHONPATH must be injected into the Ray job runtime-env.
MULTINODE_ENV=""
if [ "${ACTOR_NUM_NODES:-1}" -gt 1 ]; then
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
    \"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN\": \"1\",
    \"OPD_TEACHER_TIMEOUT\": \"${OPD_TEACHER_TIMEOUT}\",
    \"SEARCH_R1_SEARCH_URL\": \"${SEARCH_R1_SEARCH_URL}\",
    \"SEARCH_R1_STRIP_THINK\": \"${SEARCH_R1_STRIP_THINK}\",
    \"SEARCH_R1_CONCURRENCY\": \"${SEARCH_R1_CONCURRENCY}\",
    \"TAU_USER_STRATEGY\": \"${TAU_USER_STRATEGY}\",
    \"TAU_USER_MODEL_ID\": \"${TAU_USER_MODEL_ID}\",
    \"TAU_USER_SIM_MODEL\": \"${TAU_USER_SIM_MODEL}\",
    \"TAU_ENV\": \"${TAU_ENV}\",
    \"TAU_TASK_SPLIT\": \"${TAU_TASK_SPLIT}\",
    \"TAU_ENABLE_THINKING\": \"${TAU_ENABLE_THINKING}\",
    \"TAU_STRIP_HISTORICAL_THINK\": \"${TAU_STRIP_HISTORICAL_THINK}\",
    \"TAU_USER_THINK_KWARG\": \"${TAU_USER_THINK_KWARG}\",
    \"TAU_USER_MAX_TOKENS\": \"${TAU_USER_MAX_TOKENS}\"${MULTINODE_ENV}
  }
}"

# ----------------------------------------------------------------------------
# Gate on retriever readiness (it has been loading in the background since ray start,
# overlapped with the GPU-registration wait above). Only now — right before submit —
# do we require it to answer /retrieve. This does NOT block worker Ray joins.
# ----------------------------------------------------------------------------
if [ "${OPD_ENABLE_RETRIEVER:-1}" = "1" ]; then
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

# NOTE: sync train.py (not train_async.py) — it supports disaggregated (--rollout-num-gpus) +
# --sglang-config and keeps the OPD reward/loss semantics unchanged. If the Cluster GPU-idle
# watchdog kills the alternating train/rollout pools, switch to train_async.py (+ --update-weights-interval).
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --custom-generate-function-path generate_mixed_opd.generate \
   ${RESOURCE_ARGS[@]} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${RM_ARGS[@]}

#### cleanup
[ -n "${RETRIEVER_PID:-}" ] && kill "${RETRIEVER_PID}" 2>/dev/null || true
ray stop --force
pkill -9 ray || true
pkill -9 python || true
