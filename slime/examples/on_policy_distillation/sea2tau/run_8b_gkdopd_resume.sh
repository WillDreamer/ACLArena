#!/bin/bash
#
# RESUME script for run_8b_gkdopd.sh (GKD-style forward-KL-only OPD, SEARCH domain).
# The original run (wandb run gkdopd_search2tau_bs_24 == id bd64yexu) died at rollout ~135 with
# a transient "Cuda failure 999 'unknown error'". The last on-disk checkpoint is iter_0000120
# (save-interval 30), so this resumes from rollout 120 and runs the remaining 120..299.
#
# What differs from run_8b_gkdopd.sh (everything else is byte-identical on purpose):
#   1) CKPT_ARGS: --load now points at the SAVE dir (the GKD checkpoints), and
#      --finetune / --no-load-optim / --no-load-rng are REMOVED. With those flags megatron would
#      reset iteration->0 and skip optimizer/rng restore (i.e. start a *fresh* run that overwrites
#      iter_0000120..). Removing them => megatron reads latest_checkpointed_iteration.txt (=120),
#      restores weights + Adam moments + RNG, and slime auto-derives start_rollout_id=120
#      (slime/ray/placement_group.py:168 from the loaded iteration; arguments.py:1706-1718 does NOT
#      force a fresh start because --load has a latest_checkpointed_iteration.txt). train.py then
#      loops `for rollout_id in range(120, 300)`. The data iterator is restored from
#      rollout/global_dataset_state_dict_119.pt (rollout_global_dataset on by default), so data
#      ordering continues without repetition.
#   2) W&B: WANDB_RUN_ID + WANDB_RESUME env vars make init_wandb_primary() reattach to the SAME run
#      (bd64yexu) instead of creating a new one (it calls wandb.init() with no explicit id/resume,
#      so wandb honors these env vars). New points append at train/eval/rollout step >= 120; rollouts
#      120..135 get logged a second time (cheap, cosmetic — checkpoint at 120, crash at 135).
#
# usage: cd /data/user/slime && bash examples/on_policy_distillation/sea2tau/run_8b_gkdopd_resume.sh

set -ex

# pkill -9 sglang
# sleep 3
# ray stop --force
# pkill -9 ray
# pkill -9 python
# sleep 3

ROOT_DIR=/data/user
MODEL_ROOT=/shared/user
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${ROOT_DIR}/slime"   # train.py / tools / examples paths resolve from the slime root

# ----------------------------------------------------------------------------
# GPU allocation
# ----------------------------------------------------------------------------
# GPU 0 is free: the retriever is remote (131.179.168.117:8000), nothing launched locally.
TEACHER_GPU=1                   # <<<--- teacher SGLang server
TRAIN_GPU_LIST=(2 3 4 5 6 7)    # <<<--- training (Ray, colocate)
NUM_GPUS=${#TRAIN_GPU_LIST[@]}
TRAIN_CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${TRAIN_GPU_LIST[*]}")
echo "teacher GPU=${TEACHER_GPU}; training GPUs=${TRAIN_CUDA_VISIBLE_DEVICES} (${NUM_GPUS}); retriever=REMOTE"

ROLLOUT_BATCH_SIZE=24
WANDB_GROUP="gkdopd_search2tau_bs_${ROLLOUT_BATCH_SIZE}"
# Per-rollout debug dumps (tokens / loss_mask / reward / response_length / rollout_log_probs).
# Saved as rollout_<id>.pt (train) and rollout_eval_<id>.pt (eval); parent dir auto-created.
ROLLOUT_DEBUG_DIR="${MODEL_ROOT}/MultiStageRL/${WANDB_GROUP}"

# RESUME target: the W&B run id of the crashed run (gkdopd_search2tau_bs_24 -> bd64yexu).
WANDB_RESUME_ID="bd64yexu"

# Paths
STUDENT=${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau
TEACHER_TD=${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search        # megatron torch_dist ckpt
TEACHER_HF=${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search/hf_iter_0000420     # HF copy for SGLang (auto-built)
BASE_HF=${MODEL_ROOT}/Qwen3-8B-Base/                       # tokenizer + config
SAVE_DIR=${MODEL_ROOT}/MultiStageRL/OPD/Qwen3-8B-Base-TauSFT-Tau_GKD_Search    # resume from + save into this dir

export PYTHONPATH="${ROOT_DIR}/slime/examples/search-r1:${ROOT_DIR}/slime:${ROOT_DIR}/Megatron-LM:${PYTHONPATH}"

# Sanity: the resume checkpoint must exist (else slime would silently fall back to a fresh start).
if [ ! -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: no checkpoint to resume at ${SAVE_DIR} (missing latest_checkpointed_iteration.txt)." >&2
    exit 1
fi
echo "Resuming from ${SAVE_DIR} at iteration $(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt")."

# ----------------------------------------------------------------------------
# 0) Bootstrap: teacher torch_dist -> HF (SGLang serves HF). Done once (skipped if HF exists).
# ----------------------------------------------------------------------------
if [ ! -d "${TEACHER_HF}" ]; then
    ITER=$(cat "${TEACHER_TD}/latest_checkpointed_iteration.txt")
    ITER_DIR=$(printf "${TEACHER_TD}/iter_%07d" "${ITER}")
    echo "Converting teacher torch_dist -> HF (iter ${ITER}) ..."
    CUDA_VISIBLE_DEVICES="" python3 tools/convert_torch_dist_to_hf.py \
        --input-dir "${ITER_DIR}" \
        --output-dir "${TEACHER_HF}" \
        --origin-hf-dir "${BASE_HF}" \
        --vocab-size 151936
fi

# ----------------------------------------------------------------------------
# 1) Remote retriever (Search-R1 FAISS server @ 131.179.168.117:8000).
# ----------------------------------------------------------------------------
RETRIEVE_URL=http://131.179.168.117:8000/retrieve
retriever_up() {
    curl -sf -m 10 -X POST "${RETRIEVE_URL}" -H 'Content-Type: application/json' \
        -d '{"queries":["health check"],"topk":1,"return_scores":false}' >/dev/null 2>&1
}
if retriever_up; then
    echo "Remote retriever reachable at ${RETRIEVE_URL}."
else
    echo "ERROR: remote retriever NOT reachable at ${RETRIEVE_URL}." >&2
    echo "       Make sure the Search-R1 FAISS server on 131.179.168.117 is up and network-reachable." >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# 2) Teacher SGLang server on GPU 1 (HF, TP=1, scoring-only).
#    forward-KL needs per-token top-k logprobs, so the teacher must support
#    top_logprobs_num (standard sglang /generate does).
# ----------------------------------------------------------------------------
TEACHER_IP="127.0.0.1"
TEACHER_PORT=13142
TEACHER_LOG="${ROOT_DIR}/teacher_sglang_gkd.log"

CUDA_VISIBLE_DEVICES=${TEACHER_GPU} python3 -m sglang.launch_server \
    --model-path "${TEACHER_HF}" \
    --host 0.0.0.0 \
    --port ${TEACHER_PORT} \
    --tp 1 \
    --context-length 32768 \
    --chunked-prefill-size 4096 \
    --mem-fraction-static 0.8 \
    > "${TEACHER_LOG}" 2>&1 &

echo "Starting teacher SGLang server on GPU ${TEACHER_GPU}..."
until curl -sf http://${TEACHER_IP}:${TEACHER_PORT}/health_generate > /dev/null; do
    echo "Waiting for the teacher model server to start..."
    tail -n 10 "${TEACHER_LOG}"
    sleep 5
done
curl http://${TEACHER_IP}:${TEACHER_PORT}/get_model_info
echo "Teacher model server is up at ${TEACHER_IP}:${TEACHER_PORT}."
sleep 5

# ----------------------------------------------------------------------------
# 3) Training (student) config
# ----------------------------------------------------------------------------
export PYTHONBUFFERED=16
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
[ "$NVLINK_COUNT" -gt 0 ] && HAS_NVLINK=1 || HAS_NVLINK=0
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "${SCRIPT_DIR}/../../../scripts/models/qwen3-8B.sh"   # -> MODEL_ARGS

# RESUME: load weights + optimizer + rng + iteration from the last GKD checkpoint (iter_0000120),
# and keep saving into the same dir. --finetune / --no-load-optim / --no-load-rng are REMOVED
# (vs run_8b_gkdopd.sh) so megatron actually resumes instead of starting fresh. The saved
# optimizer/scheduler are from THIS run's own schedule, so there is no OptimizerParamScheduler
# mismatch (the reason the original used --finetune was to load a *different* run's STUDENT ckpt).
CKPT_ARGS=(
   --hf-checkpoint ${BASE_HF}                 # tokenizer + config
   --ref-load      ${STUDENT}                 # (KL-loss-coef=0, so ref is inert; mirror student)
   --load          ${SAVE_DIR}                # RESUME from iter_0000120 (weights+optim+rng+iteration)
   --save          ${SAVE_DIR}                # continue saving into the same dir
   --save-interval 5                          # match the (post-crash) original: checkpoint often
   --save-retain-interval 50                  # keep every 50th permanently, prune the rest
)

# Search prompt data (single domain) -> templated at load time.
ROLLOUT_ARGS=(
   --prompt-data ${ROOT_DIR}/Search-R1/data/nq_hotpotqa_train/train.parquet
   --input-key prompt
   --label-key reward_model
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}            # 24 * 8 = 192 trajectories/rollout (divisible by DP=3)
   --n-samples-per-prompt 8           # reward=0 -> grouping neutral; lower to 4 to cut rollout cost
   --rollout-max-response-len 4096    # curriculum CAP (final horizon); see flags below
   # --- entropy-/truncation-gated response-length curriculum (inherited stabilizer) ---
   # Start at a SHORT total-response horizon and only extend it when the model is stable at
   # the current horizon (truncation low AND teacher entropy still high). Forward-KL is
   # mode-covering and can also inflate length/entropy off-equilibrium, so keep the curriculum
   # as a general stabilizer. max_new_tokens is enforced as a TOTAL budget across the 5 search
   # turns, which is opt-in (default per-turn) so old scripts are unaffected -> enable it:
   --enforce-total-response-budget
   --use-length-curriculum
   --length-curriculum-init-len 2048      # start horizon (early healthy rollouts were ~1.1-1.6k)
   --length-curriculum-max-len 4096       # final horizon (== rollout-max-response-len)
   --length-curriculum-increment 512      # +512 tokens per advance
   --length-curriculum-interval 5         # at most one advance every 5 rollouts
   --length-curriculum-trunc-max 0.30     # advance only if last-rollout truncation <= 0.30
   --length-curriculum-entropy-min 0.6    # ...and mean teacher entropy >= 0.6 (safety floor)
   --rollout-temperature 1            # forward-KL assumes natural (temp=1) student logits
   --global-batch-size 192            # 192/192 = 1 grad step/rollout; 192/DP(3) = 64
   --balance-data

   # Save per-rollout trajectories for offline analysis (length/turns/trunc, raw tokens).
   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"

   # ---- eval (search task accuracy / EM) ----
   # Eval scores EM via generate_eval (bypasses async_rm / the teacher reward), so it works
   # unchanged here and never contacts the teacher. See eval_search.yaml.
   --eval-interval 25
   --eval-config "${SCRIPT_DIR}/eval_search.yaml"
)

# Reuse the EOPD reward: teacher scalar logprob (UNUSED here — no reverse-KL) + per-token top-k
# dist & entropy (the forward-KL signal). The teacher top-k flows regardless of --use-opd.
RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func_eopd
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards_eopd
   --rm-url http://${TEACHER_IP}:${TEACHER_PORT}/generate
)

# Search multi-turn rollout (student turns -> local sglang router; search tool -> retriever @ :8000).
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search_tools_qwen_sft_no_drift.generate
)

# 6 training GPUs: TP=2, PP=1, CP=1 -> DP=3.  CP MUST be 1 for the forward-KL term.
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --- PURE FORWARD-KL (GKD): reverse-KL OPD term REMOVED ---
   # --use-opd is intentionally OMITTED, so apply_opd_kl_to_advantages is never called.
   # Task reward = 0 and there is no reverse-KL penalty on advantages => the GRPO advantage is
   # identically 0 and pg_loss = 0. The ENTIRE training gradient is the forward-KL loss term below.
   #
   # forward-KL (mode-covering) on ALL response tokens. The EOPD entropy gate
   # (forward-KL only where teacher H > tau) makes sense only WITH reverse-KL covering the
   # low-entropy tokens; with reverse-KL removed, gating would leave low-entropy tokens
   # (answer / format / search-syntax) with zero gradient. So DISABLE the gate by setting the
   # threshold below any possible entropy (teacher entropy >= 0 always) => gate fires on every
   # token (eopd_highent_frac should log ~1.0).
   --use-eopd-forward-kl
   --eopd-fkl-coef 1.0
   --eopd-entropy-threshold -1.0    # <0 => gate always passes => forward-KL on ALL tokens
   --eopd-topk 16
   # forward-KL coef WARMUP: it is now the SOLE learning signal, so ramp it in gently
   # (0.3 -> 1.0 over the first 40 rollouts) to avoid early entropy/length inflation before the
   # student has learned terminating structure. NOTE: warmup is keyed on rollout_id, so on resume
   # (rollout_id >= 120 >> 40) the coef is already at the full target 1.0 — correct, no re-warmup.
   --eopd-fkl-warmup-rollouts 40
   --eopd-fkl-warmup-start-coef 0.3
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
   --wandb-project ACL-OPD
   --wandb-group ${WANDB_GROUP}
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
)

# Colocate: student rollout shares the 6 training GPUs. 2 GPUs/engine -> 3 engines, sglang TP=2.
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.6
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ----------------------------------------------------------------------------
# 4) Launch Ray on the 6 training GPUs only (retriever/teacher are outside Ray).
# ----------------------------------------------------------------------------
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false
export RAY_memory_usage_threshold=0.99
export RAY_TMPDIR="${ROOT_DIR}/ray_out"
rm -rf "$RAY_TMPDIR"; mkdir -p "$RAY_TMPDIR"

# Strip historical-turn <think> blocks: OFF -> trained sequence is byte-exact to what sglang
# prefilled (so teacher top-k logprobs line up token-for-token with the trained tokens).
export SEARCH_R1_STRIP_THINK=0

# ---- W&B RESUME: reattach to the existing run instead of creating a new one ----
# init_wandb_primary() (slime/utils/wandb_utils.py) calls wandb.init() WITHOUT an explicit id or
# resume, so wandb honors these env vars: WANDB_RUN_ID selects the run, WANDB_RESUME=allow appends
# to its history. Secondary (rollout/actor) inits then attach to the same id via args.wandb_run_id.
export WANDB_RUN_ID="${WANDB_RESUME_ID}"
export WANDB_RESUME=allow

export CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

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
   ${RM_ARGS[@]} \
   ${CUSTOM_ARGS[@]}

# ----------------------------------------------------------------------------
# 5) Cleanup. Retriever is REMOTE; only target this run's processes.
# ----------------------------------------------------------------------------
pkill -9 -f sglang || true       # teacher server + student rollout engines
sleep 3
ray stop --force || true
pkill -9 -f 'ray::' || true
pkill -9 -f train.py || true
sleep 3
