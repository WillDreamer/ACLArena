#!/bin/bash
#
# RESUME script for run_8b_eopd.sh (Entropy-Aware OPD, TAU-BENCH domain).
# The original run (wandb run eopd_tau2if_bs_24 == id umgk8xi9) crashed at rollout 69 when the
# tau-bench user simulator (was :8098) went down: ~97% of rollouts aborted at env.reset, aborted
# samples skip the teacher RM (reward stays None), and post_process_rewards_eopd choked on the
# None reward. The user simulator has since moved to :8000, and run_8b_eopd.sh now drops groups
# with aborted samples (opd_filters.drop_group_with_aborted), so a repeat is non-fatal.
#
# The last on-disk checkpoint is iter_0000060 (save-interval 30), so this resumes from rollout 60
# and runs the remaining 60..299.
#
# What differs from run_8b_eopd.sh (everything else is kept identical on purpose):
#   1) user-sim endpoint :8098 -> :8000 (the server was moved).
#   2) CKPT_ARGS: --load now points at the SAVE dir (the EOPD checkpoints), and
#      --finetune / --no-load-optim / --no-load-rng are REMOVED. With those flags megatron would
#      reset iteration->0 and skip optimizer/rng restore (a *fresh* run that overwrites iter_60).
#      Removing them => megatron reads latest_checkpointed_iteration.txt (=60), restores
#      weights + Adam moments + RNG, and slime auto-derives start_rollout_id=60. train.py then
#      loops `for rollout_id in range(60, 300)`. The data iterator is restored from
#      rollout/global_dataset_state_dict_59.pt so data ordering continues without repetition.
#      (iter_0000060 holds full optimizer state -- 12 distcp shards -- so this resumes cleanly.)
#   3) W&B: WANDB_RUN_ID + WANDB_RESUME env vars reattach to the SAME run (umgk8xi9). New points
#      append at step >= 60; rollouts 60..68 get logged a second time (cheap, cosmetic).
#   The forward-KL warmup is keyed on rollout_id, so at resume (60 >> 40) the coef is already at
#   the full target 1.0 -- correct, no re-warmup.
#
# usage: cd /data/user/slime && bash examples/on_policy_distillation/tau2if/run_8b_eopd_resume.sh

set -ex

# Clean up any orphaned engines/cluster from a previous (crashed) run before starting.
# Safe: this script's own cmdline does not contain these patterns, and pkill excludes itself.
pkill -9 -f sglang || true
ray stop --force 2>/dev/null || true
pkill -9 -f 'ray::' || true
sleep 3

ROOT_DIR=/data/user
MODEL_ROOT=/shared/user
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${ROOT_DIR}/slime"   # train.py / tools / examples paths resolve from the slime root

# ----------------------------------------------------------------------------
# GPU allocation
# ----------------------------------------------------------------------------
# GPU 0 is free: the user simulator is remote (131.179.168.120:8000), nothing launched locally.
TEACHER_GPU=1                   # <<<--- teacher SGLang server
TRAIN_GPU_LIST=(2 3 4 5 6 7)    # <<<--- training (Ray, colocate)
NUM_GPUS=${#TRAIN_GPU_LIST[@]}
TRAIN_CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${TRAIN_GPU_LIST[*]}")
echo "teacher GPU=${TEACHER_GPU}; training GPUs=${TRAIN_CUDA_VISIBLE_DEVICES} (${NUM_GPUS}); user-sim=REMOTE"

ROLLOUT_BATCH_SIZE=24
WANDB_GROUP="eopd_tau2if_bs_${ROLLOUT_BATCH_SIZE}"
# Per-rollout debug dumps (tokens / loss_mask / reward / response_length / rollout_log_probs).
ROLLOUT_DEBUG_DIR="${MODEL_ROOT}/MultiStageRL/${WANDB_GROUP}"

# RESUME target: the W&B run id of the crashed run (eopd_tau2if_bs_24 -> umgk8xi9).
WANDB_RESUME_ID="umgk8xi9"

# Paths
STUDENT=${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau-IF
TEACHER_TD=${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau        # megatron torch_dist ckpt
TEACHER_HF=${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau/hf_iter_0000300     # HF copy for SGLang (auto-built)
BASE_HF=${MODEL_ROOT}/Qwen3-8B-Base/                       # tokenizer + config
SAVE_DIR=${MODEL_ROOT}/MultiStageRL/OPD/Qwen3-8B-Base-TauSFT-Tau-IF_EOPD_Tau     # resume from + save into this dir

# examples/tau-bench -> generate_with_tau + trainable_agents + tau_bench deps.
# SCRIPT_DIR (this tau2if dir) -> generate_with_tau_opd + opd_filters.
export PYTHONPATH="${ROOT_DIR}/slime/examples/tau-bench:${SCRIPT_DIR}:${ROOT_DIR}/slime:${ROOT_DIR}/Megatron-LM:${PYTHONPATH}"

# Remote tau-bench user simulator (OpenAI-compatible). Read by tau_bench during rollout.
export OPENAI_API_BASE=http://131.179.168.120:8000/v1
export OPENAI_API_KEY=dummy

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
# 1) Remote user simulator (tau-bench user-sim LLM @ 131.179.168.120:8000).
# ----------------------------------------------------------------------------
USERSIM_URL=http://131.179.168.120:8000/v1/models
if curl -sf -m 10 "${USERSIM_URL}" >/dev/null 2>&1; then
    echo "Remote user simulator reachable at ${USERSIM_URL}."
else
    echo "ERROR: remote user simulator NOT reachable at ${USERSIM_URL}." >&2
    echo "       Make sure the tau-bench user-sim server on 131.179.168.120:8000 is up and network-reachable." >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# 2) Teacher SGLang server on GPU 1 (HF, TP=1, scoring-only).
#    EOPD needs per-token top-k logprobs (top_logprobs_num); standard sglang /generate does.
# ----------------------------------------------------------------------------
TEACHER_IP="127.0.0.1"
TEACHER_PORT=13141
TEACHER_LOG="${ROOT_DIR}/teacher_sglang_tau.log"

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

# RESUME: load weights + optimizer + rng + iteration from the last EOPD checkpoint (iter_0000060),
# and keep saving into the same dir. --finetune / --no-load-optim / --no-load-rng are REMOVED
# (vs run_8b_eopd.sh) so megatron resumes instead of starting fresh. The saved optimizer/scheduler
# are from THIS run's own schedule, so there is no OptimizerParamScheduler mismatch.
CKPT_ARGS=(
   --hf-checkpoint ${BASE_HF}                 # tokenizer + config
   --ref-load      ${STUDENT}                 # (KL-loss-coef=0, so ref is inert; mirror student)
   --load          ${SAVE_DIR}                # RESUME from iter_0000060 (weights+optim+rng+iteration)
   --save          ${SAVE_DIR}                # continue saving into the same dir
   --save-interval 30
)

# Tau-bench prompt data (task indices; the wiki + conversation are built in-env at rollout time).
ROLLOUT_ARGS=(
   --prompt-data ${ROOT_DIR}/tau-bench/retail_train_tasks.jsonl
   --input-key index
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}            # 24 * 8 = 192 trajectories/rollout (divisible by DP=3)
   --n-samples-per-prompt 8           # reward=0 -> grouping neutral; lower to 4 to cut rollout cost
   --rollout-max-response-len 2048    # tau-bench native horizon (per run_qwen3_8B.sh)
   --rollout-temperature 1            # EOPD forward-KL assumes natural (temp=1) student logits
   --global-batch-size 192            # 192 trajectories -> 1 grad step/rollout; 192/DP(3) = 64
   --balance-data
   # Drop prompt-groups with an ABORTED sample so they never hit the teacher post-process with a
   # None reward (the crash that killed this run @ rollout 69). Status-based, so it composes with
   # pure OPD's constant-0 reward; slime regenerates dropped groups. See opd_filters.py.
   --dynamic-sampling-filter-path opd_filters.drop_group_with_aborted

   # Save per-rollout trajectories for offline analysis (length/turns/trunc, raw tokens).
   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"

   # ---- eval (tau-bench task success) ----
   --eval-interval 25
   --eval-config "${SCRIPT_DIR}/eval_tau.yaml"
)

# EOPD reward: teacher scalar logprob (reverse KL) + per-token top-k dist & entropy (forward KL).
RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func_eopd
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards_eopd
   --rm-url http://${TEACHER_IP}:${TEACHER_PORT}/generate
)

# Tau-bench multi-turn rollout (TRAIN wrapper: clears task reward so the teacher RM runs).
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_tau_opd.generate
)

# 6 training GPUs: TP=2, PP=1, CP=1 -> DP=3.  CP MUST be 1 for the EOPD forward-KL term.
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
   --use-opd
   --opd-type sglang
   --opd-kl-coef 0.3
   --use-eopd-forward-kl
   --eopd-fkl-coef 1.0
   --eopd-entropy-threshold 0.4
   --eopd-topk 16
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
# 4) Launch Ray on the 6 training GPUs only (user-sim/teacher are outside Ray).
# ----------------------------------------------------------------------------
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export RAY_memory_usage_threshold=0.99
export RAY_TMPDIR="${ROOT_DIR}/ray_out"
rm -rf "$RAY_TMPDIR"; mkdir -p "$RAY_TMPDIR"

# ---- W&B RESUME: reattach to the existing run instead of creating a new one ----
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
# 5) Cleanup. User-sim is REMOTE; only target this run's processes.
# ----------------------------------------------------------------------------
pkill -9 -f sglang || true       # teacher server + student rollout engines
sleep 3
ray stop --force || true
pkill -9 -f 'ray::' || true
pkill -9 -f train.py || true
sleep 3
