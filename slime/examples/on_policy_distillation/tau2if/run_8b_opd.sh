#!/bin/bash
#
# Single-teacher on-policy distillation (OPD), TAU-BENCH domain, all-local on one 8-GPU box.
#
#   student : MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau-IF  (megatron torch_dist, --load)
#   teacher : MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau     (tau specialist) -> local SGLang server
#
# Pure distillation: task reward = 0; the only learning signal is the OPD KL penalty
# (advantage -= opd_kl_coef * (student_log_prob - teacher_log_prob), per token).
# The student (the IF-RL'd model) rolls out with the tau-bench multi-turn agent (talks to the
# remote user simulator + retail tools), and the teacher scores token-level logprobs over the
# student's own trajectory. Goal: distill the tau specialist's behavior back into the IF model.
#
# Tau-domain wiring (vs the search sea2if scripts):
#   * rollout fn = generate_with_tau_opd.generate (TRAIN): runs the tau-bench trajectory but
#     CLEARS the task reward so the OPD teacher RM (async_rm) is invoked. The stock
#     examples/tau-bench/generate_with_tau.generate always sets sample.reward, which would
#     suppress the teacher RM -- hence the thin wrapper in generate_with_tau_opd.py.
#   * eval fn   = generate_with_tau_opd.generate_eval (via eval_tau.yaml): keeps the real
#     tau-bench task reward so eval reports true task success, not teacher logprob.
#   * user simulator is REMOTE (OPENAI_API_BASE below); retail tools run in-process via tau_bench.
#
# GPU layout on this single node (8x H200):
#   GPU 0      -> free (user simulator is REMOTE @ 131.179.168.120:8000, not launched here)
#   GPU 1      -> teacher SGLang server       (:13141)    [outside Ray]
#   GPU 2..7   -> Ray: colocated actor + student rollout  (6 GPUs)
#
# usage: cd /data/user/slime && bash examples/on_policy_distillation/tau2if/run_8b_opd.sh

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
# GPU 0 is free: the user simulator is remote (131.179.168.120:8000), nothing launched locally.
TEACHER_GPU=1                   # <<<--- teacher SGLang server
TRAIN_GPU_LIST=(2 3 4 5 6 7)    # <<<--- training (Ray, colocate)
NUM_GPUS=${#TRAIN_GPU_LIST[@]}
TRAIN_CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${TRAIN_GPU_LIST[*]}")
echo "teacher GPU=${TEACHER_GPU}; training GPUs=${TRAIN_CUDA_VISIBLE_DEVICES} (${NUM_GPUS}); user-sim=REMOTE"

ROLLOUT_BATCH_SIZE=24
WANDB_GROUP="opd_tau2if_bs_${ROLLOUT_BATCH_SIZE}"
# Per-rollout debug dumps (tokens / loss_mask / reward / response_length / rollout_log_probs).
# Saved as rollout_<id>.pt (train) and rollout_eval_<id>.pt (eval); parent dir auto-created.
ROLLOUT_DEBUG_DIR="${MODEL_ROOT}/MultiStageRL/${WANDB_GROUP}"

# Paths
STUDENT=${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau-IF
TEACHER_TD=${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau        # megatron torch_dist ckpt
TEACHER_HF=${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau/hf_iter_0000300     # HF copy for SGLang (auto-built)
BASE_HF=${MODEL_ROOT}/Qwen3-8B-Base/                       # tokenizer + config

# examples/tau-bench -> generate_with_tau + trainable_agents + tau_bench deps.
# SCRIPT_DIR (this tau2if dir) -> generate_with_tau_opd (TRAIN/eval rollout wrappers).
export PYTHONPATH="${ROOT_DIR}/slime/examples/tau-bench:${SCRIPT_DIR}:${ROOT_DIR}/slime:${ROOT_DIR}/Megatron-LM:${PYTHONPATH}"

# Remote tau-bench user simulator (OpenAI-compatible). Read by tau_bench during rollout.
export OPENAI_API_BASE=http://131.179.168.120:8000/v1
export OPENAI_API_KEY=dummy

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
# 1) Remote user simulator (tau-bench user-sim LLM @ 131.179.168.120:8000). The tau-bench
#    env drives the conversation against this OpenAI-compatible endpoint. We do NOT launch
#    anything locally -- just verify it's reachable so we fail fast instead of after teacher
#    + train startup.
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

CKPT_ARGS=(
   --hf-checkpoint ${BASE_HF}                 # tokenizer + config
   --ref-load      ${STUDENT}                 # (KL-loss-coef=0, so ref is inert; mirror student)
   --load          ${STUDENT}                 # STUDENT (init point, NOT a resume)
   # STUDENT is a "release" megatron ckpt (latest_checkpointed_iteration.txt == "release"; the
   # release/ dir holds weights ONLY, no optimizer/rng/LR-scheduler state). This is a *fresh* OPD
   # run with its own schedule, so load weights only and reset iteration -> 0 so all --num-rollout
   # rollouts actually run: --no-load-optim/--no-load-rng skip the (absent) optimizer+rng, and
   # --finetune treats it as a fine-tune start (iteration 0) instead of a resume.
   --no-load-optim
   --no-load-rng
   --finetune
   --save          ${MODEL_ROOT}/MultiStageRL/OPD/Qwen3-8B-Base-TauSFT-Tau-IF_OPD_Tau/
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
   --rollout-temperature 1
   --global-batch-size 192            # 192 trajectories -> 1 grad step/rollout; 192/DP(3) = 64
   --balance-data
   # Drop prompt-groups with an ABORTED sample (e.g. a user-sim blip) so they never reach the OPD
   # teacher post-process with a None reward -- that exact crash killed eopd_tau2if @ rollout 69
   # when the user-sim went down. It is STATUS-based (not reward-std), so it composes with pure
   # OPD's constant-0 task reward (a reward-std filter would delete every group). slime regenerates
   # dropped groups via over-sampling, so the batch still fills to rollout_batch_size.
   --dynamic-sampling-filter-path opd_filters.drop_group_with_aborted
   # (no --apply-chat-template / --label-key: tau prompts are task indices, templated in-env.)

   # Save per-rollout trajectories for offline analysis (length/turns/trunc, raw tokens).
   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"

   # ---- eval (tau-bench task success) ----
   # Eval runs the tau-bench rollout and keeps the real task reward via generate_eval
   # (bypasses the OPD teacher reward). Because this run's --custom-rm-path is the teacher
   # reward and async_rm prefers it, the per-dataset custom_generate_function_path in
   # eval_tau.yaml is what makes eval report real task success without contacting the teacher.
   --eval-interval 25
   --eval-config "${SCRIPT_DIR}/eval_tau.yaml"
)

# OPD reward: per-sample teacher token-level logprobs from the SGLang server; task reward = 0.
RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
   --rm-url http://${TEACHER_IP}:${TEACHER_PORT}/generate
)

# Tau-bench multi-turn rollout (TRAIN wrapper: clears task reward so the teacher RM runs).
# Pure OPD: NO --use-tis and NO dynamic-sampling filter, matching the sea2if OPD precedent.
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_tau_opd.generate
)

# 6 training GPUs: TP=2, PP=1, CP=1 -> DP=3.  (TP must divide 8 KV groups -> {1,2,4}; 4 does not
# divide 6, so TP=2.)  CP is kept at 1 for parity with the EOPD/GKD scripts (whose forward-KL
# term requires CP=1); recompute-full keeps activation memory low for the long tau wiki prompts.
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
   --max-tokens-per-gpu 16384         # must be >= the longest single sequence (wiki ~6k + resp 2048)
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef 0.5            # pull the RL'd student back toward the tau teacher
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
# 5) Cleanup. The user simulator is REMOTE (131.179.168.120) so nothing to stop here;
#    we still avoid a blanket `pkill python` and only target this run's processes.
# ----------------------------------------------------------------------------
pkill -9 -f sglang || true       # teacher server + student rollout engines
sleep 3
ray stop --force || true
pkill -9 -f 'ray::' || true
pkill -9 -f train.py || true
sleep 3
