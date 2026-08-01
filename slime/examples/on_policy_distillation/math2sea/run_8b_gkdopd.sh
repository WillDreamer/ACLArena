#!/bin/bash
#
# GKD-style On-Policy Distillation (forward-KL ONLY), MATH domain, all-local on one 8-GPU box.
# Based on run_8b_eopd.sh, but with the reverse-KL OPD term REMOVED — pure forward-KL distillation
# on student-sampled (on-policy) trajectories. This is the forward-KL-only ablation of EOPD.
#
#   student : MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search   (megatron torch_dist, --load; search-trained,
#             has FORGOTTEN math)
#   teacher : MultiStageRL/Qwen3-8B-Base-Math                 (math specialist) -> local SGLang server
#
# Method (GKD, arXiv:2306.13649 — forward-KL on-policy distillation):
#   * NO reverse-KL term. --use-opd is intentionally omitted, so apply_opd_kl_to_advantages is
#     never called. The task reward is 0 (pure distillation) and there is no reverse-KL penalty on
#     advantages, so the GRPO advantage is identically 0 and the policy-gradient loss is 0 — the
#     ENTIRE training gradient comes from the forward-KL loss term.
#   * forward-KL (mode-covering) on ALL response tokens:
#       + eopd_fkl_coef * KL(p~_teacher_topk || p~_student_topk),
#     teacher top-k (k=16) renormalized, as a differentiable loss against the student logits.
#     The EOPD entropy gate is DISABLED here (--eopd-entropy-threshold below any possible entropy):
#     the gate only makes sense WITH reverse-KL covering the low-entropy tokens; with reverse-KL
#     removed, gating would leave low-entropy tokens (answer syntax / \boxed{} / digits) with zero
#     gradient. So forward-KL fires on every token (eopd_highent_frac should log ~1.0).
#     (--use-eopd-forward-kl / --eopd-fkl-coef / --eopd-entropy-threshold / --eopd-topk)
#
# The teacher returns BOTH the scalar logprob and the per-token top-k distribution + entropy via the
# EOPD reward fns (the scalar logprob is now unused — no reverse-KL — but the same RM is reused so
# the teacher top-k for forward-KL still flows):
#   --custom-rm-path                slime.rollout.on_policy_distillation.reward_func_eopd
#   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards_eopd
#
# Constraints (same as EOPD's forward-KL term): requires --context-parallel-size 1 (2D top-k tensors
# are not CP zigzag-sliced yet) and assumes --rollout-temperature 1 (student logits = natural dist).
#
# Math-domain wiring is identical to run_8b_opd.sh: the TRAIN rollout is the STOCK single-turn sglang
# rollout (NO --custom-generate-function-path); eval uses generate_with_math_opd.generate_eval to
# score the real AIME accuracy. NO user simulator, NO tools, NO dynamic-sampling filter.
#
# GPU layout (8x H200): GPU0 free; GPU1 teacher SGLang server; GPU2..7 Ray train+rollout.
#
# usage: cd /data/user/slime && bash examples/on_policy_distillation/math2sea/run_8b_gkdopd.sh

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
TEACHER_GPU=1                   # <<<--- teacher SGLang server
TRAIN_GPU_LIST=(2 3 4 5 6 7)    # <<<--- training (Ray, colocate)
NUM_GPUS=${#TRAIN_GPU_LIST[@]}
TRAIN_CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${TRAIN_GPU_LIST[*]}")
echo "teacher GPU=${TEACHER_GPU}; training GPUs=${TRAIN_CUDA_VISIBLE_DEVICES} (${NUM_GPUS})"

ROLLOUT_BATCH_SIZE=24
WANDB_GROUP="gkdopd_math2sea_bs_${ROLLOUT_BATCH_SIZE}"
# Per-rollout debug dumps (tokens / loss_mask / reward / response_length / rollout_log_probs).
# Saved as rollout_<id>.pt (train) and rollout_eval_<id>.pt (eval); parent dir auto-created.
ROLLOUT_DEBUG_DIR="${MODEL_ROOT}/MultiStageRL/${WANDB_GROUP}"

# Paths
STUDENT=${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search          # megatron torch_dist ckpt (iter 420)
TEACHER_TD=${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math                     # megatron torch_dist ckpt (iter 300)
TEACHER_HF=${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math/hf_iter_0000300     # HF copy for SGLang (auto-built; already present)
BASE_HF=${MODEL_ROOT}/Qwen3-8B-Base/                                         # tokenizer + config

# SCRIPT_DIR (this math2sea dir) -> generate_with_math_opd (eval rollout wrapper).
export PYTHONPATH="${SCRIPT_DIR}:${ROOT_DIR}/slime:${ROOT_DIR}/Megatron-LM:${PYTHONPATH}"

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
# 1) Teacher SGLang server on GPU 1 (HF, TP=1, scoring-only).
#    forward-KL needs per-token top-k logprobs, so the teacher must support
#    top_logprobs_num (standard sglang /generate does).
# ----------------------------------------------------------------------------
TEACHER_IP="127.0.0.1"
TEACHER_PORT=13152
TEACHER_LOG="${ROOT_DIR}/teacher_sglang_math_gkd.log"

CUDA_VISIBLE_DEVICES=${TEACHER_GPU} python3 -m sglang.launch_server \
    --model-path "${TEACHER_HF}" \
    --host 0.0.0.0 \
    --port ${TEACHER_PORT} \
    --tp 1 \
    --context-length 16384 \
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
# 2) Training (student) config
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
   # STUDENT is a NUMBERED megatron ckpt (iter 420). Fresh OPD run => load WEIGHTS only and reset
   # iteration -> 0: --no-load-optim/--no-load-rng skip any optimizer/rng state, --finetune treats
   # it as a fine-tune start (iteration 0) instead of resuming @ 420.
   --no-load-optim
   --no-load-rng
   --finetune
   --save          ${MODEL_ROOT}/MultiStageRL/OPD/Qwen3-8B-Base-Math-SeaSFT-Search_GKD_Math/
   --save-interval 20
   --save-retain-interval 100
)

# Math prompt data (dapo-math-17k): chat-list prompts + boxed-answer labels.
ROLLOUT_ARGS=(
   --prompt-data ${MODEL_ROOT}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}            # 24 * 8 = 192 trajectories/rollout (divisible by DP=3)
   --n-samples-per-prompt 8           # reward=0 -> grouping neutral; lower to 4 to cut rollout cost
   --rollout-max-response-len 8192    # math CoT horizon (forward-KL top-k over every token, so kept
                                      # at 8192 rather than the reference 16384 to bound top-k cost)
   --rollout-temperature 1            # forward-KL assumes natural (temp=1) student logits
   --global-batch-size 192            # 192 trajectories -> 1 grad step/rollout; 192/DP(3) = 64
   --balance-data
   # (no dynamic-sampling filter: math is single-turn, nothing aborts.)

   # Save per-rollout trajectories for offline analysis (length/trunc, raw tokens).
   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"

   # ---- eval (real AIME accuracy) ----
   # Eval scores real deepscaler accuracy via generate_eval (bypasses async_rm / the teacher reward),
   # so it works unchanged for GKD and never contacts the teacher. See eval_math.yaml.
   --eval-interval 20
   --eval-config "${SCRIPT_DIR}/eval_math.yaml"
)

# Reuse the EOPD reward: teacher scalar logprob (UNUSED here — no reverse-KL) + per-token top-k
# dist & entropy (the forward-KL signal). The teacher top-k flows regardless of --use-opd.
RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func_eopd
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards_eopd
   --rm-url http://${TEACHER_IP}:${TEACHER_PORT}/generate
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
   --max-tokens-per-gpu 12288         # must be >= the longest single sequence (prompt + resp 8192);
                                      # lower than the OPD-only run to leave room for top-k tensors
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
   # (answer syntax / \boxed{} / digits) with zero gradient. So DISABLE the gate by setting the
   # threshold below any possible entropy (teacher entropy >= 0 always) => gate fires on every
   # token (eopd_highent_frac should log ~1.0).
   --use-eopd-forward-kl
   --eopd-fkl-coef 1.0
   --eopd-entropy-threshold -1.0    # <0 => gate always passes => forward-KL on ALL tokens
   --eopd-topk 16
   # forward-KL coef WARMUP: it is now the SOLE learning signal, so ramp it in gently
   # (0.3 -> 1.0 over the first 40 rollouts) to avoid early entropy/length inflation before the
   # student has re-learned terminating math structure.
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
# 3) Launch Ray on the 6 training GPUs only (teacher is outside Ray).
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
   ${RM_ARGS[@]}

# ----------------------------------------------------------------------------
# 4) Cleanup. Only target this run's processes.
# ----------------------------------------------------------------------------
pkill -9 -f sglang || true       # teacher server + student rollout engines
sleep 3
ray stop --force || true
pkill -9 -f 'ray::' || true
pkill -9 -f train.py || true
sleep 3
