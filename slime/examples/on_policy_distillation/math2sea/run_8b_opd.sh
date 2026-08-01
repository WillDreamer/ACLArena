#!/bin/bash
#
# Single-teacher on-policy distillation (OPD, reverse-KL only), MATH domain, all-local on one 8-GPU box.
#
#   student : MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search   (megatron torch_dist, --load; the model
#             that was SFT'd on Sea then RL'd on Search and has FORGOTTEN math)
#   teacher : MultiStageRL/Qwen3-8B-Base-Math                 (math specialist) -> local SGLang server
#
# Goal: distill the math specialist's behavior BACK into the search-trained student to recover the
# forgotten math capability (math2sea = supplement math into the sea/search model).
#
# Pure distillation: task reward = 0; the only learning signal is the OPD KL penalty
# (advantage -= opd_kl_coef * (student_log_prob - teacher_log_prob), per token). The student rolls
# out single-turn on math prompts (the teacher's domain), and the teacher scores token-level
# logprobs over the student's own trajectory.
#
# Math-domain wiring (vs the tau2if / sea2if multi-turn scripts):
#   * TRAIN rollout = the STOCK single-turn sglang rollout (NO --custom-generate-function-path).
#     The stock rollout never sets sample.reward, so the OPD teacher RM (async_rm) is invoked
#     automatically and fills sample.teacher_log_probs. This is exactly what the reference
#     run-qwen3-8B-opd-math.sh relies on -- no tau-style "clear the task reward" wrapper is needed.
#   * eval  = generate_with_math_opd.generate_eval (via eval_math.yaml): runs the same single-turn
#     rollout but assigns the REAL rule-based (deepscaler) math reward, so eval reports true AIME
#     accuracy, not the teacher logprob.
#   * NO user simulator, NO tools, NO dynamic-sampling filter (math is single-turn; nothing aborts).
#
# GPU layout on this single node (8x H200):
#   GPU 0      -> free   (available to scale the teacher to a 2nd engine / TP=2 if scoring is the bottleneck)
#   GPU 1      -> teacher SGLang server       (:13151)    [outside Ray]
#   GPU 2..7   -> Ray: colocated actor + student rollout  (6 GPUs)
#
# usage: cd /data/user/slime && bash examples/on_policy_distillation/math2sea/run_8b_opd.sh

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
WANDB_GROUP="opd_math2sea_bs_${ROLLOUT_BATCH_SIZE}"
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
# ----------------------------------------------------------------------------
TEACHER_IP="127.0.0.1"
TEACHER_PORT=13151
TEACHER_LOG="${ROOT_DIR}/teacher_sglang_math.log"

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
   # STUDENT is a NUMBERED megatron ckpt (latest_checkpointed_iteration.txt == 420). This is a
   # *fresh* OPD run with its own schedule, so load the student WEIGHTS only and reset iteration
   # -> 0 (so all --num-rollout rollouts run): --no-load-optim/--no-load-rng skip any optimizer/rng
   # state, and --finetune treats it as a fine-tune start (iteration 0) instead of resuming @ 420.
   --no-load-optim
   --no-load-rng
   --finetune
   --save          ${MODEL_ROOT}/MultiStageRL/OPD/Qwen3-8B-Base-Math-SeaSFT-Search_OPD_Math/
   --save-interval 20
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
   --rollout-max-response-len 8192    # math CoT horizon (reference run-qwen3-8B-opd-math.sh used 16384;
                                      # halved here to fit the local teacher + keep parity with the
                                      # EOPD/GKD top-k scripts. Raise for the OPD-only run if desired.)
   --rollout-temperature 1
   --global-batch-size 192            # 192 trajectories -> 1 grad step/rollout; 192/DP(3) = 64
   --balance-data
   # (no dynamic-sampling filter: math is single-turn, nothing aborts; matches run-qwen3-8B-opd-math.sh.)

   # Save per-rollout trajectories for offline analysis (length/trunc, raw tokens).
   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"

   # ---- eval (real AIME accuracy) ----
   # Eval runs the single-turn math rollout and scores the real deepscaler reward via the eval-only
   # generate fn (generate_eval), NOT the OPD teacher reward. Because this run's --custom-rm-path is
   # the teacher reward and async_rm always prefers it, the per-dataset custom_generate_function_path
   # in eval_math.yaml is what makes eval report real accuracy without contacting the teacher.
   --eval-interval 20
   --eval-config "${SCRIPT_DIR}/eval_math.yaml"
)

# OPD reward: per-sample teacher token-level logprobs from the SGLang server; task reward = 0.
RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
   --rm-url http://${TEACHER_IP}:${TEACHER_PORT}/generate
)

# 6 training GPUs: TP=2, PP=1, CP=1 -> DP=3.  CP is kept at 1 for parity with the EOPD/GKD scripts
# (whose forward-KL term requires CP=1); recompute-full keeps activation memory low for long CoTs.
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
   --max-tokens-per-gpu 16384         # must be >= the longest single sequence (prompt + resp 8192)
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef 0.5            # reverse-KL, mode-seeking: pull the search student back toward the math teacher
                               # (the reference run-qwen3-8B-opd-math.sh used 1.0; raise if under-distilling)
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
