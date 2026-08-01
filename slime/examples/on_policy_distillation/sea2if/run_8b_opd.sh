#!/bin/bash
#
# Single-teacher on-policy distillation (OPD), SEARCH domain, all-local on one 8-GPU box.
#
#   student : MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau-IF  (megatron torch_dist, --load)
#   teacher : MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search (search specialist) -> local SGLang server
#
# Pure distillation: task reward = 0; the only learning signal is the OPD KL penalty
# (advantage -= opd_kl_coef * (student_log_prob - teacher_log_prob), per token).
# The student (the IF-RL'd model) rolls out with the SEARCH multi-turn generate fn (calls the
# retriever), and the teacher scores token-level logprobs over the student's own trajectory.
# Goal: distill the search specialist's behavior back into the IF model to recover search skill.
#
# GPU layout on this single node (8x H200):
#   GPU 0      -> free (retriever is REMOTE @ 131.179.168.117:8000, not launched here)
#   GPU 1      -> teacher SGLang server       (:13141)    [outside Ray]
#   GPU 2..7   -> Ray: colocated actor + student rollout  (6 GPUs)
#
# usage: cd /data/user/slime && bash examples/on_policy_distillation/sea2if/run_8b_opd.sh

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
WANDB_GROUP="opd_search2if_bs_${ROLLOUT_BATCH_SIZE}"
# Per-rollout debug dumps (tokens / loss_mask / reward / response_length / rollout_log_probs).
# Saved as rollout_<id>.pt (train) and rollout_eval_<id>.pt (eval); parent dir auto-created.
ROLLOUT_DEBUG_DIR="${MODEL_ROOT}/MultiStageRL/${WANDB_GROUP}"

# Paths
STUDENT=${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau-IF
TEACHER_TD=${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search        # megatron torch_dist ckpt
TEACHER_HF=${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search/hf_iter_0000420     # HF copy for SGLang (auto-built)
BASE_HF=${MODEL_ROOT}/Qwen3-8B-Base/                       # tokenizer + config

export PYTHONPATH="${ROOT_DIR}/slime/examples/search-r1:${ROOT_DIR}/slime:${ROOT_DIR}/Megatron-LM:${PYTHONPATH}"

# ----------------------------------------------------------------------------
# 0) Bootstrap: teacher torch_dist -> HF (SGLang serves HF). Done once.
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
# 1) Remote retriever (Search-R1 FAISS server @ 131.179.168.117:8000). The search
#    generate fn (generate_with_search_tools_qwen_sft_no_drift) points search_url
#    here. We do NOT launch anything locally -- just verify it's reachable so we
#    fail fast instead of after teacher + train startup.
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
# ----------------------------------------------------------------------------
TEACHER_IP="127.0.0.1"
TEACHER_PORT=13141
TEACHER_LOG="${ROOT_DIR}/teacher_sglang.log"

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
   --save          ${MODEL_ROOT}/MultiStageRL/OPD/Qwen3-8B-Base-TauSFT-Tau-IF_OPD_Search/
   --save-interval 30
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
   --rollout-max-response-len 4096
   --rollout-temperature 1
   --global-batch-size 192            # 192/96 = 2 grad steps/rollout; 96/DP(3) = 32
   --balance-data

   # Save per-rollout trajectories for offline analysis (length/turns/trunc, raw tokens).
   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"

   # ---- eval (search task accuracy / EM) ----
   # Eval runs the SEARCH rollout and scores EM via the eval-only generate fn
   # (generate_eval), NOT the OPD teacher reward. Because this run's
   # --custom-rm-path is the teacher reward and async_rm always prefers it, the
   # per-dataset custom_generate_function_path in eval_search.yaml is what makes
   # eval report real task accuracy without contacting the teacher. The eval set
   # + keys mirror the search-r1 reference run; sampling params fall back to the
   # rollout values (temperature=1, max_response_len=4096).
   --eval-interval 25
   --eval-config "${SCRIPT_DIR}/eval_search.yaml"
)

# OPD reward: per-sample teacher token-level logprobs from the SGLang server; task reward = 0.
RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
   --rm-url http://${TEACHER_IP}:${TEACHER_PORT}/generate
)

# Search multi-turn rollout (student turns -> local sglang router; search tool -> retriever @ :8000).
# Pure OPD: NO --use-tis and NO dynamic-sampling filter (reward=0 would make a reward-std filter
# delete every group), matching the multi_teacher OPD precedent.
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search_tools_qwen_sft_no_drift.generate
)

# 6 training GPUs: TP=2, PP=1, CP=1 -> DP=3.  (TP must divide 8 KV groups -> {1,2,4}.)
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
   --opd-kl-coef 0.5            # pull the RL'd student back toward the search teacher
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
# prefilled. Read by generate_with_search_tools_qwen_sft_no_drift. Must be set before ray start.
export SEARCH_R1_STRIP_THINK=0

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
# 5) Cleanup. The retriever is REMOTE (131.179.168.117) so nothing to stop here;
#    we still avoid a blanket `pkill python` as a matter of habit and only target
#    this run's processes.
# ----------------------------------------------------------------------------
pkill -9 -f sglang || true       # teacher server + student rollout engines
sleep 3
ray stop --force || true
pkill -9 -f 'ray::' || true
pkill -9 -f train.py || true
sleep 3
