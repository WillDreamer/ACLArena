#!/bin/bash
#
# EOPD v2 — Entropy-Aware On-Policy Distillation with a COMPLEMENTARY entropy gate on the
# reverse-KL term. SEARCH domain, all-local on one 8-GPU box. Based on run_8b_eopd.sh.
#
#   student : MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau-IF   (megatron torch_dist, --load)
#   teacher : MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search                 (search specialist) -> local SGLang server
#
# What changed vs v1 (run_8b_eopd.sh)
# -----------------------------------
# v1 EOPD is asymmetric: the forward-KL term is entropy-gated (fires only where teacher H > tau),
# but the reverse-KL OPD term (advantage path) fires on ALL tokens. So on HIGH-entropy tokens the
# two overlap and fight — forward-KL wants to COVER the teacher's modes, reverse-KL wants to
# COLLAPSE onto one — exactly where reverse-KL is most prone to mode-collapse.
#
# v2 adds --use-opd-entropy-gate, which restricts the reverse-KL term to LOW-entropy teacher
# tokens (H <= tau), the EXACT COMPLEMENT of the forward-KL gate. The token split becomes clean:
#   * HIGH-entropy tokens (open reasoning / phrasing)   -> forward-KL only (mode-covering / diversity)
#   * LOW-entropy  tokens (answer / format / facts)     -> reverse-KL only (mode-seeking / sharpening)
# This is the textbook EOPD entropy split; v1 only did half of it. The gate threshold defaults to
# --eopd-entropy-threshold (0.4 here), so forward and reverse partition tokens with no overlap and
# no gap: in wandb, eopd_highent_frac + opd_lowent_frac should be ~1.0.
#
# NOTE (expectation): this is a "purification", not a "cure" for the terminator-collapse root cause.
# The turn-terminator <|im_end|> is a LOW-entropy token, so reverse-KL still covers it (the teacher
# assigns it ~1e-8 prob -> a large negative advantage). v2's win is removing the forward/reverse
# overlap on high-entropy tokens (cleaner diversity) and letting reverse-KL focus on the low-entropy
# sharpening it is actually good at. Because gating REMOVES reverse-KL from high-entropy tokens, its
# total influence shrinks; if sharpening weakens, consider raising --opd-kl-coef 0.3 -> 0.5.
#
# WANDB METRICS TO WATCH (all emitted automatically; grouped by side)
# ------------------------------------------------------------------
#  distillation signal
#    opd_reverse_kl      reverse-KL magnitude (POST-gate; now only low-entropy tokens contribute)
#    opd_lowent_frac     [v2 NEW] fraction of tokens where reverse-KL fires (low-entropy coverage)
#    eopd_forward_kl     forward-KL loss magnitude
#    eopd_highent_frac   fraction of tokens where forward-KL fires (high-entropy coverage)
#    eopd_fkl_coef       forward-KL coef (shows the 0.3->1.0 warmup ramp over the first 40 rollouts)
#    >>> SANITY: eopd_highent_frac + opd_lowent_frac should be ~1.0 (gates are exact complements)
#  training stability
#    loss / pg_loss / entropy_loss (== mean student entropy) / ppo_kl / pg_clipfrac
#    teacher_entropy     mean teacher entropy (curriculum + gate driver)
#  rollout health (rollout/ prefix)
#    response_len/*      length distribution (watch for reverse-KL length runaway)
#    truncated_ratio     truncation rate (collapse early-warning; v1 OPD hit 1.0 at collapse)
#    repetition_frac     degenerate-repetition fraction
#    num_turns           mean search turns (collapse pins this at the 5-turn cap)
#  task performance (eval/ prefix, every --eval-interval)
#    eval/<key>          EM / answer_correct on the held-out set (the real objective)
#    eval/<key>-truncated_ratio
#
# Constraints (same as v1): forward-KL AND the reverse-KL gate need --context-parallel-size 1 and
# assume --rollout-temperature 1 (student logits = natural distribution).
#
# GPU layout (8x H200): GPU0 free (retriever REMOTE); GPU1 teacher SGLang server; GPU2..7 Ray train+rollout.
#
# usage: cd /data/user/slime && bash examples/on_policy_distillation/sea2if/run_8b_eopd_v2.sh

set -ex

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
WANDB_GROUP="eopd_v2_search2if_bs_${ROLLOUT_BATCH_SIZE}"
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
#    EOPD needs per-token top-k logprobs, so the teacher must support
#    top_logprobs_num (standard sglang /generate does).
#    Distinct port/log from opd(13141)/gkdopd(13142)/eopd(13141) so v2 can run without clashing.
# ----------------------------------------------------------------------------
TEACHER_IP="127.0.0.1"
TEACHER_PORT=13143
TEACHER_LOG="${ROOT_DIR}/teacher_sglang_v2.log"

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
   --no-load-optim
   --no-load-rng
   --finetune
   --save          ${MODEL_ROOT}/MultiStageRL/OPD/Qwen3-8B-Base-TauSFT-Tau-IF_EOPDv2_Search/
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
   --rollout-max-response-len 4096    # curriculum CAP (final horizon); see flags below
   # --- entropy-/truncation-gated response-length curriculum (inherited stabilizer) ---
   --enforce-total-response-budget
   --use-length-curriculum
   --length-curriculum-init-len 2048      # start horizon (early healthy rollouts were ~1.1-1.6k)
   --length-curriculum-max-len 4096       # final horizon (== rollout-max-response-len)
   --length-curriculum-increment 512      # +512 tokens per advance
   --length-curriculum-interval 5         # at most one advance every 5 rollouts
   --length-curriculum-trunc-max 0.30     # advance only if last-rollout truncation <= 0.30
   --length-curriculum-entropy-min 0.6    # ...and mean teacher entropy >= 0.6 (safety floor)
   --rollout-temperature 1            # EOPD forward-KL assumes natural (temp=1) student logits
   --global-batch-size 192            # 192/192 = 1 grad step/rollout; 192/DP(3) = 64
   --balance-data

   # Save per-rollout trajectories for offline analysis (length/turns/trunc, raw tokens).
   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"

   # ---- eval (search task accuracy / EM) ----
   --eval-interval 25
   --eval-config "${SCRIPT_DIR}/eval_search.yaml"
)

# EOPD reward: teacher scalar logprob (reverse KL) + per-token top-k dist & entropy (forward KL).
RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func_eopd
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards_eopd
   --rm-url http://${TEACHER_IP}:${TEACHER_PORT}/generate
)

# Search multi-turn rollout (student turns -> local sglang router; search tool -> retriever @ :8000).
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search_tools_qwen_sft_no_drift.generate
)

# 6 training GPUs: TP=2, PP=1, CP=1 -> DP=3.  CP MUST be 1 for the EOPD forward-KL term AND the gate.
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
   # --- FORWARD-HEAVY KL mix (inherited from v1) ---
   --use-opd
   --opd-type sglang
   --opd-kl-coef 0.3                  # reverse-KL (mode-seeking). Gating shrinks its coverage; if
                                      # sharpening weakens vs v1, try 0.5 to compensate.
   # --- v2: COMPLEMENTARY ENTROPY GATE on reverse-KL ---
   # Restrict reverse-KL to LOW-entropy teacher tokens (H <= threshold) — the exact complement of
   # the forward-KL high-entropy gate below. Threshold defaults to --eopd-entropy-threshold (0.4),
   # so the two gates partition every token with no overlap and no gap (highent_frac+lowent_frac~1).
   --use-opd-entropy-gate
   # --opd-entropy-gate-threshold 0.4   # (optional) override; unset => reuse --eopd-entropy-threshold
   # EOPD forward-KL (mode-covering) on HIGH-entropy tokens.
   --use-eopd-forward-kl
   --eopd-fkl-coef 1.0               # forward-KL, mode-covering — warmup TARGET
   --eopd-entropy-threshold 0.4     # forward-KL fires where H>0.4; reverse-KL fires where H<=0.4
   --eopd-topk 16
   # --- forward-KL coef WARMUP ---
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
