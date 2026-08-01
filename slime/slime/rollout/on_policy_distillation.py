import aiohttp
import torch

from slime.utils.types import Sample


async def reward_func(args, sample, **kwargs):
    payload = {
        # "text": sample.prompt + sample.response,
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs.
    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The actual learning signal comes from the OPD KL penalty applied in compute_advantages_and_returns.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    # Extract teacher log-probs from the sglang response
    teacher_log_probs = [
        torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for reward in raw_rewards
    ]
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards


# ---------------------------------------------------------------------------
# Entropy-Aware OPD (EOPD) — reward fn + post-process that ALSO return the
# teacher's per-token top-k distribution and entropy, on top of the scalar
# teacher log-probs used by the (reverse-KL) advantage path.
#
# Method (EOPD, arXiv:2603.07079): keep the reverse-KL OPD term everywhere
# (mode-seeking; via the advantage path, see apply_opd_kl_to_advantages), and
# ADD a forward-KL term on *high-entropy* teacher tokens (mode-covering), where
# reverse KL alone is unstable / collapses diversity. The forward-KL term needs
# the teacher's top-k distribution per token; the entropy gate needs the
# teacher's per-token entropy. Both are produced here and consumed in
# slime/backends/megatron_utils/loss.py:compute_eopd_forward_kl.
# ---------------------------------------------------------------------------


async def reward_func_eopd(args, sample, **kwargs):
    """Same teacher scoring as reward_func, but also request the teacher's
    top-k logprobs per position (``top_logprobs_num``) so EOPD can build the
    teacher top-k distribution for the forward-KL term."""
    topk = getattr(args, "eopd_topk", 16)
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": topk,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def post_process_rewards_eopd(args, samples: list[Sample], **kwargs):
    """EOPD post-process. In addition to what post_process_rewards does
    (scalar teacher log-probs -> sample.teacher_log_probs for the reverse-KL
    advantage path, task reward 0.0), it extracts the teacher's per-token
    top-k distribution and entropy for the forward-KL term:

      sample.teacher_topk_ids       [response_len, K]  int token ids
      sample.teacher_topk_log_probs [response_len, K]  teacher log-probs (raw)
      sample.teacher_entropy        [response_len]     approx teacher entropy

    The entropy is approximated from the captured top-k mass
    (H ~= -sum_{v in topk} p_v log p_v); it is a lower bound on the true
    teacher entropy, so the gate threshold --eopd-entropy-threshold may need
    tuning relative to a full-vocab entropy. K=16 captures most of the mass.
    """
    topk = getattr(args, "eopd_topk", 16)
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    # --- scalar teacher log-probs for the reverse-KL advantage path (as in OPD) ---
    teacher_log_probs = [
        torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for reward in raw_rewards
    ]
    teacher_log_probs = [
        t[-rl:] for t, rl in zip(teacher_log_probs, response_lengths, strict=False)
    ]

    pad_lp = -1e9  # padded slot: exp(-1e9)=0 -> zero weight in teacher dist / entropy / gather
    for sample, reward, rl in zip(samples, raw_rewards, response_lengths, strict=False):
        # top-k entries: input_top_logprobs[i] is a list of [logprob, token_id, (text)]
        top = reward["meta_info"]["input_top_logprobs"][1:]
        top = top[-rl:] if rl > 0 else []

        ids_rows, lp_rows, ent_rows = [], [], []
        for entry in top:
            entry = entry or []
            lps = [e[0] for e in entry][:topk]
            ids = [int(e[1]) for e in entry][:topk]
            # pad rows to exactly K so the batch is rectangular
            if len(lps) < topk:
                pad_n = topk - len(lps)
                lps = lps + [pad_lp] * pad_n
                ids = ids + [0] * pad_n
            lp_t = torch.tensor(lps, dtype=torch.float32)
            p_t = lp_t.exp()  # raw teacher probs of the top-k (do NOT sum to 1)
            # entropy over captured top-k mass; padded slots (p=0) contribute 0
            ent_rows.append(float(-(p_t * lp_t).sum().item()))
            ids_rows.append(ids)
            lp_rows.append(lps)

        if rl > 0 and ids_rows:
            sample.teacher_topk_ids = torch.tensor(ids_rows, dtype=torch.long)            # [R, K]
            sample.teacher_topk_log_probs = torch.tensor(lp_rows, dtype=torch.float32)    # [R, K]
            sample.teacher_entropy = torch.tensor(ent_rows, dtype=torch.float32)          # [R]
        else:
            sample.teacher_topk_ids = None
            sample.teacher_topk_log_probs = None
            sample.teacher_entropy = None

    # assign scalar teacher log-probs (kept separate so the loop above stays readable)
    for sample, t_lp in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_lp

    # task reward = 0.0 (pure distillation); learning signal is the KL terms.
    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
