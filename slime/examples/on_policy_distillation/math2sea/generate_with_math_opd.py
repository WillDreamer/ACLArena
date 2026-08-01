"""
OPD eval rollout wrapper for the MATH domain (math2sea).

Why this module exists (mirrors examples/search-r1 and tau2if's generate vs
generate_eval split):

On-policy distillation sets ``--custom-rm-path`` to the OPD *teacher* reward
(``slime.rollout.on_policy_distillation.reward_func`` / ``reward_func_eopd``).
slime's ``generate_and_rm`` only calls that RM (``async_rm``) when
``sample.reward is None`` (see slime/rollout/sglang_rollout.py), and ``async_rm``
ALWAYS prefers ``args.custom_rm_path`` when it is set. So if eval reused the plain
math rollout it would score every trajectory with the TEACHER logprob reward
(and need the teacher server up at eval time) — useless as a math-accuracy metric.

Unlike tau-bench / search-R1, the MATH *train* rollout needs NO custom generate
function: the stock single-turn sglang rollout (slime.rollout.sglang_rollout.generate)
never sets sample.reward, so the OPD teacher RM is invoked automatically and fills
sample.teacher_log_probs (+ teacher top-k / entropy for EOPD/GKD). That is exactly
what run-qwen3-8B-opd-math.sh relies on. Hence the run scripts pass NO
--custom-generate-function-path for training.

For EVAL we only need to bypass the teacher: run the SAME stock single-turn
rollout, then fill in sample.reward with the real rule-based MATH score
(deepscaler, which extracts \\boxed{...} and grades against the label). With the
reward already set, ``generate_and_rm`` skips ``async_rm`` and eval reports genuine
math accuracy without contacting the teacher.

Wire it ONLY for eval via the eval dataset config's ``custom_generate_function_path``
(see eval_math.yaml). ``generate_with_math_opd`` is importable because the run
scripts put this dir (math2sea) on PYTHONPATH.
"""

from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
from slime.rollout.sglang_rollout import generate as _default_generate
from slime.utils.types import Sample


async def generate_eval(args, sample: Sample, sampling_params) -> Sample:
    """EVAL rollout: run the stock single-turn math rollout, then assign the real
    rule-based math reward (deepscaler: extract \\boxed{...}, grade vs sample.label).
    With the reward set, generate_and_rm skips the OPD teacher RM and eval reports
    true math accuracy instead of the teacher logprob."""
    sample = await _default_generate(args, sample, sampling_params)
    if sample.status != Sample.Status.ABORTED and sample.reward is None:
        sample.reward = get_deepscaler_rule_based_reward(sample.response, sample.label)
    return sample
