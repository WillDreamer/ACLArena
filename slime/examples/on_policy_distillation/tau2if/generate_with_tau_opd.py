"""
OPD train/eval rollout wrappers for the tau-bench domain.

Why this module exists (mirrors examples/search-r1's generate vs generate_eval split):

On-policy distillation sets ``--custom-rm-path`` to the OPD *teacher* reward
(``slime.rollout.on_policy_distillation.reward_func`` / ``reward_func_eopd``).
slime's ``generate_and_rm`` only calls that RM (``async_rm``) when
``sample.reward is None`` (see slime/rollout/sglang_rollout.py: the single-sample
branch does ``if sample.reward is None: sample.reward = await async_rm(...)``).

The stock tau-bench rollout (examples/tau-bench/generate_with_tau.generate)
ALWAYS sets ``sample.reward`` to the tau-bench task reward. If used directly for
OPD training that would (a) suppress the teacher RM (reward is already set) so no
teacher token-logprobs are produced, and (b) leave a float in ``sample.reward``
where ``post_process_rewards`` expects the teacher's sglang logprob JSON. So:

  * generate()      -> run the tau-bench trajectory, then DROP the task reward
                       (``sample.reward = None``) so the OPD teacher RM scores the
                       student's own tokens. Use this as the TRAIN
                       ``--custom-generate-function-path``.
  * generate_eval() -> run the tau-bench trajectory and KEEP the real tau-bench
                       task reward, so ``generate_and_rm`` skips the teacher RM
                       and eval reports true task success. Wired per-dataset via
                       eval_tau.yaml's ``custom_generate_function_path``.

``generate_with_tau`` (and its trainable_agents / tau_bench deps) is importable
because the run scripts put examples/tau-bench on PYTHONPATH.
"""

import generate_with_tau as _tau

from slime.utils.types import Sample


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """TRAIN rollout: tau-bench trajectory with the task reward cleared so the OPD
    teacher RM (async_rm -> on_policy_distillation.reward_func[_eopd]) runs and
    fills in sample.teacher_log_probs (+ teacher top-k/entropy for EOPD/GKD)."""
    sample = await _tau.generate(args, sample, sampling_params)
    # OPD: pure distillation. The only learning signal is the teacher KL term(s);
    # the tau task reward is intentionally discarded so the teacher RM is invoked.
    sample.reward = None
    return sample


async def generate_eval(args, sample: Sample, sampling_params) -> Sample:
    """EVAL rollout: keep the real tau-bench task reward (res.reward), so
    generate_and_rm sees a non-None reward, skips the teacher RM, and eval scores
    true task success instead of the teacher logprob."""
    return await _tau.generate(args, sample, sampling_params)
