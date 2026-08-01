"""Per-domain EVAL rollouts for the 3-teacher MOPD run.

The MOPD TRAIN rollout (generate_mixed_opd.generate) routes each prompt to its
domain's native rollout and leaves ``sample.reward = None`` so the OPD teacher RM
scores the student tokens (pure distillation, task reward = 0). For EVAL we want
the opposite: run the SAME native rollout but fill ``sample.reward`` with the REAL
per-domain task metric, so ``generate_and_rm`` skips the teacher RM
(``async_rm`` only fires when ``sample.reward is None``) and eval reports genuine
task accuracy — math \\boxed grade, search EM, tau task success.

Unlike the single-domain OPD examples (math2sea / sea2if / tau2if each ship their
own generate_eval), MOPD needs all three importable from ONE place that is on the
run script's PYTHONPATH. The run script puts ``${SCRIPT_DIR}`` (this multi_teacher
dir), ``examples/search-r1`` and ``examples/tau-bench`` on PYTHONPATH, so:
  * search eval reuses the search-r1 module's own generate_eval verbatim,
  * tau eval mirrors tau2if/generate_with_tau_opd.generate_eval (keep res.reward),
  * math eval reuses the stock single-turn rollout + the deepscaler rule reward
    (math2sea is NOT on PYTHONPATH, so we inline the tiny wrapper here).

Wire each via eval_multi.yaml's per-dataset ``custom_generate_function_path``:
  math   -> generate_eval_mixed.generate_eval_math
  search -> generate_eval_mixed.generate_eval_search
  tau    -> generate_eval_mixed.generate_eval_tau

Each eval dataset is pre-tagged with metadata.domain by prepare_opd_eval_data.py
(harmless here — eval routing is per-dataset via the YAML, not via the domain
tag — but it keeps the eval samples' metadata consistent with train and lets the
per-domain metrics line up).
"""

# math: stock single-turn rollout + deepscaler rule reward (math2sea not on PYTHONPATH).
from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
from slime.rollout.sglang_rollout import generate as _generate_math
from slime.utils.types import Sample

# search: reuse the search-r1 module's own eval wrapper (on PYTHONPATH).
from generate_with_search_tools_qwen_sft_no_drift import generate_eval as _search_generate_eval

# tau: the native tau rollout; we keep its real task reward for eval.
from generate_with_tau import generate as _generate_tau


# An ABORTED eval trajectory produced no gradable answer = task failure = reward 0.0.
# The single-domain generate_eval wrappers leave reward=None on ABORTED, but eval's
# metric reducer does sum(rewards)/len(rewards) (rollout.py) and would crash on a None.
# So we force 0.0 here. (tau's res_to_sample already sets a numeric reward on ABORTED,
# but we guard it too for uniformity.)
def _finalize_aborted_reward(sample: Sample) -> Sample:
    if sample.status == Sample.Status.ABORTED and sample.reward is None:
        sample.reward = 0.0
    return sample


async def generate_eval_math(args, sample: Sample, sampling_params) -> Sample:
    """MATH eval: stock single-turn rollout, then the real rule-based deepscaler
    reward (extract \\boxed{...}, grade vs sample.label). Mirrors
    math2sea/generate_with_math_opd.generate_eval."""
    sample = await _generate_math(args, sample, sampling_params)
    if sample.status != Sample.Status.ABORTED and sample.reward is None:
        sample.reward = get_deepscaler_rule_based_reward(sample.response, sample.label)
    return _finalize_aborted_reward(sample)


async def generate_eval_search(args, sample: Sample, sampling_params) -> Sample:
    """SEARCH eval: delegate to the search-r1 module's own generate_eval, which runs
    the search rollout and fills sample.reward with the EM task score."""
    sample = await _search_generate_eval(args, sample, sampling_params)
    return _finalize_aborted_reward(sample)


async def generate_eval_tau(args, sample: Sample, sampling_params) -> Sample:
    """TAU eval: run the tau-bench trajectory and KEEP the real task reward
    (res.reward), so generate_and_rm skips the teacher RM and eval reports true
    task success. Mirrors tau2if/generate_with_tau_opd.generate_eval."""
    sample = await _generate_tau(args, sample, sampling_params)
    return _finalize_aborted_reward(sample)
