"""Dynamic-sampling filter for tau-domain OPD.

Drop any prompt-group that contains an ABORTED sample.

An aborted tau-bench trajectory (e.g. the user-simulator LLM call in
trainable_agents' env.reset raised because the remote user-sim @ :8098 was
flaky/down) skips the OPD teacher RM: slime's generate_and_rm returns early for
ABORTED samples (sglang_rollout.py), so async_rm is never called and the sample
carries NO teacher logprob (sample.reward stays None). If such a sample reaches
on_policy_distillation.post_process_rewards[_eopd], it tries to read teacher
meta_info off a None reward and crashes the whole run
(TypeError: 'NoneType' object is not subscriptable) -- which is exactly what
killed eopd_tau2if at rollout 69 during a user-sim outage.

Dropping groups with any aborted sample keeps the trained batch limited to
fully teacher-scored samples. slime's rollout loop regenerates dropped groups
(the over-sampling replacement path in generate_rollout_async), so the batch
still fills to rollout_batch_size. COMPLETED and TRUNCATED samples are kept:
TRUNCATED still runs the teacher RM (it is not ABORTED), so it has valid teacher
logprobs.

NOTE: this filter is keyed on sample.status, NOT on reward std, so it composes
with pure OPD where the task reward is a constant 0 (a reward-std filter would
delete every group). It does not fully protect against a TOTAL user-sim outage
(every group aborts -> the rollout loop spins regenerating); the run scripts'
startup reachability check is what guards against a pre-existing outage.
"""

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample


def drop_group_with_aborted(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    n_aborted = sum(1 for s in samples if s.status == Sample.Status.ABORTED)
    keep = n_aborted == 0
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"aborted_{n_aborted}/{len(samples)}",
    )
