import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

__all__ = ["check_reward_nonzero_std", "check_raw_task_reward_nonzero_std"]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def check_raw_task_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    """Filter on the raw 0/1 task outcome stored in metadata['raw_task_reward'],
    ignoring any additive penalties (format / length / truncation) the rollout
    code applied to sample.reward. With penalties enabled, the post-modification
    reward almost never has zero std even when every sample failed (because
    each penalty contributes its own small variance), which defeats the
    "only train on groups that contain at least one success" intent. Reading
    the raw task reward restores the original {0,1} std semantics: a group is
    kept iff it contains both a success and a failure on the underlying task.
    """
    raw_rewards = [
        float(sample.metadata.get("raw_task_reward", sample.get_reward_value(args)))
        if sample.metadata is not None
        else float(sample.get_reward_value(args))
        for sample in samples
    ]
    keep = torch.tensor(raw_rewards, dtype=torch.float64).std() > 1e-6
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"raw_zero_std_{round(raw_rewards[0], 1)}",
    )
