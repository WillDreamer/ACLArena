"""Group-gated loss-mask filter for search-r1 training.

When a rollout group contains more than 50% trajectories without a real
tool execution, zero out the ``loss_mask`` of those no-tool samples. They
still participate in group-relative advantage computation (their reward is
untouched), but contribute zero gradient — this prevents the
"answer-directly-skip-search" shortcut from dominating the policy update
while still using the no-tool samples as a baseline to push tool-call
samples toward positive advantage.

Wire via:
    --dynamic-sampling-filter-path no_tool_loss_mask_filter.no_tool_loss_mask_filter
"""

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

from qa_em_format_qwen_sft import executed_tool_call

# Strictly greater than this fraction of no-tool samples triggers masking.
_NO_TOOL_RATIO_THRESHOLD = 0.3


def no_tool_loss_mask_filter(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    if not samples:
        return DynamicFilterOutput(keep=True)

    # executed_tool_call MUST be fed the assistant trajectory only — the
    # system prompt itself contains a literal <tool_response></tool_response>
    # example which would otherwise make every sample look like a real call.
    has_tool_flags = [executed_tool_call(s.response or "") for s in samples]
    n = len(samples)
    no_tool_count = n - sum(has_tool_flags)
    no_tool_ratio = no_tool_count / n

    if no_tool_ratio <= _NO_TOOL_RATIO_THRESHOLD:
        return DynamicFilterOutput(keep=True)

    # Entire group is no-tool: zeroing every sample produces a no-op
    # gradient group. Drop it so the oversampling loop refills with a
    # useful group instead of wasting a microbatch slot.
    if no_tool_count == n:
        return DynamicFilterOutput(keep=False, reason="all_no_tool")

    for sample, has_tool in zip(samples, has_tool_flags, strict=True):
        if has_tool:
            continue
        if sample.loss_mask is None:
            sample.loss_mask = [0] * sample.response_length
        else:
            sample.loss_mask = [0] * len(sample.loss_mask)

    return DynamicFilterOutput(keep=True)
