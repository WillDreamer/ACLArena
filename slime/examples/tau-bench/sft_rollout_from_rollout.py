"""
Custom SFT rollout function that loads pre-tokenized trajectories from rollout data.

Reads samples whose `sample.prompt` is a dict produced by parse_rollout_to_sft.py
(loaded via --input-key data) and directly sets tokens / loss_mask / response_length.
No re-tokenization, no SGLang generation.

Use with:
    --rollout-function-path sft_rollout_from_rollout.generate_rollout
    --prompt-data /path/to/sft_data.jsonl
    --input-key data
    --rollout-global-dataset
    --loss-type sft_loss
    --calculate-per-token-loss
    --disable-compute-advantages-and-returns
    --debug-train-only
"""

import logging

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)

SAMPLE_PRINTED = False


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """SFT rollout: load pre-tokenized trajectories from data_buffer.

    Each sample.prompt is a dict with keys:
        input_ids       (list[int])  full token sequence (prompt + response)
        loss_mask       (list[int])  0/1 per response token; 1 = train on this token
        response_length (int)        number of response tokens

    The loss_mask already reflects masking of system prompt, tools definitions,
    tool results, and user messages — only assistant response tokens are 1.
    """
    assert not evaluation
    assert args.rollout_global_dataset

    global SAMPLE_PRINTED

    samples = data_buffer.get_samples(args.rollout_batch_size)

    for i, sample in enumerate(samples):
        (sample,) = sample

        data = sample.prompt
        assert isinstance(data, dict), (
            f"Expected sample.prompt to be a dict (from --input-key data), got {type(data)}. "
            "Make sure you set --input-key data and generated the JSONL with parse_rollout_to_sft.py."
        )

        input_ids = data["input_ids"]
        loss_mask = data["loss_mask"]
        response_length = data["response_length"]

        sample.tokens = input_ids
        sample.loss_mask = loss_mask
        sample.response_length = response_length
        sample.reward = 0

        if i == 0 and not SAMPLE_PRINTED:
            logger.info(
                f"sft_rollout_from_rollout example: "
                f"total_len={len(input_ids)} response_length={response_length} "
                f"trainable_tokens={sum(loss_mask)} "
                f"input_ids[:8]={input_ids[:8]}"
            )
            SAMPLE_PRINTED = True

    return samples
