"""
Convert saved tau-bench rollout .pt files to SFT training format.

Filters trajectories by reward threshold and outputs a JSONL file
compatible with sft_rollout_from_rollout.py.

The loss_mask in saved rollout data already has:
  - 1 for assistant response tokens (will be trained on)
  - 0 for tool results / user messages (not trained on)
  - System prompt and tools definitions are in the prompt portion,
    which is masked out entirely (not covered by loss_mask at all).

Usage:
    python parse_rollout_to_sft.py \\
        --input-dir /data/user/tau_sft_rollout_data \\
        --output-file /data/user/tau_sft_rollout_data/sft_data.jsonl \\
        --reward-threshold 1.0
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Convert rollout .pt files to SFT JSONL")
    parser.add_argument("--input-dir", required=True, help="Directory containing rollout_*.pt files")
    parser.add_argument("--output-file", required=True, help="Output JSONL file path")
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=1.0,
        help="Minimum reward to include a sample (default: 1.0 = only fully successful trajectories)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=131072,
        help="Maximum total sequence length (prompt + response) to include",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Add slime to path so Sample can be imported
    script_dir = os.path.dirname(os.path.abspath(__file__))
    slime_root = os.path.join(script_dir, "..", "..")
    sys.path.insert(0, slime_root)
    from slime.utils.types import Sample

    # Discover rollout files
    pt_files = sorted(glob.glob(os.path.join(args.input_dir, "rollout_*.pt")))
    if not pt_files:
        print(f"No rollout_*.pt files found in {args.input_dir}")
        sys.exit(1)
    print(f"Found {len(pt_files)} rollout file(s)")

    # Load all samples
    all_samples = []
    for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=False)
        for s_dict in data["samples"]:
            all_samples.append(Sample.from_dict(s_dict))
    print(f"Total samples loaded: {len(all_samples)}")

    # Reward distribution
    reward_dist = defaultdict(int)
    for s in all_samples:
        bucket = round(float(s.reward), 1) if s.reward is not None else None
        reward_dist[bucket] += 1
    print("Reward distribution:", dict(sorted((k, v) for k, v in reward_dist.items() if k is not None)))

    # Status distribution
    status_dist = defaultdict(int)
    for s in all_samples:
        status_dist[str(s.status)] += 1
    print("Status distribution:", dict(status_dist))

    # Filter by reward threshold
    filtered = [
        s for s in all_samples
        if s.reward is not None and float(s.reward) >= args.reward_threshold
    ]
    print(f"After reward filter (>= {args.reward_threshold}): {len(filtered)} samples")

    # Filter truncated / aborted samples (only keep COMPLETED)
    filtered = [s for s in filtered if str(s.status) == "Status.COMPLETED" or "completed" in str(s.status).lower()]
    print(f"After status filter (COMPLETED only): {len(filtered)} samples")

    # Filter by sequence length
    filtered = [s for s in filtered if s.tokens and len(s.tokens) <= args.max_seq_len]
    print(f"After length filter (<= {args.max_seq_len} tokens): {len(filtered)} samples")

    if not filtered:
        print("No samples passed all filters. Exiting.")
        sys.exit(1)

    # Validate and convert each sample
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    written = 0
    skipped = 0
    seq_lengths = []

    with open(args.output_file, "w") as f:
        for sample in filtered:
            tokens = list(sample.tokens)
            loss_mask = list(sample.loss_mask)
            response_length = sample.response_length
            prompt_length = len(tokens) - response_length

            # Sanity check
            if len(loss_mask) != response_length:
                print(
                    f"Warning: loss_mask length {len(loss_mask)} != response_length {response_length} "
                    f"for sample index={sample.index}. Skipping."
                )
                skipped += 1
                continue

            if prompt_length < 0:
                print(f"Warning: negative prompt_length for sample index={sample.index}. Skipping.")
                skipped += 1
                continue

            # Check that at least some response tokens are trained (loss_mask has some 1s)
            if sum(loss_mask) == 0:
                print(f"Warning: all-zero loss_mask for sample index={sample.index}. Skipping.")
                skipped += 1
                continue

            record = {
                "data": {
                    "input_ids": tokens,
                    "loss_mask": loss_mask,
                    "response_length": response_length,
                }
            }
            f.write(json.dumps(record) + "\n")
            written += 1
            seq_lengths.append(len(tokens))

    print(f"\nWritten: {written} samples  |  Skipped (validation failed): {skipped}")
    if seq_lengths:
        print(
            f"Sequence length — min: {min(seq_lengths)}  "
            f"max: {max(seq_lengths)}  "
            f"mean: {sum(seq_lengths) / len(seq_lengths):.0f}"
        )
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
