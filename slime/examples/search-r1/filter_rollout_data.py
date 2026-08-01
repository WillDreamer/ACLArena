"""
Filter rollout data by reward threshold.

Usage:
    python filter_rollout_data.py \
        --input-dir /data/user/rollout_only_search_Qwen3-30B-A3B \
        --output-dir /data/user/rollout_only_search_Qwen3-30B-A3B_filtered \
        --min-reward 0.8 \
        --batch-size 64
"""

import argparse
import glob
import os
import re
from collections import Counter

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Filter rollout data by reward")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing rollout_*.pt files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for filtered data")
    parser.add_argument("--min-reward", type=float, default=0.4, help="Minimum reward threshold (default: 0.4)")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Number of samples per output .pt file (default: 64)"
    )
    return parser.parse_args()


def load_rollout_files(input_dir):
    """Load all rollout_*.pt files (excluding eval files)."""
    pattern = os.path.join(input_dir, "rollout_*.pt")
    files = sorted(glob.glob(pattern))

    # Exclude eval files (rollout_eval_*.pt)
    files = [f for f in files if not re.search(r"rollout_eval_", os.path.basename(f))]

    print(f"Found {len(files)} rollout files in {input_dir}")
    return files


def filter_samples(files, min_reward):
    """Load and filter samples from all files."""
    all_samples = []
    total_count = 0
    reward_bins = Counter()

    for filepath in files:
        data = torch.load(filepath, weights_only=False)
        samples = data.get("samples", [])
        total_count += len(samples)

        for sample in samples:
            reward = sample.get("reward")
            status = sample.get("status", "")

            # Categorize reward for statistics
            if reward is None:
                reward_bins["None"] += 1
                continue
            elif isinstance(reward, dict):
                # If reward is a dict, skip (need --reward-key to select)
                reward_bins["dict"] += 1
                continue

            if reward > 0.4:
                reward_bins["> 0.4"] += 1
            elif reward > 0:
                reward_bins["0 < r <= 0.4"] += 1
            elif reward == 0:
                reward_bins["== 0"] += 1
            else:
                reward_bins["< 0"] += 1

            # Filter: reward >= min_reward and not failed/aborted
            if reward >= min_reward and status not in ("failed", "aborted"):
                # Skip samples whose loss_mask is all-zeros (e.g. direct-answer samples
                # masked out by the anti-collapse mask_no_tool_call_warmup mechanism).
                # Such samples contribute zero loss during SFT training.
                loss_mask = sample.get("loss_mask")
                if loss_mask is not None and sum(loss_mask) == 0:
                    reward_bins["zero_loss_mask"] += 1
                    continue
                all_samples.append(sample)

    return all_samples, total_count, reward_bins


def save_filtered_data(samples, output_dir, batch_size):
    """Save filtered samples in batches, compatible with --load-debug-rollout-data."""
    os.makedirs(output_dir, exist_ok=True)

    num_files = (len(samples) + batch_size - 1) // batch_size
    for i in range(num_files):
        batch = samples[i * batch_size : (i + 1) * batch_size]
        output_path = os.path.join(output_dir, f"rollout_{i}.pt")
        torch.save({"rollout_id": i, "samples": batch}, output_path)

    print(f"Saved {num_files} files to {output_dir}")
    print(f"  File pattern: rollout_{{0..{num_files - 1}}}.pt")
    print(f"  Use with: --load-debug-rollout-data {output_dir}/rollout_{{rollout_id}}.pt")
    print(f"  Set:      --num-rollout {num_files}")
    return num_files


def main():
    args = parse_args()

    # Load all rollout files
    files = load_rollout_files(args.input_dir)
    if not files:
        print("No rollout files found. Exiting.")
        return

    # Filter samples
    filtered_samples, total_count, reward_bins = filter_samples(files, args.min_reward)

    # Print statistics
    print(f"\n{'=' * 50}")
    print(f"Total samples scanned:  {total_count}")
    print(f"Reward distribution:")
    for k, v in sorted(reward_bins.items()):
        print(f"  {k:>15}: {v:>6} ({v / total_count * 100:.1f}%)")
    print(f"Filtered (reward >= {args.min_reward}): {len(filtered_samples)}")
    print(f"Pass rate: {len(filtered_samples) / total_count * 100:.2f}%")
    print(f"{'=' * 50}\n")

    if not filtered_samples:
        print("No samples passed the filter. Try lowering --min-reward or generating more rollout data.")
        return

    # Show reward distribution of filtered samples
    filtered_rewards = [s["reward"] for s in filtered_samples]
    avg_reward = sum(filtered_rewards) / len(filtered_rewards)
    print(f"Filtered samples stats:")
    print(f"  Count: {len(filtered_samples)}")
    print(f"  Avg reward: {avg_reward:.4f}")
    print(f"  Min reward: {min(filtered_rewards):.4f}")
    print(f"  Max reward: {max(filtered_rewards):.4f}")

    # Save
    num_files = save_filtered_data(filtered_samples, args.output_dir, args.batch_size)
    print(f"\nDone! You can now run SFT with --num-rollout {num_files}")


if __name__ == "__main__":
    main()
