import torch
import numpy as np

path = "/shared/user/MultiStageRL/tau-bench_Qwen3-8B_bs_64/rollout_15.pt"
d = torch.load(path, weights_only=False)

print("rollout_id:", d["rollout_id"])
print("samples 数量:", len(d["samples"]))

# 看前 2 个 sample 的主要字段
st = 12
for i, s in enumerate(d["samples"][st:st+1]):
    print("\n--- sample", i, "---")
    print("prompt :", (s.get("prompt") or "")[:])
    print("response :", (s.get("response") or "")[:])
    print("reward:", s.get("reward"))
    # print("response_length:", s.get("response_length"))


# # 统计有多少个reward大于0的序列
# count_positive_reward = 0
# positive_reward_samples = []
# for s in d["samples"]:
#     if s.get("reward", 0) > 0:
#         print(s.get("response"))
#         count_positive_reward += 1
#         # print("response :", (s.get("response") or "")[:], "...")
# print(f"reward > 0 的序列数量: {count_positive_reward} / {len(d['samples'])}")

# # 统计有多少个reward大于0的序列
# count_positive_reward = 0
# positive_reward_samples = []
# for s in d["samples"]:
#     if s.get("reward", 0) > 0.4:
#         print(s.get("response"))
#         count_positive_reward += 1
# print(f"reward > 0.4 的序列数量: {count_positive_reward} / {len(d['samples'])}")

# import os

# path = "/data/user/rollout_only_Qwen3-30B-A3B/"
# rollout_dir = os.path.dirname(path)

# total_count = 0
# total_samples = 0
# for fname in os.listdir(rollout_dir):
#     if not fname.endswith(".pt"):
#         continue
#     fpath = os.path.join(rollout_dir, fname)
#     try:
#         d = torch.load(fpath, weights_only=False)
#     except Exception as e:
#         print(f"skip {fname} due to error: {e}")
#         continue
#     cnt = sum(1 for s in d.get("samples", []) if s.get("reward", 0) > 0.8)
#     total_count += cnt
#     n_sample = len(d.get("samples", []))
#     total_samples += n_sample
#     print(f"{fname}: reward > 0.8 的序列数量: {cnt} / {n_sample}")

# print(f"\nALL: reward > 0.8 的序列数量: {total_count} / {total_samples}")