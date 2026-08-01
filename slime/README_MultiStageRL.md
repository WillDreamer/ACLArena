## Multi-stage RL

### 8B Version
1. Math Reasoning
```bash
bash examples/maths_reasoning/run-qwen3-8B-base.sh
bash examples/maths_reasoning/resume-qwen3-8B-base.sh
```
Model is saved at `/shared/user/MultiStageRL/Qwen3-8B-Base-Math`

2. Agentic Search

2.1 Launch Retriever
The database is under `/shared/user/Search_data`
```bash
cd Search-R1
conda activate retriever
bash Search-R1/retrieval_launch.sh
```

2.2 RFT
SFT data is saved at `/shared/user/MultiStageRL/rollout_only_search_Qwen3-32B`
```bash
# 1.
bash examples/search-r1/run_qwen3_8b_rollout_only.sh 

# 2. 
python examples/search-r1/filter_rollout_data.py \
    --input-dir /shared/user/MultiStageRL/rollout_only_search_Qwen3-8B \
    --output-dir /shared/user/MultiStageRL/rollout_only_search_Qwen3-8B/filtered \
    --min-reward 0.6 --batch-size 64

# 3. 
bash examples/search-r1/run_qwen3_8b_sft.sh
```

2.3 RL
```bash
bash examples/search-r1/run_qwen3_8b_seq_gspo_sft.sh
```


3. Agentic Custom Service

3.1 Serve a user model

3.2 RFT
SFT data is saved at `/shared/user/MultiStageRL/rollout_only_tau_Qwen3-32B`
```bash
bash examples/tau-bench/run_qwen3_8B_rollout_only.sh 

bash examples/tau-bench/run_qwen3_8B_sft.sh 
```
3.3 RL
```bash
bash examples/tau-bench/run_qwen3_8B.sh
```

### Qwen3.5-9B version
1. Run the docker
```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /shared/user:/shared/user \
  -it slimerl/slime:latest /bin/bash
```
Install the env
```bash
# 路径可根据实际情况调整
cd /root/slime
git pull
pip uninstall -y nvidia-cudnn-cu12 nvidia-cudnn-cu11 nvidia-cudnn 2>/dev/null || true
pip install -e . --no-deps
```

Quit and save the docker
```bash
# Find docker container ID
docker ps
docker commit xxx user/MSRL:latest
```
<!-- export  CUDA_VISIBLE_DEVICES=6,7
sglang serve \
  --model-path /shared/user/Qwen3.5/Qwen3.5-9B-Base \
  --dtype bfloat16 \
  --tp 1 \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code -->

2. VLM Reasoning

2.1 Download the data
```bash
hf download --repo-type dataset VeraIsHere/geo3k_imgurl_processed --local-dir /shared/user/geo3k_imgurl_processed
```

2.2


## OPD

1. Deploy the teacher on scai7
```bash
# --model-path /shared/user/MultiStageRL/Qwen3-8B-Base-Math/hf_iter_0000300 \
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path /shared/user/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search/hf_iter_0000420 \
  --host 0.0.0.0 --port 30012 \
  --tp 1 --mem-fraction-static 0.9 \
  --reasoning-parser qwen3 --tool-call-parser qwen
```

2. Related Work
- Sparse Logit Sampling (ACL 2025)
它证明了“缓存 top-K 概率再重归一化”是对老师分布的有偏估计（把概率质量人为集中到被选中的 token 上），会损害学生性能和 calibration。它的修正不是 top-k，而是按老师概率随机采样 token（importance sampling），得到无偏估计，开销 <10%。
→ 这正是我上一轮提醒你的“坑 B”的论文级佐证：你要的 top-20 重归一化，统计上是 biased 的。

- Entropy-Aware OPD / EOPD (2026) —— 真的用了 top-k 重归一化 KL，但有讲究。
它发现 reverse KL 在“老师高熵”的 token 上不稳定，会塌缩多样性（toy 实验里 top-1 来回跳 84 次 vs 低熵 7.3 次）。它的方案：L = L_reverseKL + 𝟙[H_teacher > τ]·L_forwardKL（τ=0.8），即默认 reverse KL，只在老师高熵 token 上额外加 forward KL；而那个 forward-KL 项就是在老师 top-k（k=16）上重归一化算的。
→ 这是“top-k 重归一化 KL”在 on-policy 蒸馏里的一个真实先例，但注意它用的是 teacher top-k（不是 intersection），且 k=16，且只在高熵处用 forward KL。

- EMA Policy Gradient (Lunjun Zhang & Jimmy Ba, 2026) —— top-k KL 直接放进 RL，和你的场景最像。它在 RL 里用 top-k KL（在 top-k 集合上重归一化）作为对 anchor/老师的 KL 约束，理由就是：全词表太贵、单 token 信号不够、top-k 是效率/稳定性的折中。

→ 说明“在 RL/advantage 框架里用 top-k KL”是被认可的折中。

- DistillM-2 (ICML 2025) —— 另一条路：对比式 logit 蒸馏。不是匹配老师分布，而是加权对比（WCLD）：抬高老师回答的似然、压低学生回答的似然，并强调“不同数据（on-policy/off-policy）要配不同 loss”。如果你最终目标是“把跑偏的学生拉回老师”，这条对比思路也值得看。