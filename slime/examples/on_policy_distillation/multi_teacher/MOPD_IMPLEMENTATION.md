# 多教师在线策略蒸馏 (Multi-Teacher On-Policy Distillation, MOPD) — 实现说明

> 目标读者:接手这个 run 的人。本文描述 **当前实际实现**(而非设计草案),含每个文件的职责、
> 域路由机制、Cluster 上的部署方式、以及第一次运行踩到的坑。
>
> 最后更新:2026-07-21 · Qwen3-8B · Cluster BatchService 3-node disaggregated

---

## 1. 一句话概括

一个 **Qwen3-8B student** 在 **math + search + tau 三域混合** 的 prompt 流上做 rollout,每条轨迹由
**拥有该域的那个 teacher** 打分,学习信号 **只有** 逐 token 的反向 KL(student → teacher),
**task reward 恒为 0**(纯蒸馏,不是 RL)。三个 teacher 是 math2sea / sea2tau / tau2if 实验里的
各阶段专家;把它们的能力同时蒸回已经漂移的 end-of-chain IF student,一次性恢复 math + search + tau。

```
advantage -= opd_kl_coef * (student_log_prob - teacher_log_prob)   # 每 token,slime loss.py
```

- **student**:`ACLArena/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau-IF_torch_dist`(megatron torch_dist,`--load` 初始化点)
- **teachers**(均 in-cluster frozen 服务):
  - `teacher_math`   ← `ACLArena/Qwen3-8B-Base-Math`
  - `teacher_search` ← `ACLArena/Qwen3-8B-Base-SeaSFT-Search`
  - `teacher_tau`    ← `ACLArena/Qwen3-8B-Base-TauSFT-Tau`

---

## 2. 核心机制:静态域路由(没有分类器)

路由完全靠数据集里每行写死的字符串 **`sample.metadata["domain"] ∈ {math, search, tau}`**。
它在 **两个地方** 驱动分派:

| 阶段 | 文件 / 配置项 | 按 domain 做什么 |
|---|---|---|
| **rollout** | `generate_mixed_opd.generate`(`--custom-generate-function-path`) | 选用哪个原生 rollout 函数生成轨迹 |
| **打分** | `opd_multi_teacher.reward_func`(`--custom-rm-path`) | 选用哪个 teacher sglang 模型给轨迹打 logprob |

没有 prompt 判别器 / 分类器 —— domain 在 **离线建数据集时** 就 tag 死了(见 §3)。

---

## 3. 数据管道 — `prepare_opd_mixed_data.py`

**问题**:三域的 prompt 约定互相冲突,而一个训练 job 只有一个全局 `--apply-chat-template` 开关:

| 域 | 原始 input | 约定 |
|---|---|---|
| math   | chat-list | `--input-key prompt` + `--apply-chat-template`(模板化文本) |
| search | chat-list | `--input-key prompt` + `--apply-chat-template`(模板化文本) |
| tau    | 任务索引  | `--input-key index` + **无** chat template(`sample.prompt` = 任务号字符串) |

**解法(离线消解冲突)**:
1. math / search 用 **slime 自己的 `Dataset`(`apply_chat_template=True`)** 预先把模板烘进 `sample.prompt`
   —— 与训练期模板化 **逐字节一致**。
2. tau 保留任务索引原文(tau 的 generate 里做 `int(sample.prompt)`)。
3. 每行 tag `metadata.domain`,合并成 **一个 JSONL**。
4. run script 用 `--input-key prompt` + **不加** `--apply-chat-template` 加载它。

**域均衡(`--per-domain-target N`,默认 3000)**:原始池大小悬殊(dapo-math ~17k / nq_hotpotqa ~170k /
retail ~500)。不均衡的话一个 rollout batch 几乎全是 search,tau/math teacher 基本不触发。所以每个域
resize 到恰好 N 行(大池截断,小池 cycle-repeat)→ batch 里约 **1:1:1**,三个 teacher 每步都活跃。
tau prompt 被重复无害:student 每次采新轨迹,只是复用 prompt。

**产物**:`${MODEL_ROOT}/ACLArena/opd_mixed3/train.jsonl`(默认 3000×3 = 9000 行),
run script 首次运行时若不存在则自动生成。

---

## 4. Rollout 分派 — `generate_mixed_opd.py`

slime 只允许一个 `--custom-generate-function-path`。这个 dispatcher 按 domain 转发到各域的 **原生** rollout:

```python
async def generate(args, sample, sampling_params):
    domain = (sample.metadata or {}).get("domain")
    if domain == "math":    return await _generate_math(args, sample, sampling_params)   # slime 自带单轮
    if domain == "search":  return await _generate_search(args, sample, sampling_params) # search-r1 工具 rollout
    if domain == "tau":
        sample = await _generate_tau(args, sample, sampling_params)                       # tau 环境 rollout
        sample.reward = None   # 关键:清掉 tau 的 task reward
        return sample
    raise ValueError(...)
```

- `math`   → `slime.rollout.sglang_rollout.generate`(STOCK 单轮,slime 内置,无需 example 目录)
- `search` → `generate_with_search_tools_qwen_sft_no_drift.generate`
  (student 轮走 sglang router;search 工具走 CPU-faiss 检索器,URL 由 `SEARCH_R1_SEARCH_URL` 提供)
- `tau`    → `generate_with_tau.generate`(student 轮走 router;user 轮走 in-cluster GLM user-sim)

**`reward=None` 不变量**(极其重要,否则训坏):OPD teacher RM 只在 `sample.reward is None` 时被
`generate_and_rm` 调用。math / search 的 rollout 本就不设 reward,天然 OK;**tau 的 rollout 总会把
`sample.reward` 设成 tau 任务 reward(float)**,若不清掉会 (a) 抑制 teacher RM (b) 把 float 塞进
`post_process` 里期望 teacher logprob JSON 的位置。所以 tau 分支显式 `sample.reward = None`。

`examples/search-r1` 和 `examples/tau-bench` 必须在 PYTHONPATH 上(run script 已加)。

---

## 5. Teacher 打分 / reward — `opd_multi_teacher.py`

**域 → teacher 模型名** 的映射:

```python
_DOMAIN_TO_MODEL = {"math": "teacher_math", "search": "teacher_search", "tau": "teacher_tau"}
```

**`reward_func(args, sample)`**:
1. 由 `sample.metadata["domain"]` 取 teacher 模型名。
2. **安全断言**:`get_model_url` 找不到模型名时会 **静默回退到默认 (actor) router** —— 那就变成
   "用 student 给自己打分"。所以先 `assert model_name in args.sglang_model_routers`,配置漏了 teacher
   会显式报错,而不是悄悄训坏。
3. `url = get_model_url(args, model_name, "/generate")`(按名解析出该 teacher 自己的 sglang router,
   **无硬编码 IP** —— 这是能在无独立服务节点的 Cluster batch 上跑的关键)。
4. POST student 的 **原始 token ids**,`max_new_tokens=0, temperature=0, return_logprob=True,
   logprob_start_len=0`(只打分不生成)。

**`post_process_rewards(args, samples)`**:
- 从 sglang 返回的 `meta_info["input_token_logprobs"]` 取逐 token logprob(跳过第 1 个 —— 无前文无 logprob),
  取 **response span 的最后 `response_length` 个**(与 `sample.loss_mask` 1:1 对齐),写入 `sample.teacher_log_probs`。
- 被 mask 掉的 token(工具/observation 轮)也带 teacher logprob,但 policy loss 乘 loss_mask 后不贡献梯度。
- 返回 **标量 reward = 0.0**(纯 OPD)。

**打分是 domain-routing(只由所属 teacher 打),不是三 teacher 平均** —— math teacher 给 tau 轨迹打分是无意义的。

> ⚠️ 注意:`opd_multi_teacher.py` docstring 里的 teacher 权重名(如 `...Math-SeaSFT-Search`)是旧注释,
> 真实 S3 名是 `Qwen3-8B-Base-SeaSFT-Search` / `Qwen3-8B-Base-TauSFT-Tau`。路由靠的是 sglang-config
> 里的 **逻辑名** `teacher_search`/`teacher_tau`,与 run script 生成的 YAML 一致,不影响运行。

---

## 6. 在集群内服务这些模型 — 生成的 `--sglang-config`

run script 用 heredoc 生成一个 **5-model YAML**(`opd_multi_sglang.generated.yaml`)。每个 named model
拿到 **自己的 sglang router**;custom fn 用 `get_model_url(args, name, endpoint)` 按名解析。只有
`update_weights:true` 的 `actor` 收 student 权重同步;其余 4 个 frozen 模型被 slime 自动排除在权重同步外。

| model | update_weights | num_gpus | 说明 |
|---|---|---|---|
| `actor`         | **true**  | 5 | student rollout(TP=1,5 DP engine) |
| `user_sim`      | false | 8 | GLM-4.7-Flash(tau user-sim;`tool_call_parser: glm47`,`reasoning_parser: glm45`) |
| `teacher_math`  | false | 1 | 8B TP1 scoring-only,`context_length: 16384` |
| `teacher_search`| false | 1 | 同上 |
| `teacher_tau`   | false | 1 | 同上 |

**硬约束**:`sum(num_gpus) == ROLLOUT_NUM_GPUS`(slime `rollout.py` 的 assert)。
5 + 8 + 1 + 1 + 1 = **16** = ROLLOUT_NUM_GPUS。✓

run script 从 `USER_SIM_NUM_GPUS` 推导 actor 份额:
`ACTOR_ROLLOUT_GPUS = ROLLOUT_NUM_GPUS - USER_SIM_NUM_GPUS - 3(teachers) = 16 - 8 - 3 = 5`(有 `>0` 守卫)。

---

## 7. Cluster 拓扑(3 节点,disaggregated)

submit 参数:`--num-nodes 3 --rollout-nodes 2 --user-sim-nodes 1`
→ bootstrap 导出 `ROLLOUT_NUM_GPUS=16`,`USER_SIM_NUM_GPUS=8`,`ACTOR_NUM_NODES=1`。

- **Node 0(train,8 GPU)**:student megatron **TP4 × PP1 × CP2 = 8**。**同时在主节点跑 CPU-faiss 检索器**
  (绑 `0.0.0.0:8000`,仅内存;P5EN ~2TB RAM 够放 79GB 索引)。rollout 节点通过 `MASTER_ADDR:8000` 访问。
- **Nodes 1-2(rollout 池,16 GPU)**:上表 5 个 sglang 模型,`num_gpus` 之和 = 16。

**为什么必须 disaggregated**:frozen teachers + GLM 要活在 rollout GPU 池里;`--colocate` 会把 rollout
锁死在 actor GPU 上,放不下这些额外模型(sdft CLI 甚至硬报错 "`--user-sim-nodes` requires `--rollout-nodes > 0`")。
同步 `train.py` 支持 disaggregated(`--rollout-num-gpus`)+ `--sglang-config`,不需要 async。

**跨节点组网**:所有节点组一个 Ray cluster,slime placement group 按 node-IP 排序切分 actor / rollout。
⚠️ 若 3 节点跨不同 /24 子网,actor→rollout 首次权重同步的 TCP rendezvous 可能 hang(见 §11)。

---

## 8. 提交 CLI — `cluster_cli_mopd.py`(专用,与 SDFT 隔离)

`/workspace/ACLArena/cluster_cli_mopd.py` 是 `cluster_cli_sdft.py` 的独立 sibling(同 `sdft_balance` 惯例:
整份 copy,只改 DEFAULTS + docstring)。**存在的理由是避免和并发的 SDFT job 冲突**,隔离了三处资源:

| 资源 | SDFT | MOPD |
|---|---|---|
| `code_s3`(`s3 sync --delete`) | `code/slime-sdft` | **`code/slime-mopd`** —— 否则两者互删对方 staged 代码 + `_bootstrap/` |
| `job_prefix` | `user-sdft-` | **`user-mopd-`** |
| `output_prefix`(每 300s 后台回传 `$MODEL_ROOT/<prefix>`) | `ACLArena/` | **`ACLArena/OPD/`** —— MOPD 的 teacher/student 权重就 stage 在 `ACLArena/` 下,用宽前缀会每 5 分钟重传 ~100GB 冻结权重 + 抢 staging 带宽;收窄到 run script 真正 `--save` 的 `ACLArena/OPD/` |

**base-model 路径重映射**:S3 里 base 在 `Qwen3/Qwen3-8B-Base/`,但 run script 找的是裸路径
`$MODEL_ROOT/Qwen3-8B-Base`。所以 stage 条目支持 `SRC=>DEST` 形式(`_split_remap` in `_build_bootstrap`):
`"Qwen3/Qwen3-8B-Base/=>Qwen3-8B-Base/"`。

**bootstrap 做的事**(每个节点跑同一份):配跨账号凭证(assume `CLUSTER_DEV_ROLE` @ 339 访问 YOUR_BUCKET)
→ `s3 sync` 代码到 `/root/slime` → 下模型/数据到 NVMe(`/tmp/instance_storage/user`)→ 后台每 300s 把
`$MODEL_ROOT/ACLArena/OPD/` sync 回 S3(+EXIT 再一次)→ 按 `AWS_BATCH_JOB_NODE_INDEX` 分叉主/worker 节点
(worker `ray start ... --block` 不跑 train.py;主节点 export `MASTER_ADDR/ACTOR_NUM_NODES/ROLLOUT_NUM_GPUS/
USER_SIM_NUM_GPUS` 再 `bash <run script>`)。

---

## 9. 预置数据(已 staged 到 s3://YOUR_BUCKET,2026-07-21 验证)

| 类型 | S3 位置 | 大小 | 备注 |
|---|---|---|---|
| student(torch_dist) | `ACLArena/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau-IF_torch_dist/` | — | `--load` 初始化 |
| teacher_math | `ACLArena/Qwen3-8B-Base-Math/` | — | flat HF |
| teacher_search | `ACLArena/Qwen3-8B-Base-SeaSFT-Search/` | — | flat HF |
| teacher_tau | `ACLArena/Qwen3-8B-Base-TauSFT-Tau/` | — | flat HF |
| base(tokenizer/config + actor 初始化) | `Qwen3/Qwen3-8B-Base/` | — | 重映射到裸 `Qwen3-8B-Base` |
| user-sim | `GLM/GLM-4.7-Flash/` | ~62GB | — |
| search 检索器编码器 | `e5-base-v2/` | 0.88GB | model_type=bert/h768 |
| math 数据 | `data/dapo-math-17k/dapo-math-17k.jsonl` | 10MB | — |
| tau 数据 | `data/tau-bench/retail_*_tasks.jsonl` | — | run script 也能用 `tau1_mock.py` 现生成 |
| search 数据 | `data/nq_hotpotqa_train/train.parquet` | 0.356GB | 169,615 行,Search-R1 schema |
| **wiki-18 索引** | `data/wiki-18/e5_Flat.index` | **64.56GB** | part_aa+part_ab concat |
| **wiki-18 语料** | `data/wiki-18/wiki-18.jsonl` | **14.39GB** | 21,015,324 行,gunzip |

**staging 方法**(可复用):在 GPU sdb 上配 `credential_source=EcsContainer` + assume `CLUSTER_DEV_ROLE`
(自动刷新,无过期)→ HF Xet 下载 → concat/gunzip → boto3 multipart 上传 → head 校验。

---

## 10. 训练超参 / 步数(可用 `--env KEY=VALUE` 覆盖)

| 参数 | 默认 | 含义 |
|---|---|---|
| `--num-rollout` (`NUM_ROLLOUT`) | **300** | 训练步数 = 300 rollout→train 迭代 |
| `--rollout-batch-size` | 32 | 每步 32 prompt |
| `--n-samples-per-prompt` | 8 | 每 prompt 8 条轨迹(reward=0 → grouping 中性,可降以省 rollout 成本) |
| `--global-batch-size` | 256 | 32×8,每步 256 条轨迹 → 1 次梯度更新 |
| `--per-domain-target` (`PER_DOMAIN_TARGET`) | 3000 | 每域 3000 行 → 混合集 9000 prompt(约 1 epoch+) |
| `--opd-kl-coef` (`OPD_KL_COEF`) | 0.5 | OPD KL 系数(唯一学习信号) |
| `--kl-loss-coef` | 0.00 | 无额外 KL(ref 因此惰性) |
| `--lr` (`LR`) | 1e-6 | constant |
| `--rollout-max-response-len` | 8192 | max(math 8192, search 4096, tau 2048) |
| `--save-interval` | 20 | 每 20 步存 checkpoint → 回传 `ACLArena/OPD/` |
| `--max-tokens-per-gpu` | 12288 | dynamic batch(≥ 最长单序列 / CP) |
| `--advantage-estimator` | grpo | OPD 叠在 GRPO 上 |

**其它 run 时环境变量**:`TAU_USER_STRATEGY=local` / `TAU_USER_MAX_TOKENS=1024` / `TAU_ENABLE_THINKING=0` /
`TAU_STRIP_HISTORICAL_THINK=1`(tau user-sim);`SEARCH_R1_SEARCH_URL=http://$MASTER_ADDR:8000/retrieve` /
`SEARCH_R1_CONCURRENCY=64`(search);`OPD_TEACHER_TIMEOUT=600`(teacher prefill 慢,给足时间)。

---

## 11. 已知问题 / 已应用修复

### ✅ 已修:检索器缺 faiss(第一次运行的失败原因)
**job `user-mopd-run-qwen3-8B-opd-multi-599733`(2026-07-21)FAILED**:上游全对(staging、base 重映射、
wiki-18 79GB、sglang-config、disaggregated 拓扑、代码隔离都工作),但 CPU-faiss 检索器启动即崩:
```
ModuleNotFoundError: No module named 'faiss'
→ run script 就绪轮询: FATAL: retriever process died → exit 1  (启动 ~30s,还没到 Ray/训练)
```
镜像自带 torch/numpy/transformers/datasets/tqdm,但 **缺 faiss/fastapi/uvicorn/pydantic**
(`retrieval_server.py:23-31`)。

**修复(已应用,bash 语法已校验)**:run script 在检索器启动前、**仅主节点**做带守卫的运行时
`pip install -q faiss-cpu fastapi uvicorn pydantic`(只有主节点跑检索器,rollout 节点走 HTTP),
沿用 bootstrap 已用的 `pip install awscli` 模式。装完再 `import` 验证,失败即显式 FATAL。
开关:`OPD_RETRIEVER_PIP=0`(默认 1)。

### ✅ 已修:faiss-cpu 在 96-vCPU 节点 OpenBLAS 溢出 SIGSEGV(第二次运行的失败原因)
**job `user-mopd-run-qwen3-8B-opd-multi-501535`(2026-07-21)FAILED**:faiss `pip install` 生效了
(`deps OK: faiss 1.14.3`),检索器加载 64GB 索引约 60s 后 **段错误 (core dumped)**,日志刷屏
`BLAS : Program is Terminated. Because you tried to allocate too many memory regions.`。根因:OpenBLAS
有编译期的并发内存区上限(~NUM_THREADS×2),在 96 核上不加限制会启太多 BLAS 线程 → 溢出 abort → 崩掉 faiss。
**修复(已应用)**:给检索器进程设 `OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS/
NUMEXPR_NUM_THREADS/VECLIB_MAXIMUM_THREADS = RETRIEVER_NUM_THREADS(默认 32)`。flat 索引 + 小 e5 BERT
编码器不需要 96 线程,且它与 8-GPU student 共享节点,本就不该占满核。

### ✅ 已修:检索器阻塞在 `ray start --head` 之前 → worker 超时(同 501535,潜在第二 bug)
501535 里 worker 节点(a7f24165)启动后 4+ 分钟探不到 head `10.2.92.82:6379`。根因不(仅)是跨子网:
run script 原本 **先同步等检索器加载 64GB 索引 + 14GB 语料(可达 1200s),再 `ray start --head`** —— 期间
head 的 GCS :6379 根本不存在,而 worker 的 join 预算只有 **600s**(bootstrap §5b),会先超时退出。这次因为
BLAS 段错误 ~60s 就死了,没暴露到这一步,但 BLAS 修好后完整索引加载必然撞上。
**修复(已应用,重排顺序)**:(1) 先 `ray start --head` → (2) 后台起检索器,让它在 30 分钟 GPU 注册等待
**期间** 并行加载(免费)→ (3) 在 `ray job submit` 前才 gate 检索器就绪。worker join 不再被索引加载阻塞。

### ✅ 已修:sglang 模型跨节点 straddle → TCPStore rendezvous 超时(第三次运行失败原因)
**job `user-mopd-run-qwen3-8B-opd-multi-244004`(2026-07-21)FAILED**,但跑到了目前最远:24/24 GPU、
3 节点全 join、检索器起来(BLAS 修复生效)、`ray job submit` 成功、sglang 引擎 fired up。然后 rollout-node-2
(10.2.243.179)上的一个 user_sim 引擎崩:`torch.distributed.DistNetworkError: client socket timed out
after 600000ms connecting to (10.2.222.246, 15296)`。根因:slime 按 YAML `config.models` 顺序用累计
gpu_offset 分配 rollout GPU(rollout.py:1103-1154),而每引擎的 dist_init_addr 按
`node_index = local_rank // engines_per_node`(rollout.py:936)分组——**假设每个 model group 从节点边界开始**。
旧顺序 [actor=5, user_sim=8, teachers=3] 在 2×8-GPU rollout 节点上,user_sim 从 GPU offset 5(节点中间)开始、
**横跨两个节点**,slime 把它所有引擎都指向 node-0 的 IP → 物理在 node-1 上的 5 个 user_sim 引擎试图跨子网
TCPStore rendezvous 到 node-0 → 600s 超时。**注意这与 §跨子网权重同步 hang 是不同机制**(那个是 actor→rollout
权重同步;这个是 user_sim 模型内部引擎 rendezvous)。**修复(已应用)**:重排生成的 YAML,让单 GPU 模型排前面
= [actor(5)+teacher_math+teacher_search+teacher_tau = 8 = 恰好 node 0],user_sim(8) = node 1。每个 group
节点对齐,无横跨。+ 加了 NODE-ALIGNMENT GUARD 断言(actor+3==NUM_GPUS;USER_SIM_NUM_GPUS % NUM_GPUS==0)让坏的
节点/GPU 切分显式报错而非 hang。**依赖 2 rollout 节点 + actor=5+3teachers=8 的布局;节点/GPU 数变了断言会拦下。**

### ✅ 已修(真定论):partition_stride 断言在 TP=1 下也误触发 → slime-core 短路补丁 + actor TP=1
第 9 次(job <JOB_ID>,TP=1)**仍然**崩在同一个 `partition_stride != 1`。原因:`partition_stride` 是 Megatron 在
**层构造时**给融合 QKV 设的(stride=3),**与运行时 TP world size 无关**,所以任何 TP 值都躲不过——断言只是触发得
太早。**关键领悟**:tp_size==1 时 `all_gather_param` 是个 no-op(param_partitions=[param.data],concat/GLU-rechunk
往返回到原张量,partition_stride 根本没被用来 reorder 任何东西),断言在 TP=1 下是虚假的。**修复**(改
`slime/backends/megatron_utils/update_weight/common.py`,随 code-sync 上传,**不用重建镜像**):在 `all_gather_param`
(sync)和 `all_gather_params_async`(bucket)两处、断言之前加 `if tp_size == 1: return param.data`。与 tp>1 路径
对单一分片算出的结果完全一致。配合 actor TP=1(见上),构成完整修复。(将来若要在 disaggregated 路径上用 TP>1,需要
真正的 de-interleave 补丁——暂缓。)

### (作废)以为 TP=1 就够 / 任何 TP>1 必崩
第 8 次(job <JOB_ID>,TP=2)转换了 1 个参数后仍崩在 QKV 的 `partition_stride != 1`,证明**不是 TP=4 特有,而是任何
TP>1 都会**:disaggregated 路径的 `all_gather_param`(common.py:37)对带 stride 的 GQA 融合 QKV(num_query_groups=8)
没有 de-interleave 逻辑(只有 colocate 路径的 `convert_qwen2_to_hf` 用 view/split 拆,而我们放不下 3 teacher+GLM
不能 colocate)。权衡三方案:(A) TP=1【零风险:无 TP 分片→partition_stride 恒 1】(B) 打补丁 de-interleave【可能静默
喂错权重】(C) colocate【放不下 teacher,不可行】。**用户选 A**。修复:actor TP=4→**TP=1**,去掉 --sequence-parallel
(需 TP>1),保留 CP=2 → train 节点 8 GPU = TP1×CP2×PP1→DP=4。8B 在 TP=1 下 H200(141GB)放得下。rollout 不受影响。

### (作废)TP=4/TP=2 在 disaggregated 路径上的 partition_stride 断言
**先诊断错了(以为缺 `--sequence-parallel`),第 7 次(job <JOB_ID>)同样断言再次证明不对。真正根因**:slime 的权重同步
按 colocate 分两条路径——`UpdateWeightFromTensor`(colocate,走 convert_to_hf,**无** stride 断言)vs
`UpdateWeightFromDistributed`(disaggregated,走 all_gather_param,**硬断言 partition_stride==1**)。tau-bench 的
TP=4 能跑**只因为它是 --colocate**(走另一条路);MOPD 是 disaggregated(为了 frozen teachers 必须),所以命中带断言那条。
TP=4 时 Qwen3-8B 的 GQA 融合 QKV(num_query_groups=8)得到 partition_stride>1 → 断言崩。**修复**:TP=4→**TP=2**
(train 节点 8 GPU:TP2×CP2×PP1→DP=2),对齐**已验证的 disaggregated 模板**
`examples/on_policy_distillation/run-qwen3-8B-opd-megatron.sh`(TP=2 + --sequence-parallel,同一条
UpdateWeightFromDistributed 路径,权重同步正常)。CP=2 不影响 striding(切激活不切参数)。所有 disaggregated
slime 例子都用 TP≤2,只有 colocate 才用 TP=4。

### (作废)TP=4 缺 --sequence-parallel → partition_stride 断言(第六次的错误归因)
**job `user-mopd-run-qwen3-8B-opd-multi-743113`(2026-07-21)FAILED**,但跑过了此前所有关卡:NVMe/staging、
24/24 GPU、检索器就绪、模型节点对齐加载、actor↔rollout 握手、wandb 上线、5 个 sglang 模型全 fired up,
**并且跨子网 actor→rollout 权重同步组也成功建立**(`init custom process group ... world_size=6, backend=nccl`
+ 跨节点 `POST /update_weights_from_distributed 200 OK`)——所以下面那个"跨子网权重同步 hang"风险**没有触发**
(EFA/RDMA 扛住了)。真正崩在 Megatron→SGLang 的**参数转换**:`ray::MegatronTrainRayActor.update_weights()
→ AssertionError: partition_stride != 1 is not supported`(common.py:37)。根因:TP=4 且缺 `--sequence-parallel`
时,Qwen3-8B 的 GQA 融合 QKV + qk-layernorm 参数得到**带 stride 的 TP 分片**(partition_stride>1),
slime 的转换器不支持。**修复(已应用)**:PERF_ARGS 加 `--sequence-parallel` —— 已验证能跑的
`examples/tau-bench/run_qwen3_8B.sh`(RL,做权重同步)用的是**完全相同**的 qwen3-8B MODEL_ARGS + TP=4/CP=2
且带 `--sequence-parallel`,所以它权重同步正常。

### ✅ 已验证(不再是风险):跨子网 actor→rollout 权重同步
第六次运行实证:3 节点跨子网(train 10.2.0.52 / rollout 10.2.138.38 / head 10.2.149.195),权重同步组照常
建立、`update_weights_from_distributed` 全 200 OK。[[slime-disagg-weightsync-crosssubnet-hang]] 那个 hang
在本 setup(EFA 镜像 392e9980)未复现。
第一次 3 节点跨不同 /24 子网(10.2.253/173/83),但因死在检索器阶段没跑到 Ray/权重同步,所以
[[slime-disagg-weightsync-crosssubnet-hang]] 的跨子网 rendezvous hang 风险 **本次未验证**。重投后需盯着:
actor→rollout 首次权重同步若长时间无训练日志,即命中此问题(修复方向:显式短 init_process_group timeout
让它快速报错,或强制同子网调度)。

### 其它待观察风险
- GLM 的 `glm47`/`glm45` sglang parser 是否在镜像里可用。
- CPU-faiss 在 rollout concurrency 下的吞吐(可能成为 search rollout 瓶颈;`SEARCH_R1_CONCURRENCY` 可调)。
- GPU-idle watchdog 是否杀掉同步 train 的交替 train/rollout 池(fallback = `train_async.py` + `--update-weights-interval`)。
- Qwen3-8B 的 tau tool-parse 兼容性(ACLArena 那套是 Qwen3.5;此处保留了 ACLArena 的 Qwen3-8B `trainable_agents.py`)。

---

## 12. 提交 / 监控

**提交**(stage 参数都已内置为默认,可省略):
```bash
cd /workspace/ACLArena
python3 cluster_cli_mopd.py batch \
  --script examples/on_policy_distillation/multi_teacher/run-qwen3-8B-opd-multi.sh \
  --num-nodes 3 --rollout-nodes 2 --user-sim-nodes 1
```
先 `--dry-run` 可打印 job def + bootstrap(注意 dry-run 仍会上传代码/bootstrap 到 `code/slime-mopd`,但不提交)。

**读 BatchService 日志**(CloudWatch,账号 821):
```bash
<cred-tool> credentials update --account ACCOUNT_ID --role CONSOLE_ACCESS_ROLE \
  --provider <provider> --profile cluster-console --once
# 列 stream(每节点一个):
aws logs describe-log-streams --log-group-name /aws/batch/job \
  --log-stream-name-prefix "<job-name>" --region ap-south-1 --profile cluster-console
# 取事件(get-log-events 常返 0 条的 token 坑;用 filter-log-events + 时间窗 + max-items 落文件再 grep):
aws logs filter-log-events --log-group-name /aws/batch/job \
  --log-stream-names "<stream>" --start-time <submit_ms> --max-items 4000 \
  --region ap-south-1 --profile cluster-console --output json > /tmp/log.json
```
> 坑:CLI 的 `status`/`describe` 只从旧到新翻 5000 个 job,今天的 job 可能翻不到 —— 直接分页
> `listjobs` 按 JobId/JobName 找(YOUR_INITIATIVE_ID 初始化有 >5000 个 job)。

**wandb**:上传到 **公有云 api.wandb.ai**,project `MOPD`,group `mopd_Qwen3-8B_math_search_tau`
(可用 `--env WANDB_GROUP=...` 覆盖)。run script 的 `WANDB_ARGS`(`--use-wandb --wandb-project MOPD ...`)
已接进 `ray job submit`;key 由 bootstrap 注入的 `WANDB_API_KEY` 提供。**预期**:OPD-KL loss 非零;
reward 恒为 0(纯 OPD 正常);首个 rollout batch 三域都出轨迹(math 自足 / search 命中检索器 / tau 到达
GLM user-sim),`sample.teacher_log_probs` 按域填充,无 align 失败。

---

## 13. 文件清单

| 文件 | 职责 |
|---|---|
| `run-qwen3-8B-opd-multi.sh` | 主 run script:生成 sglang-config、起检索器、Ray、`ray job submit train.py` |
| `generate_mixed_opd.py` | rollout dispatcher(按 domain 转发到 math/search/tau 原生 rollout) |
| `opd_multi_teacher.py` | teacher RM:域路由打分 + `post_process` 写 teacher_log_probs,task reward=0 |
| `prepare_opd_mixed_data.py` | 离线建混合数据集(预模板化 math/search + tau 索引 + domain tag + 均衡) |
| `opd_multi_sglang.generated.yaml` | run script 自动生成的 5-model sglang-config(勿手改) |
| `/workspace/ACLArena/cluster_cli_mopd.py` | 专用提交 CLI(与 SDFT 隔离 code_s3/job_prefix/output_prefix) |
| 依赖(PYTHONPATH):`examples/search-r1/` | search rollout + `local_dense_retriever/retrieval_server.py` |
| 依赖(PYTHONPATH):`examples/tau-bench/` | tau rollout + vendored `tau_bench` + GLM local user-sim |
