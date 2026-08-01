# ACLArena: Agent Continual Learning in Multi-stage Post-training

ACLArena studies what happens to an LLM agent when post-training is applied **in
stages, one task after another** — and how much of each earlier skill survives.
It is built on [slime](https://github.com/THUDM/slime) (Megatron-LM for training,
SGLang for rollout).

The setup has three parts:

1. **Four tasks**, each with its own RL environment and reward.
2. **Sequential (multi-stage) post-training** — train the tasks one after another
   and measure the forgetting this induces.
3. **Two remedies** for that forgetting — **SDFT** (oracle → distilled SFT
   mixture) and **on-policy distillation** (specialist teachers → the drifted
   end-of-chain student, including a multi-teacher variant).

## The four tasks

| Task | Short name | Where | Environment / reward |
| --- | --- | --- | --- |
| Math reasoning | `math` | `slime/examples/maths_reasoning/` | DAPO-Math-17k prompts, `--rm-type deepscaler`, AIME-2024 for eval |
| Multi-turn search | `search` | `slime/examples/search-r1/` | Search-R1 style retrieval loop against a local dense retriever, F1/EM reward |
| Agentic tool use | `tau` | `slime/examples/tau-bench/`, `slime/examples/tau-bench-async/` | Vendored tau-bench retail/airline env + LLM user simulator, task-success reward |
| Instruction following | `if` | see note below | IFBench verifiable constraints (`rm_type: ifbench`) |

The sequential chain trained in this work is

```
Qwen3-8B-Base -> Math -> (SeaSFT) -> Search -> (TauSFT) -> Tau -> IF
```

which is why checkpoints are named e.g.
`Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau-IF`.

> **Note on `if`.** This repository contains the IF stage's *evaluation* path
> (`slime/examples/eval_multi_task/` with `rm_type: ifbench`, implemented in
> `slime/slime/rollout/rm_hub/ifbench.py`) and everything that *consumes* the IF
> stage — it is the student in `on_policy_distillation/sea2if`, `tau2if` and
> `multi_teacher`, and one of the four domains in the SDFT mixture. A dedicated
> IF RL launch script is not included here; the IF stage reuses the standard
> slime GRPO path with the `ifbench` reward.

## Repository layout

| Path | Contents |
| --- | --- |
| `slime/` | Derivative of upstream slime with the training/rollout changes this work needs. |
| `slime/examples/maths_reasoning/` | Math RL (8B / 30B-A3B / 355B-A32B, LoRA variant, resume script). |
| `slime/examples/search-r1/` | Search RL, local dense retriever, rollout-only + SFT conversion tooling. |
| `slime/examples/tau-bench/` | tau-bench agent RL: vendored env, user simulator, trainable multi-turn agent. |
| `slime/examples/tau-bench-async/` | Asynchronous / self-play variant of the tau rollout. |
| `slime/examples/SDFT/` | Oracle-mixture SFT, including the domain-balanced batch sampler. |
| `slime/examples/on_policy_distillation/` | OPD / EOPD / GKD recipes, pairwise and `multi_teacher/`. |
| `slime/examples/eval_multi_task/` | One eval config covering AIME, GPQA and IFBench. |
| `lora_probe/` | Standalone LoRA weight-update diagnostics used while debugging the LoRA path. |

## 1. Sequential (multi-stage) post-training

Each stage is trained **in that task's own directory** — there is no separate
"sequential" driver. Run the stages in order, pointing each stage's `--load` at
the checkpoint the previous stage produced:

```bash
cd slime

# stage 1 — math
bash examples/maths_reasoning/run-qwen3-8B-base.sh

# stage 2 — search   (--load = the math checkpoint)
bash examples/search-r1/run_qwen3_8b_seq_search.sh

# stage 3 — tau      (--load = the search checkpoint)
bash examples/tau-bench/run_qwen3_8B.sh

# stage 4 — IF       (standard slime GRPO with --rm-type ifbench)
```

The `*_seq_*` scripts (e.g. `search-r1/run_qwen3_8b_seq_search.sh`,
`search-r1/run_qwen3_8b_seq_gspo_sft.sh`) are the sequential-stage variants:
same environment, but initialised from the previous stage instead of from base.
`resume-*.sh` / `*_resume.sh` scripts continue an interrupted run of the same
stage.

## 2. SDFT — oracle first, then distilled SFT

SDFT recovers earlier skills without replaying their RL environments. The
pipeline is per-task **oracle → trajectories → mixed SFT**:

**Step 1 — get each task's oracle.** Train the single-task specialist (the
"oracle") for each of the four tasks using its own directory from section 1.
These are the per-task upper bounds: `Qwen3-8B-Base-Math`,
`Qwen3-8B-Base-SeaSFT-Search`, `Qwen3-8B-Base-TauSFT-Tau`, and the IF stage.

**Step 2 — distill trajectories out of each oracle.** Run rollout-only against
the oracle, then filter by reward and convert to SFT records:

```bash
# tau example
bash examples/tau-bench/run_qwen3_30B_rollout_only.sh      # dumps rollout_{id}.pt
python examples/tau-bench/parse_rollout_to_sft.py \
    --input-dir   $DATA_ROOT/tau_sft_rollout_data \
    --output-file $DATA_ROOT/tau_sft_rollout_data/sft_data.jsonl \
    --reward-threshold 1.0

# search equivalents
bash examples/search-r1/run_qwen3_8b_rollout_only.sh
python examples/search-r1/filter_rollout_data.py ...
```

**Step 3 — SFT on the mixture.** Concatenate the four per-domain files into one
`messages`-format jsonl and train:

```bash
# globally shuffled mixture
bash examples/SDFT/run_qwen3_8b_oracle_mix_sft.sh

# domain-balanced: equal share per domain in EVERY batch
bash examples/SDFT/run_qwen3_8b_oracle_mix_sft_balance.sh
```

The balanced run uses `slime.rollout.balanced_data_source.BalancedRolloutDataSource`
and needs a top-level `data_source` field (`if|search|math|tau`) on each record.
Without it the batch is dominated by the large domains and the small one is
barely seen. Configure it with:

| Env var | Default | Meaning |
| --- | --- | --- |
| `BALANCED_DOMAINS` | `if,math,search,tau` | Domains to draw from, equally. |
| `BALANCED_DOMAIN_KEY` | `data_source` | Record field holding the domain tag. |
| `BALANCED_DROP_UNKNOWN` | `0` | Drop records whose tag is not listed. |

Both scripts use the SFT path `--rollout-function-path
slime.rollout.sft_rollout.generate_rollout` with `--loss-type sft_loss
--loss-mask-type qwen3 --calculate-per-token-loss`, so only assistant turns are
trained and the loss is averaged over all unmasked tokens in the batch.

## 3. On-policy distillation (including multi-teacher)

Lives under `slime/examples/on_policy_distillation/`. A **specialist teacher**
supervises the drifted student on the student's *own* rollouts, so no replay of
the teacher's environment is needed. Directories are named
`<teacher-domain>2<student-stage>`:

| Directory | Teacher | Student | Purpose |
| --- | --- | --- | --- |
| `math2sea/` | math specialist | search-stage model | restore math after the search stage |
| `sea2tau/` | search specialist | tau-stage model | restore search after the tau stage |
| `sea2if/` | search specialist | end-of-chain IF model | restore search at end of chain |
| `tau2if/` | tau specialist | end-of-chain IF model | restore tau at end of chain |
| `multi_teacher/` | math + search + tau, all three at once | end-of-chain IF model | restore everything in one run |

Each **pairwise** directory offers three variants:

- `run_8b_opd.sh` — reverse-KL OPD (mode-seeking).
- `run_8b_eopd.sh` — entropy-aware OPD: reverse-KL everywhere plus a forward-KL
  term on high-entropy teacher tokens.
- `run_8b_gkdopd.sh` — GKD-style objective.

(`*_resume.sh` / `_v2.sh` variants continue or revise a specific run.)

Key flags (documented in `on_policy_distillation/README.md`): `--use-opd`,
`--opd-type {sglang,megatron}`, `--opd-kl-coef`, `--opd-teacher-load`.

**Multi-teacher.** `multi_teacher/` serves all three specialists as *frozen*
SGLang models alongside the trainable actor. The student rolls out over a mixed
math+search+tau prompt stream and each trajectory is scored by the teacher that
owns its domain (routing in `opd_multi_teacher.py`). See
`multi_teacher/MOPD_IMPLEMENTATION.md` for the design and
`prepare_opd_mixed_data.py` / `prepare_opd_eval_data.py` for data prep.

## Environment setup

### Base stack (all tasks)

```bash
git clone https://github.com/WillDreamer/ACLArena.git
cd ACLArena/slime
pip install -e . --no-deps
```

You also need Megatron-LM and SGLang importable. The run scripts read these
roots from the environment and fall back to generic defaults, so set them for
your machine:

| Variable | Used for |
| --- | --- |
| `ROOT_DIR` | Working root; `${ROOT_DIR}/slime` and `${ROOT_DIR}/Megatron-LM` go on `PYTHONPATH`. |
| `MODEL_ROOT` | HF checkpoints and `*_torch_dist` Megatron checkpoints. |
| `DATA_ROOT` | Datasets. |
| `WANDB_API_KEY` | Optional; scripts pass `--use-wandb`. |

Convert every HF checkpoint to Megatron format once:

```bash
source scripts/models/qwen3-8B.sh
PYTHONPATH=$ROOT_DIR/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint $MODEL_ROOT/Qwen3-8B-Base \
    --save $MODEL_ROOT/Qwen3-8B-Base_torch_dist
```

Paths in the scripts are placeholders (`/data/user`, `/shared/user`,
`s3://YOUR_BUCKET`, `ACCOUNT_ID`) — replace them with your own.

### `math`

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir $DATA_ROOT/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir $DATA_ROOT/aime-2024
bash examples/maths_reasoning/run-qwen3-8B-base.sh
```

No external service is needed: the reward is rule-based (`--rm-type deepscaler`).
Pick a GPU set by editing `GPU_LIST` at the top of the script.

### `search`

Needs the Search-R1 QA data plus a **local dense retriever** served over HTTP.

```bash
git clone https://github.com/PeterGriffinJin/Search-R1.git
cd Search-R1 && pip install -e . --no-deps && pip install tensordict
python scripts/data_process/qa_search_train_merge.py \
    --local_dir $DATA_ROOT/nq_hotpotqa_train --data_sources nq,hotpotqa
```

The retriever needs the wiki-18 corpus, its FAISS index, and the `e5-base-v2`
encoder. Because GPU FAISS conflicts with the training env, install it into a
**separate conda env** (`examples/search-r1/README.md` appendix has the full
recipe). Then configure:

| Variable | Meaning |
| --- | --- |
| `SEARCH_ENABLE_RETRIEVER` | Start the retriever alongside training. |
| `RETRIEVER_CONDA_DIR` | Conda prefix holding the `retriever` env. |
| `RETRIEVER_USE_GPU` | GPU FAISS (recommended) vs CPU. |
| `WIKI_CORPUS`, `WIKI_INDEX`, `E5_MODEL` | Corpus, FAISS index and encoder paths. |
| `SEARCH_URL` | Point at an already-running retriever instead. |
| `RETRIEVER_WATCHDOG` | Restart the retriever if it dies mid-run. |

A hosted search backend can be used instead by filling the `api_key` field in
`SEARCH_R1_CONFIGS` inside `generate_with_search*.py`.

### `tau`

The tau-bench environment is vendored at
`slime/examples/tau-bench/tau_bench/` (retail + airline data included), so no
external clone is required. What you must choose is the **user simulator**:

| Variable | Meaning |
| --- | --- |
| `TAU_ENV` | `retail` or `airline`. |
| `TAU_TASK_SPLIT` | `train` / `dev` / `test`. |
| `TAU_USER_SIM_URL`, `TAU_USER_MODEL_ID` | `local` mode: any OpenAI-compatible endpoint (we serve a frozen GLM-4.7-Flash in-cluster as a second SGLang model). |
| `TAU_BEDROCK_REGION`, `TAU_USER_MODEL_ID` | `claude` mode: Bedrock via boto3, standard credential chain, no API key in code. |
| `TAU_USER_TEMP`, `TAU_USER_MAX_TOKENS`, `TAU_USER_NO_THINK` | Decoding for the simulator. |
| `TAU_ENV_WORKERS`, `TAU_ENV_THREAD_WORKERS` | Env concurrency; the env is synchronous and runs in a thread pool. |
| `TAU_USER_MAX_RETRIES`, `TAU_USER_HTTP_TIMEOUT`, `TAU_USER_RETRY_BASE/CAP` | Retry policy for simulator calls. |

The two backends are selected by `load_user(...)` in
`tau_bench/envs/user.py` (`local` / `claude`).

### `if`

IFBench scoring needs a few extra packages; the reward function prepares the
rest on first use.

```bash
pip install -r examples/eval_multi_task/requirements_ifbench.txt
hf download --repo-type dataset zyzshishui0627/IFBench --local-dir $DATA_ROOT/ifbench
```

Then point `examples/eval_multi_task/multi_task.yaml` at the jsonl and run with
`--eval-config`. The same file also configures AIME (`deepscaler`) and GPQA, so
one eval pass covers several stages of the chain.

## Launching

Every recipe is a self-contained shell script that starts Ray and submits a job,
so a single node is just:

```bash
cd slime && bash examples/<task>/<script>.sh
```

Multi-node runs read `MASTER_ADDR`, `ACTOR_NUM_NODES`, `ROLLOUT_NUM_GPUS` and
`USER_SIM_NUM_GPUS` from the environment; the internal job-submission wrapper we
used is **not** part of this repository, so supply those variables from your own
launcher (Ray, Slurm, or plain `bash`). Comments that mention a
`cluster_cli_*.py` submission command refer to that omitted wrapper.

## License

Apache-2.0, inherited from upstream slime. See `slime/LICENSE` and `NOTICE`.
