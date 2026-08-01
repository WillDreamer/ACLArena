# ACLArena

Multi-task RL / SFT training recipes for agentic LLMs, built on
[slime](https://github.com/THUDM/slime) (Megatron-LM training + SGLang rollout).

## Layout

| Path | Contents |
| --- | --- |
| `slime/` | Derivative of upstream slime, with the training/rollout changes this work relies on. |
| `slime/examples/SDFT/` | Multi-task SFT recipes, including the domain-balanced batch sampler. |
| `slime/examples/tau-bench/` | tau-bench agent rollout (vendored env, LLM user simulator, sync + async). |
| `slime/examples/search-r1/` | Search-R1 style multi-turn retrieval RL, with a local dense retriever. |
| `slime/examples/on_policy_distillation/` | On-policy / multi-teacher distillation recipes. |
| `slime/examples/proevolve/` | Executable-environment RL with DB-diff rewards. |
| `lora_probe/` | Standalone LoRA weight-update diagnostics used while debugging the LoRA path. |

## Environment

Install the training stack per the upstream slime instructions:

```bash
cd slime
pip install -e . --no-deps
```

Megatron-LM and SGLang must be importable; `PYTHONPATH` is set by the run
scripts under `slime/examples/`.

## Running

Each recipe is a self-contained shell script that submits a Ray job, for
example:

```bash
cd slime
bash examples/SDFT/run_qwen3_8b_oracle_mix_sft_balance.sh
```

The scripts read cluster-specific locations from environment variables
(`ROOT_DIR`, `MODEL_ROOT`, `DATA_ROOT`, `SFT_DATA`, ...) and fall back to
generic defaults. Model, dataset, and object-storage locations are
placeholders (`s3://YOUR_BUCKET`, `ACCOUNT_ID`, `/data/user`) — set them for
your own environment before running. The job-submission wrapper used for our
internal scheduler is not part of this repository; submit the scripts with
your own launcher (Ray, Slurm, or plain `bash` on a single node).

## License

Apache-2.0, inherited from upstream slime. See `slime/LICENSE` and `NOTICE`.
