"""Build the mixed Math + Search + Tau prompt dataset for 3-teacher OPD.

The three domains have INCOMPATIBLE prompt conventions:
  * math:   --input-key prompt + --apply-chat-template  (sample.prompt = templated text)
  * search: --input-key prompt + --apply-chat-template  (sample.prompt = templated text)
  * tau:    --input-key index   + NO chat template       (sample.prompt = task index)

A single training job has only one global --apply-chat-template flag, so we resolve
the conflict OFFLINE: pre-apply the chat template to the math + search prompts here
(using slime's own Dataset, so it is byte-identical to training-time templating), keep
the tau index verbatim, tag every row with metadata.domain, and emit one merged JSONL.
The run script then loads it with --input-key prompt and NO --apply-chat-template.

Domain balancing (--per-domain-target N): the raw pools are wildly different sizes
(dapo-math ~17k, nq_hotpotqa ~170k, retail ~500). Left as-is a random rollout batch is
~all search and the tau (and even math) teacher barely fires. If --per-domain-target is
set, each domain is resized to exactly N rows (truncate large pools; cycle-repeat small
pools) so batches carry ~1:1:1 math:search:tau and all three teachers stay active every
step. Repeating tau prompts is harmless for OPD — the student samples fresh trajectories
each time; only the prompt is reused.

Usage:
    python prepare_opd_mixed_data.py \
        --hf /shared/user/Qwen3-8B-Base \
        --math-jsonl     /shared/user/dapo-math-17k/dapo-math-17k.jsonl \
        --search-parquet /data/user/Search-R1/data/nq_hotpotqa_train/train.parquet \
        --tau-jsonl      /data/user/tau-bench/retail_train_tasks.jsonl \
        --per-domain-target 3000 \
        --out /shared/user/MultiStageRL/opd_mixed3/train.jsonl
"""

import argparse
import json
import os

from transformers import AutoTokenizer

from slime.utils.data import Dataset


def _to_jsonable(x):
    """parquet metadata can contain numpy types; coerce to plain JSON."""
    import numpy as np

    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _to_jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def _resize(rows: list, target: int | None) -> list:
    """Resize a domain's rows to exactly `target`: truncate if longer, cycle-repeat if
    shorter. `target=None` -> return rows unchanged (use the full pool)."""
    if target is None or not rows:
        return rows
    if len(rows) >= target:
        return rows[:target]
    out = []
    while len(out) < target:
        out.extend(rows)
    return out[:target]


def _templated_rows(path: str, tokenizer, input_key: str, domain: str, target: int | None) -> list:
    """Load a chat-list prompt source (math / search), pre-apply the chat template via
    slime's Dataset (byte-identical to training-time templating), tag domain. Reads only
    as many origin_samples as needed when a target cap is set."""
    ds = Dataset(
        path,
        tokenizer,
        processor=None,
        max_length=None,
        prompt_key=input_key,
        apply_chat_template=True,  # <-- bake the template into sample.prompt
    )
    rows = []
    for s in ds.origin_samples:
        rows.append({"prompt": s.prompt, "metadata": {"domain": domain}})
        if target is not None and len(rows) >= target:
            break  # no need to load the whole (possibly 170k-row) pool just to truncate it
    return _resize(rows, target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="HF dir for the tokenizer (e.g. Qwen3-8B-Base)")
    ap.add_argument("--math-jsonl", required=True)
    ap.add_argument("--search-parquet", required=True)
    ap.add_argument("--tau-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--math-input-key", default="prompt")
    ap.add_argument("--search-input-key", default="prompt")
    ap.add_argument("--tau-input-key", default="index")
    ap.add_argument(
        "--per-domain-target",
        type=int,
        default=None,
        help="Resize EACH domain to exactly this many rows (truncate large / repeat small) "
        "for ~1:1:1 batches. Omit to use every domain's full pool.",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    target = args.per_domain_target

    # ---- Math + Search: reproduce --apply-chat-template exactly via slime Dataset ----
    math_rows = _templated_rows(args.math_jsonl, tokenizer, args.math_input_key, "math", target)
    search_rows = _templated_rows(args.search_parquet, tokenizer, args.search_input_key, "search", target)

    # ---- Tau: keep the task index verbatim; no templating ----
    tau_rows = []
    with open(args.tau_jsonl) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            meta = _to_jsonable(row.get("metadata") or {})
            meta["domain"] = "tau"
            tau_rows.append({"prompt": str(row[args.tau_input_key]), "metadata": meta})  # tau generate does int(sample.prompt)
    tau_rows = _resize(tau_rows, target)

    with open(args.out, "w") as fout:
        for row in (*math_rows, *search_rows, *tau_rows):
            fout.write(json.dumps(row) + "\n")

    print(
        f"wrote {len(math_rows)} math + {len(search_rows)} search + {len(tau_rows)} tau "
        f"= {len(math_rows) + len(search_rows) + len(tau_rows)} rows -> {args.out}"
    )


if __name__ == "__main__":
    main()
