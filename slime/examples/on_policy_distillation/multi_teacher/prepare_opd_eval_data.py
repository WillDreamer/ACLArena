"""Build the per-domain EVAL datasets for 3-teacher MOPD (math + search + tau).

Companion to prepare_opd_mixed_data.py (which builds the TRAIN mix). Eval is
per-domain (one dataset per domain in eval_multi.yaml), NOT a single mixed file,
so each domain's native generate_eval + real task reward is used.

Same --apply-chat-template constraint as train: the MOPD run trains/evals WITHOUT
--apply-chat-template (the run has no single global template flag it can set —
math/search prompts are chat-lists, tau is a bare index). So we pre-apply the chat
template OFFLINE to the math + search eval prompts (byte-identical to training-time
templating, via slime's own Dataset), keep the tau task index verbatim, and tag
every row with metadata.domain. The three outputs:

  * eval_math.jsonl   : {"prompt": <templated str>, "label": <answer>, "metadata": {"domain":"math"}}
  * eval_search.jsonl : {"prompt": <templated str>, "reward_model": {...}, "metadata": {"domain":"search"}}
  * eval_tau.jsonl    : {"index": <task idx>, "metadata": {"domain":"tau"}}   (copied verbatim; no template)

Search test_small has 3610 rows — far too many to eval every 10 steps (tau eval
alone is GLM-user-sim-bound). --search-eval-size subsamples it (default 200,
deterministic shuffle) so a periodic eval stays cheap. Math (aime, 30) and tau
(retail_test, 115) are small enough to keep whole.

Usage (run once, offline; then upload the 3 files to S3 and stage them):
    python prepare_opd_eval_data.py \
        --hf /workspace/datasets/models/Qwen3-8B-Base \
        --math-jsonl     aime-2024.jsonl \
        --search-parquet test_small.parquet \
        --tau-jsonl      retail_test_tasks.jsonl \
        --search-eval-size 200 \
        --out-dir /tmp/opd_eval
"""

import argparse
import json
import os
import random

from transformers import AutoTokenizer

# NOTE: intentionally NO `from slime.utils.data import Dataset` — that pulls in
# torch/aiohttp and can't run in a light offline env. We replicate slime Dataset's
# templating EXACTLY (data.py:231 -> tokenizer.apply_chat_template(prompt, tools=None,
# tokenize=False, add_generation_prompt=True) with no apply_chat_template_kwargs, which
# is what the MOPD train prep uses), so the eval prompts are byte-identical to how the
# same prompt would be templated at training time. Reads jsonl OR parquet.


def _to_jsonable(x):
    """parquet cells / metadata can contain numpy types; coerce to plain JSON."""
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


def _read_rows(path):
    """Yield dict rows from a .jsonl or .parquet file."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        for row in pq.read_table(path).to_pylist():
            yield _to_jsonable(row)
    else:
        with open(path) as fin:
            for line in fin:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _templated_eval_rows(path, tokenizer, input_key, label_key, domain, size):
    """Load a chat-list prompt source (math / search), pre-apply the chat template
    (byte-identical to slime training-time templating), keep the label under its
    ORIGINAL key (so eval_multi.yaml's label_key resolves it), tag domain."""
    rows = []
    for data in _read_rows(path):
        prompt = data[input_key]
        # slime's _build_messages: a str prompt with as_conversation=True is wrapped as
        # a single user turn; a list is used as-is.
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        templated = tokenizer.apply_chat_template(
            prompt,
            tools=None,
            tokenize=False,
            add_generation_prompt=True,
        )
        row = {"prompt": templated, "metadata": {"domain": domain}}
        if label_key is not None:
            row[label_key] = _to_jsonable(data[label_key])
        rows.append(row)
    if size is not None and len(rows) > size:
        rng = random.Random(42)
        rng.shuffle(rows)
        rows = rows[:size]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="HF dir for the tokenizer (Qwen3-8B-Base)")
    ap.add_argument("--math-jsonl", required=True, help="aime-2024.jsonl (prompt chat-list + label)")
    ap.add_argument("--search-parquet", required=True, help="test_small.parquet (prompt chat-list + reward_model)")
    ap.add_argument("--tau-jsonl", required=True, help="retail_test_tasks.jsonl (task index rows)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--math-input-key", default="prompt")
    ap.add_argument("--math-label-key", default="label")
    ap.add_argument("--search-input-key", default="prompt")
    ap.add_argument("--search-label-key", default="reward_model")
    ap.add_argument("--tau-input-key", default="index")
    ap.add_argument(
        "--search-eval-size",
        type=int,
        default=200,
        help="Subsample the (3610-row) search test set to this many rows for a cheap periodic eval "
        "(deterministic shuffle, seed 42). Set <=0 to keep the whole set.",
    )
    ap.add_argument(
        "--math-eval-size",
        type=int,
        default=None,
        help="Optional cap on math eval rows (aime is only 30, so default keeps all).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)

    # ---- Math + Search: reproduce --apply-chat-template exactly via slime Dataset ----
    math_rows = _templated_eval_rows(
        args.math_jsonl, tokenizer, args.math_input_key, args.math_label_key, "math",
        args.math_eval_size if (args.math_eval_size and args.math_eval_size > 0) else None,
    )
    search_rows = _templated_eval_rows(
        args.search_parquet, tokenizer, args.search_input_key, args.search_label_key, "search",
        args.search_eval_size if args.search_eval_size and args.search_eval_size > 0 else None,
    )

    # ---- Tau: keep the task index verbatim; no templating (env applies it) ----
    tau_rows = []
    with open(args.tau_jsonl) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            meta = _to_jsonable(row.get("metadata") or {})
            meta["domain"] = "tau"
            tau_rows.append({"index": str(row[args.tau_input_key]), "metadata": meta})

    out_math = os.path.join(args.out_dir, "eval_math.jsonl")
    out_search = os.path.join(args.out_dir, "eval_search.jsonl")
    out_tau = os.path.join(args.out_dir, "eval_tau.jsonl")
    for path, rows in ((out_math, math_rows), (out_search, search_rows), (out_tau, tau_rows)):
        with open(path, "w") as fout:
            for row in rows:
                fout.write(json.dumps(row) + "\n")

    print(
        f"wrote {len(math_rows)} math -> {out_math}\n"
        f"      {len(search_rows)} search -> {out_search}\n"
        f"      {len(tau_rows)} tau -> {out_tau}"
    )


if __name__ == "__main__":
    main()
