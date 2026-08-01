import json
import asyncio
import os
import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
INPUT_PATH = "/data/user/Search-R1/data/nq_hotpotqa_train/train.parquet"
OUTPUT_PATH = "/data/user/Search-R1/data/nq_hotpotqa_train/train_filtered.parquet"
CHECKPOINT_PATH = OUTPUT_PATH.replace(".parquet", "_checkpoint.parquet")
PROMPT_COLUMN = "prompt"           # 改成你 parquet 里存 prompt 的列名
MODEL = "gpt-5-mini"
MAX_CONCURRENCY = 20               # 并发数，按 rate limit 调
KEEP_LABEL = "SEARCH_REQUIRED"     # 保留哪种标签
SAVE_EVERY = 100                   # 每处理多少条保存一次 checkpoint

client = AsyncOpenAI()             # 默认读 OPENAI_API_KEY

# ──────────────────────────────────────────────
# Instruction template
# ──────────────────────────────────────────────
INSTRUCTION_TEMPLATE = """You are an expert evaluator for search-agent benchmarking.

Your task: decide whether a user prompt is suitable for evaluating a search agent. We want to KEEP prompts that ask for specific facts, because these are exactly the prompts where a search agent can demonstrate value — even if a model might guess the answer.

## Labels (pick exactly one)

- **SEARCH_REQUIRED** — the prompt asks for specific factual information. This includes: specific numbers (counts, dates, years, episodes, statistics), specific names (people, places, actors, authors, characters), specific details about real-world entities (shows, movies, songs, products, events, places, organizations), or any factual question where the answer is a concrete, verifiable fact rather than an explanation or opinion.

- **SEARCH_BENEFICIAL** — the prompt asks a factual question but the answer is widely known enough that most educated adults would know it confidently (e.g., "What is the capital of France?", "Who wrote Romeo and Juliet?"). Search could still verify, but the question is arguably too easy.

- **TOO_ANSWERABLE_WITHOUT_SEARCH** — the prompt does NOT ask for factual retrieval at all. It asks for explanations, definitions, reasoning, math, opinions, creative writing, coding help, or conceptual understanding. A search engine is simply not the right tool for these prompts.

## Key principle
The distinction is about the NATURE of the question, not how hard it is:
- Asking for a **specific fact** (who, what, when, where, how many) → SEARCH_REQUIRED or SEARCH_BENEFICIAL
- Asking for **understanding/reasoning/explanation** (why, how does X work, explain, compare concepts) → TOO_ANSWERABLE_WITHOUT_SEARCH

## Examples

SEARCH_REQUIRED (specific factual lookups — KEEP these):
- "How many episodes are in Game of Thrones season 7?"
- "Who played Charlie in Charlie and the Chocolate Factory?"
- "When does season 5 of Bates Motel come out?"
- "Who sang Waiting for a Girl Like You?"
- "Where do you cross the Arctic Circle in Norway?"
- "Who is the current CEO of Intel?"
- "What is the acceptance rate of the 2026 ICML main track?"

SEARCH_BENEFICIAL (trivially known facts):
- "What is the capital of France?"
- "Who wrote Pride and Prejudice?"
- "What language is spoken in Brazil?"

TOO_ANSWERABLE_WITHOUT_SEARCH (not factual retrieval):
- "Explain the difference between TCP and UDP."
- "If a train travels 60 miles in 1.5 hours, what is its average speed?"
- "Why does regularization help reduce overfitting?"
- "Write a Python function to sort a list."
- "What are the pros and cons of microservices?"

## Output format
Return **valid JSON only**, no markdown fences.

{{"label": "SEARCH_REQUIRED" or "SEARCH_BENEFICIAL" or "TOO_ANSWERABLE_WITHOUT_SEARCH", "confidence": 0.0-1.0, "brief_rationale": "1-3 sentences"}}

---

Now evaluate this user prompt:

[USER_PROMPT]
{user_prompt}
[/USER_PROMPT]"""


# ──────────────────────────────────────────────
# Core logic
# ──────────────────────────────────────────────
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


async def judge_one(prompt: str) -> dict:
    """Call GPT-5 to judge a single prompt. Returns parsed dict with raw_response."""
    async with semaphore:
        sent_prompt = INSTRUCTION_TEMPLATE.replace("{user_prompt}", prompt)
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise JSON-only evaluator."},
                    {"role": "user", "content": sent_prompt},
                ],
                max_completion_tokens=4096,
            )
            text = resp.choices[0].message.content.strip()
            raw_response = text
            # 尝试去掉可能的 markdown code fence
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
            parsed["raw_response"] = raw_response
            parsed["sent_prompt"] = sent_prompt
            return parsed
        except json.JSONDecodeError:
            return {"label": "PARSE_ERROR", "confidence": 0.0, "brief_rationale": text,
                    "raw_response": text, "sent_prompt": sent_prompt}
        except Exception as e:
            return {"label": "API_ERROR", "confidence": 0.0, "brief_rationale": str(e),
                    "raw_response": str(e), "sent_prompt": sent_prompt}


def save_checkpoint(df, completed_mask, results, msg=""):
    """Save current progress to checkpoint file."""
    df_cp = df.copy()
    df_cp["_judge_done"] = completed_mask
    df_cp["judge_label"] = [r.get("label", "UNKNOWN") if r else None for r in results]
    df_cp["judge_confidence"] = [r.get("confidence", 0.0) if r else None for r in results]
    df_cp["judge_rationale"] = [r.get("brief_rationale", "") if r else None for r in results]
    df_cp["sent_prompt"] = [r.get("sent_prompt", "") if r else None for r in results]
    df_cp["raw_response"] = [r.get("raw_response", "") if r else None for r in results]
    df_cp.to_parquet(CHECKPOINT_PATH, index=False)
    done_count = sum(completed_mask)
    print(f"  [Checkpoint] {done_count}/{len(df)} saved. {msg}")


def load_checkpoint(df):
    """Load checkpoint if exists. Returns (results list, completed mask, start index)."""
    n = len(df)
    results = [None] * n
    completed = [False] * n
    if not os.path.exists(CHECKPOINT_PATH):
        return results, completed

    print(f"  Found checkpoint: {CHECKPOINT_PATH}")
    df_cp = pd.read_parquet(CHECKPOINT_PATH)

    if "_judge_done" not in df_cp.columns or len(df_cp) != n:
        print("  Checkpoint shape mismatch, starting from scratch.")
        return results, completed

    for i in range(n):
        if df_cp["_judge_done"].iloc[i]:
            completed[i] = True
            results[i] = {
                "label": df_cp["judge_label"].iloc[i],
                "confidence": df_cp["judge_confidence"].iloc[i],
                "brief_rationale": df_cp["judge_rationale"].iloc[i],
                "sent_prompt": df_cp["sent_prompt"].iloc[i],
                "raw_response": df_cp["raw_response"].iloc[i],
            }

    resumed = sum(completed)
    print(f"  Resumed {resumed}/{n} from checkpoint.")
    return results, completed


async def main():
    # 1. 读数据
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Total samples: {len(df)}, columns: {df.columns.tolist()}")

    # # ⬇️ 调试模式：只取前 N 行，正式跑时注释掉这行
    # df = df.head(10)
    # print(f"  [DEBUG] Only using first {len(df)} rows")

    if PROMPT_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{PROMPT_COLUMN}' not found. "
            f"Available columns: {df.columns.tolist()}. "
            f"Please set PROMPT_COLUMN at the top of this script."
        )

    # prompt 列可能是 chat message list，提取纯文本
    def extract_text(p):
        if isinstance(p, list):
            # [{'content': '...', 'role': 'user'}, ...]
            return "\n".join(m["content"] for m in p if isinstance(m, dict) and "content" in m)
        return str(p)

    prompts = df[PROMPT_COLUMN].apply(extract_text).tolist()

    # 2. 加载 checkpoint（如果有）
    results, completed = load_checkpoint(df)
    todo_indices = [i for i, done in enumerate(completed) if not done]

    if not todo_indices:
        print("All samples already judged, skipping to save.")
    else:
        print(f"Calling {MODEL} on {len(todo_indices)} remaining prompts (concurrency={MAX_CONCURRENCY}) ...")

        # 用 lock 保护 results 写入和 checkpoint 保存
        lock = asyncio.Lock()
        done_since_save = 0
        pbar = tqdm(total=len(todo_indices), desc="Judging")

        async def judge_and_store(idx: int):
            nonlocal done_since_save
            result = await judge_one(prompts[idx])
            async with lock:
                results[idx] = result
                completed[idx] = True
                done_since_save += 1
                pbar.update(1)
                if done_since_save >= SAVE_EVERY:
                    save_checkpoint(df, completed, results)
                    done_since_save = 0

        tasks = [judge_and_store(i) for i in todo_indices]
        await asyncio.gather(*tasks)
        pbar.close()

        # 最终 checkpoint
        save_checkpoint(df, completed, results, msg="final")

    # 3. 解析结果，写入 DataFrame
    df["judge_label"] = [r.get("label", "UNKNOWN") for r in results]
    df["judge_confidence"] = [r.get("confidence", 0.0) for r in results]
    df["judge_rationale"] = [r.get("brief_rationale", "") for r in results]
    df["sent_prompt"] = [r.get("sent_prompt", "") for r in results]
    df["raw_response"] = [r.get("raw_response", "") for r in results]

    # 4. 统计
    counts = df["judge_label"].value_counts()
    print("\n--- Label distribution ---")
    print(counts.to_string())

    errors = counts.get("PARSE_ERROR", 0) + counts.get("API_ERROR", 0)
    if errors:
        print(f"\n⚠  {errors} samples had errors, kept for inspection.")

    # 5. 保存完整结果（含错误行，方便排查）
    full_output = OUTPUT_PATH.replace(".parquet", "_full.parquet")
    df.to_parquet(full_output, index=False)
    print(f"Full results saved to {full_output}")

    # 6. 分别保存保留 & 筛掉的
    keep_labels = {KEEP_LABEL}
    df_kept = df[df["judge_label"].isin(keep_labels)].copy()
    df_removed = df[~df["judge_label"].isin(keep_labels)].copy()

    df_kept.to_parquet(OUTPUT_PATH.replace(".parquet", "_kept.parquet"), index=False)
    removed_path = OUTPUT_PATH.replace(".parquet", "_removed.parquet")
    df_removed.to_parquet(removed_path, index=False)

    print(f"\nKept {len(df_kept)}/{len(df)} samples (labels: {keep_labels})")
    print(f"  → {OUTPUT_PATH}")
    print(f"Removed {len(df_removed)}/{len(df)} samples")
    print(f"  → {removed_path}")

    # 7. 清理 checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print(f"Checkpoint removed: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    # asyncio.run(main())
    import pandas as pd

    df_check = pd.read_parquet("/data/user/Search-R1/data/nq_hotpotqa_train/train_filtered_kept.parquet")
    print(df_check.head())
    print("\nColumns:", df_check.columns.tolist())
    print("\nInfo:")
    print(df_check.info())

    # # 统计 judge_confidence 的分布（直方图和基本统计量）
    # print("\n--- judge_confidence describe ---")
    # print(df_check["judge_confidence"].describe())

    # print("\n--- judge_confidence value counts (bin counts) ---")
    # print(pd.cut(df_check["judge_confidence"], bins=10).value_counts().sort_index())

    # # INSERT_YOUR_CODE
    # # 将confidence大于0.88的单独保存成/data/user/Search-R1/data/nq_hotpotqa_train/train_filtered_conf.parquet
    # df_conf = df_check[df_check["judge_confidence"] > 0.88].copy()
    # conf_path = "/data/user/Search-R1/data/nq_hotpotqa_train/train_filtered_conf.parquet"
    # df_conf.to_parquet(conf_path, index=False)
    # print(f"\nSaved {len(df_conf)} samples with judge_confidence > 0.88 to {conf_path}")