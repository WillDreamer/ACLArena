"""Search-R1 generate function — SFT-friendly variant.

Same generation/multi-turn logic as generate_with_search_tools_qwen.py.
The only differences are:

1. Imports the new scorer from qa_em_format_qwen_sft (Fix #1 + #3).
2. reward_func uses executed_tool_call (Fix #1) instead of substring match.
3. reward_func short-circuits truncated samples to a negative shaping reward
   (Fix #2) by passing sample.status into the scorer.
4. The anti-collapse loss-mask gating is now driven by executed_tool_call,
   so faking a <tool_call> token in <think> no longer rescues a sample from
   being masked.
5. metadata['has_tool_call'] is preserved for backward-compatible wandb
   plotting, and a new metadata['executed_tool_call'] is recorded.

See analysis of step-48 reward collapse in
multi_stage_rl_log/search_Qwen3-30B-A3B_tis_bs_64_memory_qwen_strict_v3_gspo_cold_then_mask_format_sft/
for the motivation behind these changes.
"""

import asyncio
import json
import logging
import re

from qa_em_format_qwen_sft import (  # type: ignore
    _parse_bare_json_tool_call,
    compute_score_em_sft,
    em_check,
    executed_tool_call,
    extract_solution,
    is_retrieval_correct,
    is_valid_sequence,
)

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


SEARCH_R1_CONFIGS = {
    # ============== General Configuration ==============
    "max_turns": 5,
    "topk": 3,
    "search_concurrency": 256,
    # ============== Search Backend Selection ==============
    "search_backend": "local",
    # ============== Local Search Configuration ==============
    "local": {
        "search_url": "http://131.179.168.117:8000/retrieve",
        "proxy": None,
    },
    # ============== Google Search Configuration ==============
    "google": {
        "api_key": "your_api_key_here",
        "snippet_only": True,
        "proxy": None,
    },
    # ============== Log Probability Collection ==============
    "return_logprob": True,
    # ============== Memory Context Window ==============
    "context_window_k": 2,
    "max_docs_compressed": 1,
    "max_chars_per_doc_compressed": 1000,
}


SEMAPHORE = asyncio.Semaphore(SEARCH_R1_CONFIGS["search_concurrency"])


SEARCH_TOOL_DESC = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search for information using a search engine",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        },
    },
}


OLD_INSTRUCTION_SUBSTRING = (
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
    "and it will return the top searched results between <information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
    "without detailed illustrations. For example, <answer> Beijing </answer>."
)

OLD_NEW_INSTRUCTION_PREFIX = (
    "You must conduct reasoning inside <think> and </think> first every time you get new information.\n"
    "After reasoning, if you find you lack some knowledge, you may call a function to help.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:"
)

TOOL_CALLING_INSTRUCTION = (
    "First conduct reasoning in english every time you try to get new information.\n"
    "After reasoning, if you find you lack some knowledge, you may call a function to help.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    + json.dumps(SEARCH_TOOL_DESC, indent=2)
    + "\n</tools>\n\n"
    "For each function call, return a JSON object with function name and arguments "
    "within <tool_call></tool_call> XML tags. For example:\n"
    "<tool_call>\n"
    '{"name": "search", "arguments": {"query": "your search query"}}\n'
    "</tool_call>\n"
    "The search results will be returned in <tool_response></tool_response> tags.\n"
    "You can search as many times as you want.\n"
    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
    "without detailed illustrations. For example, <answer> Beijing </answer>. \n"
)


# ---------------------------------------------------------------------------
# Prompt parsing
# ---------------------------------------------------------------------------

def parse_prompt_to_messages(prompt_text: str) -> list[dict]:
    messages = []
    pattern = r"<\|im_start\|>(\w+)\n(.*?)(?:<\|im_end\|>)"
    for match in re.finditer(pattern, prompt_text, re.DOTALL):
        role = match.group(1)
        content = match.group(2).strip()
        if role == "assistant" and not content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def clean_instruction_in_messages(messages: list[dict]) -> list[dict]:
    for msg in messages:
        content = msg["content"]
        if OLD_INSTRUCTION_SUBSTRING in content:
            content = content.replace(OLD_INSTRUCTION_SUBSTRING, TOOL_CALLING_INSTRUCTION)
            msg["content"] = content
            return messages
        if OLD_NEW_INSTRUCTION_PREFIX in content:
            pattern = (
                r"You must conduct reasoning inside <think>.*?"
                r"For example, <answer> Beijing </answer>\."
            )
            content = re.sub(pattern, TOOL_CALLING_INSTRUCTION, content, flags=re.DOTALL)
            msg["content"] = content
            return messages

    for msg in messages:
        if msg["role"] == "user":
            msg["content"] = TOOL_CALLING_INSTRUCTION + "\n\n" + msg["content"]
            break

    return messages


# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------

def _passages2string(retrieval_result):
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
    return format_reference


async def search(query: str) -> str:
    backend = SEARCH_R1_CONFIGS["search_backend"]

    if backend == "local":
        from local_search_server import local_search

        local_config = SEARCH_R1_CONFIGS["local"]
        result = await local_search(
            local_config["search_url"],
            query,
            SEARCH_R1_CONFIGS["topk"],
            proxy=local_config["proxy"],
        )
    elif backend == "google":
        from google_search_server import google_search

        google_config = SEARCH_R1_CONFIGS["google"]
        result = await google_search(
            google_config["api_key"],
            query,
            SEARCH_R1_CONFIGS["topk"],
            snippet_only=google_config["snippet_only"],
            proxy=google_config["proxy"],
        )
    else:
        raise ValueError(
            f"Unknown search backend: {backend}. Must be either 'local' or 'google'."
        )

    return _passages2string(result)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def postprocess_responses(resp: str) -> str:
    if "</tool_call>" in resp:
        return resp.split("</tool_call>")[0] + "</tool_call>"
    if "</answer>" in resp:
        return resp.split("</answer>")[0] + "</answer>"
    bare_pattern = r'\{[^{}]*"name"\s*:\s*"search"[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
    match = re.search(bare_pattern, resp)
    if match:
        return resp[: match.end()]
    return resp


def _extract_search_query(data: dict) -> tuple[str | None, str]:
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        return None, ""

    name = data.get("name")
    if name != "search":
        return None, ""

    arguments = data.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            return None, ""
    if not isinstance(arguments, dict):
        return None, ""
    query = arguments.get("query", "")
    if not isinstance(query, str):
        return None, ""
    return "search", query.strip()


def _parse_tool_call_block(block: str):
    pattern = r"<tool_call>(.*?)</tool_call>"
    match = re.search(pattern, block, re.DOTALL)
    if not match:
        return None, ""
    raw = match.group(1).strip()
    try:
        data = json.loads(raw)
    except Exception:
        return None, ""
    return _extract_search_query(data)


def postprocess_predictions(prediction: str):
    action, content = _parse_tool_call_block(prediction)
    if action is not None:
        return action, content

    action, content = _parse_bare_json_tool_call(prediction)
    if action is not None:
        return action, content

    ans_pattern = r"<answer>(.*?)</answer>"
    match = re.search(ans_pattern, prediction, re.DOTALL)
    if match:
        content = match.group(1).strip()
        return "answer", content

    return None, ""


# ---------------------------------------------------------------------------
# Compression utilities
# ---------------------------------------------------------------------------

def compress_search_result(text: str, max_docs: int, max_chars_per_doc: int) -> str:
    doc_pattern = r"(Doc \d+\(Title: [^)]*\))\s*(.*?)(?=Doc \d+\(Title:|$)"
    docs = re.findall(doc_pattern, text, re.DOTALL)
    if not docs:
        return text

    compressed_parts = []
    for _i, (header, body) in enumerate(docs[:max_docs]):
        body = body.strip()
        if len(body) > max_chars_per_doc:
            body = body[:max_chars_per_doc] + "..."
        compressed_parts.append(f"{header} {body}")

    return "\n".join(compressed_parts)


# ---------------------------------------------------------------------------
# Message-based Context Window Manager
# ---------------------------------------------------------------------------

class MessageContextWindowManager:
    def __init__(self, prompt_text: str, tokenizer, config: dict):
        self.prompt_text = prompt_text
        self.tokenizer = tokenizer
        self.window_k = config["context_window_k"]
        self.max_docs = config["max_docs_compressed"]
        self.max_chars = config["max_chars_per_doc_compressed"]
        self.turns: list[dict] = []

    @staticmethod
    def _format_tool_response(search_result: str) -> str:
        return (
            f"\n<|im_start|>user\n<tool_response>\n{search_result}\n</tool_response><|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    @staticmethod
    def _format_tool_response_for_tokens(search_result: str) -> str:
        return (
            f"\n"
            f"<|im_start|>user\n<tool_response>\n{search_result}\n</tool_response><|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def add_turn(
        self,
        model_text: str,
        model_token_ids: list[int] | None = None,
        model_log_probs: list[float] | None = None,
        search_result: str | None = None,
    ):
        turn = {
            "model_text": model_text,
            "model_token_ids": model_token_ids or [],
            "model_log_probs": model_log_probs or [],
            "search_result_full": search_result,
            "search_result_compressed": compress_search_result(
                search_result, self.max_docs, self.max_chars
            )
            if search_result
            else None,
        }
        self.turns.append(turn)

    def _is_within_window(self, turn_idx: int) -> bool:
        num_search_turns = sum(1 for t in self.turns if t["search_result_full"] is not None)
        search_count = 0
        for i, t in enumerate(self.turns):
            if t["search_result_full"] is not None:
                search_count += 1
                if i == turn_idx:
                    return search_count > num_search_turns - self.window_k
        return True

    def _get_search_result(self, turn_idx: int) -> str | None:
        turn = self.turns[turn_idx]
        if turn["search_result_full"] is None:
            return None
        if self._is_within_window(turn_idx):
            return turn["search_result_full"]
        return turn["search_result_compressed"]

    def build_context(self, add_final_generation_prompt: bool = True) -> str:
        context = self.prompt_text
        for i, turn in enumerate(self.turns):
            is_last = i == len(self.turns) - 1
            model_text = turn["model_text"]
            if "<|endoftext|>" in model_text:
                if is_last:
                    model_text = model_text.replace(
                        "<|endoftext|>", "<|im_end|><|endoftext|>"
                    )
                else:
                    model_text = model_text.replace("<|endoftext|>", "<|im_end|>")
            context += model_text
            search_result = self._get_search_result(i)
            if search_result is not None:
                context += self._format_tool_response(search_result)
            elif not is_last or add_final_generation_prompt:
                context += "\n<|im_start|>assistant\n"
        return context

    def build_training_data(
        self, return_logprob: bool
    ) -> tuple[list[int], list[int], list[float] | None]:
        response_token_ids: list[int] = []
        loss_mask: list[int] = []
        rollout_log_probs: list[float] | None = [] if return_logprob else None

        endoftext_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        for i, turn in enumerate(self.turns):
            is_last = i == len(self.turns) - 1

            cur_token_ids = list(turn["model_token_ids"])
            cur_log_probs = list(turn["model_log_probs"]) if return_logprob else None
            while cur_token_ids and cur_token_ids[-1] in (endoftext_id, im_end_id):
                cur_token_ids.pop()
                if cur_log_probs is not None:
                    cur_log_probs.pop()
            if is_last:
                cur_token_ids.append(im_end_id)
                cur_token_ids.append(endoftext_id)
                if cur_log_probs is not None:
                    cur_log_probs.append(0.0)
                    cur_log_probs.append(0.0)
            else:
                cur_token_ids.append(im_end_id)
                if cur_log_probs is not None:
                    cur_log_probs.append(0.0)

            response_token_ids += cur_token_ids
            loss_mask += [1] * len(cur_token_ids)
            if return_logprob:
                rollout_log_probs += cur_log_probs

            search_result = self._get_search_result(i)
            if search_result is not None:
                obs_text = self._format_tool_response_for_tokens(search_result)
                obs_token_ids = self.tokenizer(
                    obs_text, add_special_tokens=False
                )["input_ids"]
                response_token_ids += obs_token_ids
                loss_mask += [0] * len(obs_token_ids)
                if return_logprob:
                    rollout_log_probs += [0.0] * len(obs_token_ids)
            elif i < len(self.turns) - 1:
                sep_text = "\n<|im_start|>assistant\n"
                sep_token_ids = self.tokenizer(
                    sep_text, add_special_tokens=False
                )["input_ids"]
                response_token_ids += sep_token_ids
                loss_mask += [0] * len(sep_token_ids)
                if return_logprob:
                    rollout_log_probs += [0.0] * len(sep_token_ids)

        return response_token_ids, loss_mask, rollout_log_probs

    def get_stats(self) -> dict:
        num_turns = len(self.turns)
        compressed_count = sum(
            1
            for i, t in enumerate(self.turns)
            if t["search_result_full"] is not None and not self._is_within_window(i)
        )
        full_count = sum(
            1
            for i, t in enumerate(self.turns)
            if t["search_result_full"] is not None and self._is_within_window(i)
        )
        return {
            "num_turns": num_turns,
            "obs_compressed": compressed_count,
            "obs_full": full_count,
        }


# ---------------------------------------------------------------------------
# Main generate function (identical to the original)
# ---------------------------------------------------------------------------

async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert (
        not args.partial_rollout
    ), "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    config = SEARCH_R1_CONFIGS
    return_logprob = config["return_logprob"]
    tools = [SEARCH_TOOL_DESC]

    raw_prompt: str = sample.prompt
    initial_messages = parse_prompt_to_messages(raw_prompt)
    if not initial_messages:
        logger.warning("Failed to parse prompt into messages, falling back to raw text.")
        initial_messages = [{"role": "user", "content": raw_prompt}]

    initial_messages = clean_instruction_in_messages(initial_messages)

    prompt_text = state.tokenizer.apply_chat_template(
        initial_messages, tokenize=False, add_generation_prompt=True, tools=tools
    )
    prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    manager = MessageContextWindowManager(prompt_text, state.tokenizer, config)

    last_finish_reason = None

    # Force sglang to stop the moment the model finishes an action tag, so the
    # rollout trajectory never contains post-action garbage. Verified against
    # this sglang build: `no_stop_trim=True` keeps the matched stop string in
    # both `text` and `output_token_logprobs`, so cur_response / cur_token_ids /
    # cur_log_probs stay perfectly aligned without any manual trimming.
    action_stops = ["</tool_call>", "</answer>"]
    turn_sampling = dict(sampling_params)
    turn_sampling["stop"] = list(turn_sampling.get("stop") or []) + action_stops
    turn_sampling["no_stop_trim"] = True

    for _turn_idx in range(config["max_turns"]):
        context = manager.build_context()
        payload: dict = {
            "text": context,
            "sampling_params": turn_sampling,
        }
        if return_logprob:
            payload["return_logprob"] = True

        output = await post(url, payload)

        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]
        last_finish_reason = output["meta_info"]["finish_reason"]["type"]

        if return_logprob:
            if "output_token_logprobs" not in output["meta_info"]:
                raise RuntimeError(
                    "output_token_logprobs not found in output meta_info. "
                    "Make sure 'return_logprob': True is set in the payload."
                )
            cur_token_ids = [
                item[1] for item in output["meta_info"]["output_token_logprobs"]
            ]
            cur_log_probs = [
                item[0] for item in output["meta_info"]["output_token_logprobs"]
            ]
        else:
            cur_response = postprocess_responses(cur_response)
            cur_token_ids = state.tokenizer(
                cur_response, add_special_tokens=False
            )["input_ids"]
            cur_log_probs = None

        if last_finish_reason == "length":
            manager.add_turn(
                cur_response, cur_token_ids, cur_log_probs, search_result=None
            )
            break

        action, content = postprocess_predictions(cur_response)

        if action == "answer":
            manager.add_turn(
                cur_response, cur_token_ids, cur_log_probs, search_result=None
            )
            break
        elif action == "search":
            async with SEMAPHORE:
                search_results = await search(content)
            manager.add_turn(
                cur_response,
                cur_token_ids,
                cur_log_probs,
                search_result=search_results.strip(),
            )
        else:
            manager.add_turn(
                cur_response, cur_token_ids, cur_log_probs, search_result=None
            )

    response_token_ids, loss_mask, rollout_log_probs = manager.build_training_data(
        return_logprob
    )

    stats = manager.get_stats()
    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["num_turns"] = stats["num_turns"]
    logger.debug(
        f"MessageContextWindow stats: turns={stats['num_turns']}, "
        f"obs_full={stats['obs_full']}, obs_compressed={stats['obs_compressed']}"
    )

    full_context = manager.build_context(add_final_generation_prompt=False)
    response = full_context[len(prompt_text):]

    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    sample.prompt = prompt_text

    if return_logprob:
        sample.rollout_log_probs = rollout_log_probs if rollout_log_probs else None

    match last_finish_reason:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


# ---------------------------------------------------------------------------
# Reward function — SFT-friendly version
# ---------------------------------------------------------------------------
#
# New scoring table (see qa_em_format_qwen_sft.compute_score_em_sft):
#
#   answer_correct + valid_format + executed_tool   ->  1.0
#   answer_correct + valid_format + no_tool         ->  0.3
#   answer_correct + format_bad                     ->  0.2
#   answer_wrong   + valid_format + retrieved       ->  0.4
#   answer_wrong   + valid_format + executed_no_hit ->  0.1
#   answer_wrong   + valid_format + no_tool         -> -0.05
#   answer_wrong   + format_bad                     -> -0.2
#   truncated (executed tool, no answer extracted)  -> -0.05
#   truncated (otherwise)                           -> -0.2
#
async def reward_func(args, sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    solution_str = sample.prompt + sample.response

    # Sample.status may be an enum or string depending on the slime version.
    status_str = getattr(sample.status, "value", None) or str(sample.status or "")

    score = compute_score_em_sft(
        solution_str=solution_str,
        ground_truth=sample.label["ground_truth"],
        status=status_str,
        response=sample.response,
    )

    # ---- Metrics for wandb ----
    is_valid, _ = is_valid_sequence(solution_str)
    # Backward-compatible: keep the old (loose) flag so existing dashboards
    # do not break, AND emit the new strict flag.
    has_tool_call_loose = (
        "<tool_call>" in sample.response
        or _parse_bare_json_tool_call(sample.response)[0] is not None
    )
    # Strict check: only assistant trajectory; empty <tool_response></tool_response>
    # documentation pairs do not count.
    real_tool_call = executed_tool_call(sample.response)
    answer = extract_solution(solution_str)
    answer_correct = bool(
        answer and em_check(answer, sample.label["ground_truth"]["target"])
    )

    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["valid_format"] = int(is_valid)
    sample.metadata["has_tool_call"] = int(has_tool_call_loose)
    sample.metadata["executed_tool_call"] = int(real_tool_call)
    sample.metadata["answer_correct"] = int(answer_correct)
    sample.metadata["retrieval_correct"] = int(
        is_retrieval_correct(solution_str, sample.label["ground_truth"]["target"])
    )

    return score
