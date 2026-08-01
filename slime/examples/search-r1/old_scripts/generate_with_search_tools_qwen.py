"""Search-R1 generate function using structured messages and apply_chat_template with tools.

Key differences from generate_with_search_memory_qwen.py:
1. Uses apply_chat_template(tools=...) to inject tool definitions in the initial prompt
2. Uses <tool_response> tags for search results (instead of <information> tags)
3. Uses the same string-concatenation + direct model_token_ids approach for training data
4. Maintains the same reward function and compression logic for compatibility
"""

import asyncio
import json
import logging
import re

from qa_em_format_qwen import (  # type: ignore
    compute_score_em,
    em_check,
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
    # ============== Reward Model Configuration ==============
    "format_score": 0.2,
    # ============== Memory Context Window ==============
    "context_window_k": 2,
    "max_docs_compressed": 1,
    "max_chars_per_doc_compressed": 300,
    # ============== Anti-Collapse Masking ==============
    "mask_no_tool_call_warmup": 15,  # start masking after this many rollouts
}

# Module-level rollout counter. Each rollout calls reward_func for
# (rollout_batch_size * n_samples_per_prompt) samples. We count calls
# and estimate the rollout number.  Thread-safe via GIL for single-worker.
_reward_call_count = 0


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


# Old instruction substring to detect and replace in incoming prompts
OLD_INSTRUCTION_SUBSTRING = (
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
    "and it will return the top searched results between <information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
    "without detailed illustrations. For example, <answer> Beijing </answer>."
)

# Old NEW_INSTRUCTION that manually embeds tool schema (to be replaced)
OLD_NEW_INSTRUCTION_PREFIX = (
    "You must conduct reasoning inside <think> and </think> first every time you get new information.\n"
    "After reasoning, if you find you lack some knowledge, you may call a function to help.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:"
)

# Simplified instruction: tool schema is now injected by apply_chat_template(tools=...)
# TOOL_CALLING_INSTRUCTION = (
#     "You must conduct reasoning inside <think> and </think> first every time you get new information.\n"
#     "After reasoning, if you find you lack some knowledge, you may call the search function to help.\n"
#     "You can search as many times as you want.\n"
#     "If you find no further external knowledge needed, you can directly provide the answer "
#     "inside <answer> and </answer>, without detailed illustrations. "
#     "For example, <answer> Beijing </answer>."
# )
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
    """Parse Qwen chat-template formatted text back into a messages list.

    Handles the standard format:
        <|im_start|>role\\ncontent<|im_end|>
    Skips empty assistant messages (generation prompts).
    """
    messages = []
    pattern = r"<\|im_start\|>(\w+)\n(.*?)(?:<\|im_end\|>)"
    for match in re.finditer(pattern, prompt_text, re.DOTALL):
        role = match.group(1)
        content = match.group(2).strip()
        if role == "assistant" and not content:
            continue  # skip the trailing generation prompt
        messages.append({"role": role, "content": content})
    return messages


def clean_instruction_in_messages(messages: list[dict]) -> list[dict]:
    """Remove old manually-embedded tool descriptions from messages and inject
    the simplified TOOL_CALLING_INSTRUCTION.

    The tool schema itself will be injected by apply_chat_template(tools=...).
    """
    for msg in messages:
        content = msg["content"]

        # Remove old <search>-style instruction
        if OLD_INSTRUCTION_SUBSTRING in content:
            content = content.replace(OLD_INSTRUCTION_SUBSTRING, TOOL_CALLING_INSTRUCTION)
            msg["content"] = content
            return messages

        # Remove old NEW_INSTRUCTION that manually embeds tool schema
        if OLD_NEW_INSTRUCTION_PREFIX in content:
            # Find and replace the whole block up to the answer example
            # The block ends with "For example, <answer> Beijing </answer>."
            pattern = (
                r"You must conduct reasoning inside <think>.*?"
                r"For example, <answer> Beijing </answer>\."
            )
            content = re.sub(pattern, TOOL_CALLING_INSTRUCTION, content, flags=re.DOTALL)
            msg["content"] = content
            return messages

    # No old instruction found: inject TOOL_CALLING_INSTRUCTION into user message
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
# Response parsing (reused from original)
# ---------------------------------------------------------------------------

def postprocess_responses(resp: str) -> str:
    if "</tool_call>" in resp:
        return resp.split("</tool_call>")[0] + "</tool_call>"
    if "</answer>" in resp:
        return resp.split("</answer>")[0] + "</answer>"
    # Truncate after bare JSON tool call if present
    bare_pattern = r'\{[^{}]*"name"\s*:\s*"search"[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
    match = re.search(bare_pattern, resp)
    if match:
        return resp[: match.end()]
    return resp


def _extract_search_query(data: dict) -> tuple[str | None, str]:
    """Extract search query from a parsed JSON dict.

    Returns (action, query) where action is 'search' or (None, '').
    """
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
    """Parse JSON inside <tool_call>...</tool_call>.

    Returns (action, content) where action is 'search' or None.
    """
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


def _parse_bare_json_tool_call(text: str):
    """Fallback: match bare {"name": "search", "arguments": {...}} without <tool_call> tags.

    Returns (action, content) where action is 'search' or None.
    """
    # Match JSON object containing "name" and "arguments" keys
    pattern = r'\{[^{}]*"name"\s*:\s*"search"[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
    match = re.search(pattern, text)
    if not match:
        # Try reversed key order: arguments before name
        pattern = r'\{[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*"name"\s*:\s*"search"[^{}]*\}'
        match = re.search(pattern, text)
    if not match:
        return None, ""
    try:
        data = json.loads(match.group(0))
    except Exception:
        return None, ""
    return _extract_search_query(data)


def postprocess_predictions(prediction: str):
    """Parse model output into (action, content).

    Priority: <tool_call> tag > bare JSON > <answer> tag.

    Returns:
        ("search", query) if a valid tool_call for search is found
        ("answer", answer_text) if an <answer> block is found
        (None, "") otherwise
    """
    # 1. Try <tool_call>...</tool_call> first
    action, content = _parse_tool_call_block(prediction)
    if action is not None:
        return action, content

    # 2. Fallback: bare JSON {"name": "search", ...}
    action, content = _parse_bare_json_tool_call(prediction)
    if action is not None:
        return action, content

    # 3. <answer>...</answer>
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
    """Compress raw search result text by keeping fewer docs with truncated text."""
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
    """Manages multi-turn context with tool response formatting.

    Uses string concatenation for context building (like the proven
    generate_with_search_memory_qwen approach) to ensure consistency
    between the context sent to sglang and the training data.

    The initial prompt uses apply_chat_template(tools=...) to inject
    tool definitions natively. Subsequent turns are concatenated as text.
    """

    def __init__(self, prompt_text: str, tokenizer, config: dict):
        self.prompt_text = prompt_text
        self.tokenizer = tokenizer
        self.window_k = config["context_window_k"]
        self.max_docs = config["max_docs_compressed"]
        self.max_chars = config["max_chars_per_doc_compressed"]
        self.turns: list[dict] = []

    @staticmethod
    def _format_tool_response(search_result: str) -> str:
        """Format search result for build_context() (text-based).

        sglang includes <|im_end|> (and sometimes <|endoftext|>) in output["text"],
        so model_text already closes the assistant turn.  We only need a newline
        before <|im_start|>tool to match Qwen chat format.
        """
        return (
            f"\n<|im_start|>tool\n<tool_response>\n{search_result}\n</tool_response><|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    @staticmethod
    def _format_tool_response_for_tokens(search_result: str) -> str:
        """Format search result for build_training_data() (token-based).

        Does NOT prepend <|im_end|>\\n because model_token_ids from sglang's
        output_token_logprobs already contains the <|im_end|> token (151645)
        as the last generated token.  Adding it again would produce a double
        EOS in the training sequence.
        """
        return (
            f"\n"
            f"<|im_start|>tool\n<tool_response>\n{search_result}\n</tool_response><|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def add_turn(
        self,
        model_text: str,
        model_token_ids: list[int] | None = None,
        model_log_probs: list[float] | None = None,
        search_result: str | None = None,
    ):
        """Record a turn (assistant response + optional search result)."""
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
        """Check if a turn's search result should be kept in full."""
        num_search_turns = sum(1 for t in self.turns if t["search_result_full"] is not None)
        search_count = 0
        for i, t in enumerate(self.turns):
            if t["search_result_full"] is not None:
                search_count += 1
                if i == turn_idx:
                    return search_count > num_search_turns - self.window_k
        return True

    def _get_search_result(self, turn_idx: int) -> str | None:
        """Get search result for a turn (full or compressed based on window)."""
        turn = self.turns[turn_idx]
        if turn["search_result_full"] is None:
            return None
        if self._is_within_window(turn_idx):
            return turn["search_result_full"]
        return turn["search_result_compressed"]

    def build_context(self, add_final_generation_prompt: bool = True) -> str:
        """Build the full context string.

        add_final_generation_prompt=True  (default): used during the inference
            loop — always append \\n<|im_start|>assistant\\n after the last turn
            so sglang knows to generate as the assistant on the next call.
        add_final_generation_prompt=False: used after the loop to build
            sample.response — the final answer/truncation turn should not have
            a trailing generation prompt in the saved trajectory.
        """
        context = self.prompt_text
        for i, turn in enumerate(self.turns):
            is_last = i == len(self.turns) - 1
            # sglang may output <|endoftext|> as EOS; replace with <|im_end|>
            # to match Qwen chat format.  On the last turn, keep <|endoftext|>
            # after <|im_end|> so the sequence is properly terminated.
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
        """Build response_token_ids, loss_mask, and rollout_log_probs.

        Uses the same proven approach as generate_with_search_memory_qwen:
        - Assistant tokens use model_token_ids directly from sglang (loss_mask=1)
        - Tool response tokens are tokenized from rendered text (loss_mask=0)
        """
        response_token_ids: list[int] = []
        loss_mask: list[int] = []
        rollout_log_probs: list[float] | None = [] if return_logprob else None

        endoftext_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        for i, turn in enumerate(self.turns):
            is_last = i == len(self.turns) - 1

            # === Assistant tokens (from sglang output, trainable) ===
            # sglang stops on <|endoftext|> (eos_token), but the model often
            # generates <|im_end|> first (from chat-template training), producing
            # [..., im_end, endoftext].  We need to normalise the tail to:
            #   - non-last turn: [..., im_end]          (one im_end, no endoftext)
            #   - last turn:     [..., im_end, endoftext]
            # Strip all trailing im_end / endoftext, then re-append the correct ending.
            cur_token_ids = list(turn["model_token_ids"])
            cur_log_probs = list(turn["model_log_probs"]) if return_logprob else None
            # Strip trailing special tokens
            while cur_token_ids and cur_token_ids[-1] in (endoftext_id, im_end_id):
                cur_token_ids.pop()
                if cur_log_probs is not None:
                    cur_log_probs.pop()
            # Re-append correct ending
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

            # === Tool response tokens (environment, not trainable) ===
            search_result = self._get_search_result(i)
            if search_result is not None:
                # Use the token variant: model_token_ids already ends with the
                # <|im_end|> token (151645), so we must NOT prepend it again.
                obs_text = self._format_tool_response_for_tokens(search_result)
                obs_token_ids = self.tokenizer(
                    obs_text, add_special_tokens=False
                )["input_ids"]
                response_token_ids += obs_token_ids
                loss_mask += [0] * len(obs_token_ids)
                if return_logprob:
                    rollout_log_probs += [0.0] * len(obs_token_ids)
            elif i < len(self.turns) - 1:
                # Intermediate turn with no search result (invalid action):
                # model_token_ids already ends with <|im_end|>, so the separator
                # only needs \n<|im_start|>assistant\n (no leading <|im_end|>).
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
# Main generate function
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

    # --- Step 1: Parse incoming prompt into messages ---
    raw_prompt: str = sample.prompt
    initial_messages = parse_prompt_to_messages(raw_prompt)
    if not initial_messages:
        logger.warning("Failed to parse prompt into messages, falling back to raw text.")
        initial_messages = [{"role": "user", "content": raw_prompt}]

    # --- Step 2: Clean up instruction (remove manual tool schema, add simplified instruction) ---
    initial_messages = clean_instruction_in_messages(initial_messages)

    # --- Step 3: Build prompt using apply_chat_template with tools ---
    prompt_text = state.tokenizer.apply_chat_template(
        initial_messages, tokenize=False, add_generation_prompt=True, tools=tools
    )
    prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # --- Step 4: Initialize context manager ---
    manager = MessageContextWindowManager(prompt_text, state.tokenizer, config)

    last_finish_reason = None

    # --- Step 5: Multi-turn interaction loop ---
    for _turn_idx in range(config["max_turns"]):
        context = manager.build_context()
        payload: dict = {
            "text": context,
            "sampling_params": sampling_params,
        }
        if return_logprob:
            payload["return_logprob"] = True

        output = await post(url, payload)

        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]
        last_finish_reason = output["meta_info"]["finish_reason"]["type"]


        # Get token IDs and log probs from sglang output
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

        # Handle max token length truncation
        if last_finish_reason == "length":
            manager.add_turn(
                cur_response, cur_token_ids, cur_log_probs, search_result=None
            )
            break

        # Parse model output for action
        action, content = postprocess_predictions(cur_response)

        if action == "answer":
            manager.add_turn(
                cur_response, cur_token_ids, cur_log_probs, search_result=None
            )
            break
        elif action == "search":
            # Execute search
            async with SEMAPHORE:
                search_results = await search(content)
            manager.add_turn(
                cur_response,
                cur_token_ids,
                cur_log_probs,
                search_result=search_results.strip(),
            )
        else:
            # Invalid action: add turn without search result, the model will see the
            # context without a tool response and should try again on next turn.
            # We add a user message hint so the model knows to retry.
            manager.add_turn(
                cur_response, cur_token_ids, cur_log_probs, search_result=None
            )
            # Note: without a tool response, the model just continues from its last output.
            # The next context will include the assistant message without a tool result,
            # prompting the model to try a different approach.

    # --- Step 6: Build training data ---
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

    # Build response text from the final (compressed) context view.
    # add_final_generation_prompt=False so the saved trajectory does not end
    # with a dangling \n<|im_start|>assistant\n after the last turn.
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
# Reward function (same scoring logic as original)
# ---------------------------------------------------------------------------

# 答对 + 格式正确	1.0	满分
# 答对 + 格式不对	0.4	0.6*score - structure_format_score = 0.6 - 0.2
# 答错 + 格式正确 + 检索到答案	0.3	structure_format_score + retrieval_score = 0.2 + 0.1
# 答错 + 格式正确 + 没检索到	0.2	structure_format_score = 0.2
# 答错 + 格式不对	-0.1	final_format_score = -0.1
# 没提取到答案 + 格式不对	-0.1	最低
async def reward_func(args, sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    global _reward_call_count
    _reward_call_count += 1

    config = SEARCH_R1_CONFIGS
    fmt = config["format_score"]
    solution_str = sample.prompt + sample.response
    score = compute_score_em(
        solution_str=solution_str,
        ground_truth=sample.label["ground_truth"],
        structure_format_score=fmt,
        final_format_score=-0.1,
        retrieval_score=fmt,
    )

    # Track tool calling and format metrics for wandb
    is_valid, _ = is_valid_sequence(solution_str)
    has_tool_call = (
        "<tool_call>" in sample.response
        or _parse_bare_json_tool_call(sample.response)[0] is not None
    )
    answer = extract_solution(solution_str)
    answer_correct = bool(
        answer and em_check(answer, sample.label["ground_truth"]["target"])
    )

    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["valid_format"] = int(is_valid)
    sample.metadata["has_tool_call"] = int(has_tool_call)
    sample.metadata["answer_correct"] = int(answer_correct)
    sample.metadata["retrieval_correct"] = int(
        is_retrieval_correct(solution_str, sample.label["ground_truth"]["target"])
    )

    # --- Anti-collapse: selective masking with warmup ---
    # Estimate current rollout number from call count.
    # Each rollout = rollout_batch_size * n_samples_per_prompt calls.
    samples_per_rollout = getattr(args, "rollout_batch_size", 64) * getattr(args, "n_samples_per_prompt", 5)
    current_rollout = _reward_call_count // samples_per_rollout
    warmup = config["mask_no_tool_call_warmup"]

    if current_rollout >= warmup and not has_tool_call:
        if sample.loss_mask is not None:
            sample.loss_mask = [0] * len(sample.loss_mask)


    return score
