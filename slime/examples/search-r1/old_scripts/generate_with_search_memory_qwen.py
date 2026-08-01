import asyncio
import json
import logging
import re

from qa_em_format_qwen import compute_score_em, is_valid_sequence, em_check, extract_solution, is_retrieval_correct  # type: ignore

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


NEW_INSTRUCTION = (
    "You must conduct reasoning inside <think> and </think> first every time you get new information.\n"
    "After reasoning, if you find you lack some knowledge, you may call a function to help.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    + json.dumps(SEARCH_TOOL_DESC, indent=2)
    + "\n</tools>\n\n"
    "For each function call, return a JSON object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": "search", "arguments": {"query": "your search query"}}\n'
    "</tool_call>\n"
    "The search results will be returned in <tool_response></tool_response> tags.\n"
    "You can search as many times as you want.\n"
    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
    "without detailed illustrations. For example, <answer> Beijing </answer>."
)


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


def postprocess_responses(resp: str) -> str:
    if "</tool_call>" in resp:
        return resp.split("</tool_call>")[0] + "</tool_call>"
    if "</answer>" in resp:
        return resp.split("</answer>")[0] + "</answer>"
    return resp


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


def postprocess_predictions(prediction: str):
    # Prefer tool_call over answer if both appear
    action, content = _parse_tool_call_block(prediction)
    if action is not None:
        return action, content

    ans_pattern = r"<answer>(.*?)</answer>"
    match = re.search(ans_pattern, prediction, re.DOTALL)
    if match:
        content = match.group(1).strip()
        return "answer", content

    return None, ""


async def execute_predictions(prediction: str) -> str:
    action, content = postprocess_predictions(prediction)

    if action == "search":
        search_query = content
        async with SEMAPHORE:
            search_results = await search(search_query)
        next_obs = f"\n\n<tool_response>{search_results.strip()}</tool_response>\n\n"
        done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        next_obs = (
            "\nMy previous action is invalid. "
            "If I want to search, I should call the search tool inside <tool_call> and </tool_call> with a JSON object. "
            "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
            "Let me try again.\n"
        )
        done = False

    return next_obs, done


def compress_observation(obs_text: str, max_docs: int, max_chars_per_doc: int) -> str:
    """Compress a <tool_response>...</tool_response> block by keeping fewer docs with truncated text."""
    info_match = re.search(r"<tool_response>(.*?)</tool_response>", obs_text, re.DOTALL)
    if not info_match:
        return obs_text

    inner = info_match.group(1).strip()
    doc_pattern = r"(Doc \d+\(Title: [^)]*\))\s*(.*?)(?=Doc \d+\(Title:|$)"
    docs = re.findall(doc_pattern, inner, re.DOTALL)

    compressed_parts = []
    for i, (header, text) in enumerate(docs[:max_docs]):
        text = text.strip()
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "..."
        compressed_parts.append(f"{header} {text}")

    compressed_inner = "\n".join(compressed_parts)
    return f"\n\n<tool_response>{compressed_inner}\n</tool_response>\n\n"


class ContextWindowManager:
    """Manages multi-turn context with a sliding compression window for Qwen-style tool calls."""

    def __init__(self, prompt_text: str, tokenizer, config: dict):
        self.prompt_text = prompt_text
        self.tokenizer = tokenizer
        self.window_k = config["context_window_k"]
        self.max_docs = config["max_docs_compressed"]
        self.max_chars = config["max_chars_per_doc_compressed"]
        self.turns: list[dict] = []

    def add_turn(
        self,
        model_text: str,
        model_token_ids: list[int],
        model_log_probs: list[float] | None = None,
        observation: str | None = None,
    ):
        turn = {
            "model_text": model_text,
            "model_token_ids": model_token_ids,
            "model_log_probs": model_log_probs or [],
            "obs_full": observation,
            "obs_compressed": compress_observation(
                observation, self.max_docs, self.max_chars
            )
            if observation
            else None,
        }
        self.turns.append(turn)

    def _is_within_window(self, turn_idx: int) -> bool:
        """Check if a turn's observation should be kept in full."""
        num_obs_turns = sum(1 for t in self.turns if t["obs_full"] is not None)
        obs_count = 0
        for i, t in enumerate(self.turns):
            if t["obs_full"] is not None:
                obs_count += 1
                if i == turn_idx:
                    return obs_count > num_obs_turns - self.window_k
        return True

    def _get_obs_text(self, turn_idx: int) -> str:
        turn = self.turns[turn_idx]
        if turn["obs_full"] is None:
            return ""
        if self._is_within_window(turn_idx):
            return turn["obs_full"]
        return turn["obs_compressed"]

    def build_context(self) -> str:
        """Build the full context string for the next SGLang call."""
        context = self.prompt_text
        for i, turn in enumerate(self.turns):
            context += turn["model_text"]
            context += self._get_obs_text(i)
        return context

    def build_training_data(
        self, return_logprob: bool
    ) -> tuple[list[int], list[int], list[float] | None]:
        """Build final response_token_ids, loss_mask, and rollout_log_probs."""
        response_token_ids: list[int] = []
        loss_mask: list[int] = []
        rollout_log_probs: list[float] | None = [] if return_logprob else None

        for i, turn in enumerate(self.turns):
            response_token_ids += turn["model_token_ids"]
            loss_mask += [1] * len(turn["model_token_ids"])
            if return_logprob:
                rollout_log_probs += turn["model_log_probs"]

            obs_text = self._get_obs_text(i)
            if obs_text:
                obs_tokens = self.tokenizer(
                    obs_text, add_special_tokens=False
                )["input_ids"]
                response_token_ids += obs_tokens
                loss_mask += [0] * len(obs_tokens)
                if return_logprob:
                    rollout_log_probs += [0.0] * len(obs_tokens)

        return response_token_ids, loss_mask, rollout_log_probs

    def get_stats(self) -> dict:
        num_turns = len(self.turns)
        compressed_count = sum(
            1
            for i, t in enumerate(self.turns)
            if t["obs_full"] is not None and not self._is_within_window(i)
        )
        full_count = sum(
            1
            for i, t in enumerate(self.turns)
            if t["obs_full"] is not None and self._is_within_window(i)
        )
        return {
            "num_turns": num_turns,
            "obs_compressed": compressed_count,
            "obs_full": full_count,
        }


async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert (
        not args.partial_rollout
    ), "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    config = SEARCH_R1_CONFIGS
    return_logprob = config["return_logprob"]

    prompt_text: str = sample.prompt
    if OLD_INSTRUCTION_SUBSTRING in prompt_text:
        prompt_text = prompt_text.replace(OLD_INSTRUCTION_SUBSTRING, NEW_INSTRUCTION)
    elif "<tool_call>" not in prompt_text and "<search>" not in prompt_text:
        # Eval data has no search instructions at all.
        # Inject NEW_INSTRUCTION before the last user message boundary so the
        # model knows how to use search tools.
        user_tag = "<|im_start|>user\n"
        last_user_pos = prompt_text.rfind(user_tag)
        if last_user_pos != -1:
            insert_pos = last_user_pos + len(user_tag)
            prompt_text = (
                prompt_text[:insert_pos]
                + NEW_INSTRUCTION
                + "\n\n"
                + prompt_text[insert_pos:]
            )

    prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)[
        "input_ids"
    ]

    manager = ContextWindowManager(prompt_text, state.tokenizer, config)

    last_finish_reason = None

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
            manager.add_turn(cur_response, cur_token_ids, cur_log_probs, observation=None)
            break

        next_obs, done = await execute_predictions(cur_response)

        if done:
            manager.add_turn(cur_response, cur_token_ids, cur_log_probs, observation=None)
            break

        assert next_obs != "", "Next observation should not be empty."
        manager.add_turn(cur_response, cur_token_ids, cur_log_probs, observation=next_obs)

    response_token_ids, loss_mask, rollout_log_probs = manager.build_training_data(
        return_logprob
    )

    stats = manager.get_stats()
    sample.metadata["num_turns"] = stats["num_turns"]
    logger.debug(
        f"ContextWindow stats: turns={stats['num_turns']}, "
        f"obs_full={stats['obs_full']}, obs_compressed={stats['obs_compressed']}"
    )

    # Build the full response text from the final (compressed) context view
    full_context = manager.build_context()
    response = full_context[len(prompt_text) :]

    sample.tokens = prompt_tokens_ids + response_token_ids
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


# 答对 + 格式正确	1.0	满分
# 答对 + 格式不对	0.4	0.6*score - structure_format_score = 0.6 - 0.2
# 答错 + 格式正确 + 检索到答案	0.3	structure_format_score + retrieval_score = 0.2 + 0.1
# 答错 + 格式正确 + 没检索到	0.2	structure_format_score = 0.2
# 答错 + 格式不对	-0.1	final_format_score = -0.1
# 没提取到答案 + 格式不对	-0.1	最低
async def reward_func(args, sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    fmt = SEARCH_R1_CONFIGS["format_score"]
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
    has_tool_call = "<tool_call>" in sample.response
    answer = extract_solution(solution_str)
    answer_correct = bool(answer and em_check(answer, sample.label["ground_truth"]["target"]))

    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["valid_format"] = int(is_valid)
    sample.metadata["has_tool_call"] = int(has_tool_call)
    sample.metadata["answer_correct"] = int(answer_correct)
    sample.metadata["retrieval_correct"] = int(is_retrieval_correct(solution_str, sample.label["ground_truth"]["target"]))

    return score

