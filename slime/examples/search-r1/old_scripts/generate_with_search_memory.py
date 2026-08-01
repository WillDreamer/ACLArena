# Memory Context Window variant of generate_with_search.py
# Within the context window (last K turns), observations are kept in full.
# Outside the window, observations are compressed (truncated docs + limited chars).
# Model-generated content is always kept in full regardless of window position.

import asyncio
import logging
import re

from qa_em_format import compute_score_em

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
        "search_url": "http://131.179.168.118:8000/retrieve",
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
        raise ValueError(f"Unknown search backend: {backend}. Must be either 'local' or 'google'.")

    return _passages2string(result)


def postprocess_responses(resp: str) -> str:
    return (
        resp.split("</search>")[0] + "</search>"
        if "</search>" in resp
        else resp.split("</answer>")[0] + "</answer>" if "</answer>" in resp else resp
    )


def postprocess_predictions(prediction: str):
    pattern = r"<(search|answer)>(.*?)</\1>"
    match = re.search(pattern, prediction, re.DOTALL)
    if match:
        content = match.group(2).strip()
        action = match.group(1)
    else:
        content = ""
        action = None
    return action, content


async def execute_predictions(prediction: str) -> str:
    action, content = postprocess_predictions(prediction)

    if action == "search":
        search_query = content
        async with SEMAPHORE:
            search_results = await search(search_query)
        next_obs = f"\n\n<information>{search_results.strip()}</information>\n\n"
        done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        next_obs = (
            "\nMy previous action is invalid. "
            "If I want to search, I should put the query between <search> and </search>. "
            "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
            "Let me try again.\n"
        )
        done = False

    return next_obs, done


def compress_observation(obs_text: str, max_docs: int, max_chars_per_doc: int) -> str:
    """Compress an <information>...</information> block by keeping fewer docs with truncated text."""
    info_match = re.search(r"<information>(.*?)</information>", obs_text, re.DOTALL)
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
    return f"\n\n<information>{compressed_inner}\n</information>\n\n"


class ContextWindowManager:
    """Manages multi-turn context with a sliding compression window.

    Tracks per-turn model outputs and observations. When building the context
    string for the next SGLang call, observations outside the context window
    (older than the last K turns) are compressed.
    """

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
            "obs_compressed": compress_observation(observation, self.max_docs, self.max_chars)
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

    def build_training_data(self, return_logprob: bool) -> tuple[list[int], list[int], list[float] | None]:
        """Build final response_token_ids, loss_mask, and rollout_log_probs.

        Model output tokens use the exact IDs from SGLang.
        Observation tokens are re-tokenized based on their final window position.
        """
        response_token_ids = []
        loss_mask = []
        rollout_log_probs = [] if return_logprob else None

        for i, turn in enumerate(self.turns):
            response_token_ids += turn["model_token_ids"]
            loss_mask += [1] * len(turn["model_token_ids"])
            if return_logprob:
                rollout_log_probs += turn["model_log_probs"]

            obs_text = self._get_obs_text(i)
            if obs_text:
                obs_tokens = self.tokenizer(obs_text, add_special_tokens=False)["input_ids"]
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
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    config = SEARCH_R1_CONFIGS
    return_logprob = config["return_logprob"]

    prompt_text = sample.prompt
    prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    manager = ContextWindowManager(prompt_text, state.tokenizer, config)

    last_finish_reason = None

    for _turn_idx in range(config["max_turns"]):
        context = manager.build_context()
        payload = {
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
            cur_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            cur_response = postprocess_responses(cur_response)
            cur_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]
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

    response_token_ids, loss_mask, rollout_log_probs = manager.build_training_data(return_logprob)

    stats = manager.get_stats()
    logger.debug(
        f"ContextWindow stats: turns={stats['num_turns']}, "
        f"obs_full={stats['obs_full']}, obs_compressed={stats['obs_compressed']}"
    )

    # Build the full response text from the final (compressed) context view
    full_context = manager.build_context()
    response = full_context[len(prompt_text):]

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


async def reward_func(args, sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    score = compute_score_em(
        solution_str=sample.prompt + sample.response,
        ground_truth=sample.label["ground_truth"],
        format_score=SEARCH_R1_CONFIGS["format_score"],
    )

    return score
