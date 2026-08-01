"""Search-R1 generate function — drift-aware variant (chat-template based).

Built on top of generate_with_search_tools_qwen_sft.py. Goal: minimize the
gap between the token sequence the model actually saw at rollout time
(sglang's prefill of turn N+1 sees the result of turn-by-turn-tokenized
text) and the token sequence the optimizer trains on
(`prompt_token_ids + response_token_ids`).

The four drift sources from the audit are addressed here:

1. Chat-template alignment.
   The original script let sglang stop on `</tool_call>`/`</answer>` and
   never appended `<|im_end|>` between assistant and the next user
   tool_response. Training, on the other hand, injected `<|im_end|>` after
   every model turn with `loss_mask=1`, training the model on a token it
   never produced and shifting every subsequent observation token by 1.

   This file delegates ALL conversation wrapping to the upstream
   `tokenizer.apply_chat_template` (Qwen3 default). Glue between turns
   (`<|im_end|>\n`, `<|im_start|>user\n<tool_response>...`,
   `<|im_start|>assistant\n`) gets `loss_mask = 0` and a placeholder
   `rollout_log_prob = 0.0`.

   Caveat: Qwen3's default template re-renders the `<think>...</think>`
   block in canonical form (`<think>\n{R}\n</think>\n\n{C}`). When the
   model already emits canonical format — which is the expectation for a
   model SFT'd on Qwen3 thinking format and conditioned on a properly-
   rendered multi-turn history — re-rendering is a no-op and sglang's
   tokens line up exactly with the rendered text. When the model deviates
   (rare: truncation, parse-failure turns without `</think>`, etc.) the
   incremental render in `build_training_data` detects it via a
   startswith check and falls back to `loss_mask=0` for that whole turn,
   so the IS ratio for downstream turns stays clean. The count of such
   turns is surfaced via `sample.metadata["template_drift_turns"]`.

2. Context window / rolling compression removed.
   Search results are always rendered in full.

3. Decode -> encode self-consistency check.
   At the end of `build_training_data` we decode `prompt_token_ids +
   response_token_ids` to text and re-encode it. If BPE re-merges anywhere
   we surface it via `sample.metadata["bpe_drift"]`.

4. TIS wrapper (`compute_mis_weights_with_cp_no_drift`).
   Defense-in-depth: any position whose `rollout_log_prob == 0.0` is
   forced to mask = 0 before invoking the standard TIS function.
"""

import asyncio
import importlib
import json
import logging
import os
import re
from typing import Any

from qa_em_format_qwen_sft import (  # type: ignore
    _parse_bare_json_tool_call,
    em_check,
    executed_tool_call,
    extract_solution,
    is_retrieval_correct,
    is_valid_sequence,
)

# 8B-specific RELAXED scorer (correct answer always positive; checked before
# the truncated short-circuit). Lives in its own file so the shared
# qa_em_format_qwen_sft.compute_score_em_sft — still used by the 30B run via
# generate_with_search_tools_qwen_sft.py — is left untouched.
from qa_em_format_qwen_sft_8b import compute_score_em_sft  # type: ignore

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


SEARCH_R1_CONFIGS = {
    "max_turns": int(os.environ.get("SEARCH_R1_MAX_TURNS", "5")),
    "topk": int(os.environ.get("SEARCH_R1_TOPK", "3")),
    # On Cluster the retriever runs CPU-faiss + CPU e5 on the main node, so high
    # concurrency can bottleneck rollout; make it tunable (default preserves dev-box behavior).
    "search_concurrency": int(os.environ.get("SEARCH_R1_CONCURRENCY", "256")),
    "search_backend": "local",
    "local": {
        # SEARCH_R1_SEARCH_URL points at the in-cluster retriever (e.g.
        # http://${MASTER_ADDR}:8000/retrieve on Cluster). Default = the dev-box host.
        "search_url": os.environ.get("SEARCH_R1_SEARCH_URL", "http://131.179.168.117:8000/retrieve"),
        "proxy": None,
    },
    "google": {
        "api_key": "your_api_key_here",
        "snippet_only": True,
        "proxy": None,
    },
    "return_logprob": True,
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

# Behavioral instruction injected into the user message. The tool
# signature itself is rendered ONCE by the chat template (system block
# via tools=[SEARCH_TOOL_DESC]); we deliberately do NOT embed another
# copy of <tools>...</tools> here to avoid duplicating ~200 tokens.
TOOL_CALLING_INSTRUCTION = (
    "First conduct reasoning in english every time you try to get new information.\n"
    "After reasoning, if you find you lack some knowledge, you may call a function to help.\n\n"
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
# Prompt parsing (unchanged)
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
# Search functions (unchanged)
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
# Response parsing (unchanged)
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


def _strip_think_block(text: str) -> str:
    """Strip the ``<think>...</think>`` portion from a model turn.

    Mirrors the Qwen3 chat template's in-content think extraction:
    ``text.split('</think>')[-1].lstrip('\\n')`` — i.e. drop everything up
    to and including the LAST ``</think>`` and trim leading newlines from
    what remains. Done identically here so ``build_training_data``'s
    case-(b) ``n_stripped`` token count matches what the template will
    actually consume when rendering a stripped past turn.
    """
    if "</think>" not in text:
        return text
    return text.split("</think>")[-1].lstrip("\n")


def _should_strip_think(args) -> bool:
    """Decide whether historical-turn ``<think>`` blocks are stripped.

    Two control surfaces, checked in order:
      1. ``args.strip_historical_think`` — set via ``--custom-config-path``
         YAML (``strip_historical_think: true``). Most reliable: ``args``
         is passed straight into ``generate()``.
      2. ``$SEARCH_R1_STRIP_THINK`` env var (``1``/``true`` → on). Handy for
         launch scripts that don't carry a custom-config YAML.

    Default OFF (keep think on every turn). With stripping OFF the model
    is trained / rolled out on the *full* think-on-every-turn format,
    which keeps the SFT distribution and rollout distribution identical
    and avoids the "think on the first turn or not?" ambiguity that
    stripping introduces for a model that still hasn't mastered the
    format. Turn ON for RL once the policy reliably emits canonical
    ``<think>...</think>`` and context-length compression matters.
    """
    v = getattr(args, "strip_historical_think", None)
    if v is not None:
        return bool(v)
    env = os.environ.get("SEARCH_R1_STRIP_THINK")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return False


def _truncate_at_first_endoftext(
    tokenizer,
    text: str,
    token_ids: list[int] | None,
    log_probs: list[float] | None,
) -> tuple[str, list[int] | None, list[float] | None]:
    """If <|endoftext|> appears mid-stream, truncate text and the parallel
    token / log_prob streams at the first occurrence so all three views
    stay byte / index consistent.

    This MUST run before `postprocess_predictions` and any other
    semantic parsing of the model output, otherwise downstream code can
    see an `<answer>` / `<tool_call>` block that's actually past an EOT
    leak and won't be present in the saved trajectory.
    """
    eot_str = "<|endoftext|>"
    if not text or eot_str not in text:
        return text, token_ids, log_probs
    text = text[: text.find(eot_str)]
    if token_ids:
        eot_id = tokenizer.convert_tokens_to_ids(eot_str)
        try:
            tok_cut = token_ids.index(eot_id)
        except ValueError:
            tok_cut = None
        if tok_cut is not None:
            token_ids = token_ids[:tok_cut]
            if log_probs:
                log_probs = log_probs[:tok_cut]
    return text, token_ids, log_probs


# ---------------------------------------------------------------------------
# Conversation manager — apply_chat_template-driven, no manual concatenation.
# ---------------------------------------------------------------------------

class ChatTemplateConversationManager:
    """All wrapping (system, user, assistant, tool_response, generation
    prompt) is delegated to the upstream `tokenizer.apply_chat_template`
    (Qwen3 default — no custom Jinja).

    Historical-think compression (Qwen3 multi-turn convention)
    ----------------------------------------------------------
    Qwen3's default template normally strips `<think>...</think>` from
    every assistant turn whose index is < `last_query_index` (i.e. before
    the last *real* user query). In search-R1 multi-turn rollouts the
    only real user message is the original question — every subsequent
    user is a `<tool_response>` wrapper which the template explicitly
    excludes from `last_query_index`. Consequence: with the unmodified
    pipeline ALL prior assistants kept their think blocks, the rollout
    context grew turn-over-turn, and the 4k generation budget started
    being eaten by re-prefilled past thinks instead of new tokens.

    Fix: in `_materialized_messages` we manually strip `<think>...</think>`
    from non-trained assistant turns, and the chat template renders them
    as plain content (no canonical-think wrapper). The flag
    `keep_last_think` selects which turn keeps its think:

      * `False` — rollout (sglang prefill view). Every entry in
        `self.turns` is in the past, so all of them are stripped.
      * `True`  — training view. The LAST turn is the one we train on,
        so its think is preserved (sglang generated those think tokens
        and we want to credit-assign them).

    Past-turn gradient (IS-biased): stripping past thinks changes the KV
    context the historical tool_call tokens were generated under, so the
    rollout/train log-prob ratio is no longer a clean IS weight. We
    accept that bias and still train on past tool_calls (matching
    tau-bench's `_build_training_tensor` design): for each past turn,
    case (b) below sets `loss_mask=1` on the post-`</think>` tail of
    sglang's sampled tokens and uses the corresponding post-think slice
    of sampled log-probs as `rollout_log_prob`. The trade-off: gradient
    signal on every well-aligned turn at the price of biased IS weights
    on past tool_calls. The LAST turn's think is preserved so its body
    aligns 1:1 with sglang (case a, unbiased). The TIS wrapper at the
    bottom of this file additionally masks any position whose
    `rollout_log_prob == 0.0` (context glue / drift) as defense in depth.

    Drift handling for `</think>` reformatting of the trained turn
    --------------------------------------------------------------
    Qwen3's template re-emits the LAST assistant's think as canonical
    `<think>\n{R}\n</think>\n\n{C}`. When the model already produces
    canonical format the re-emit is a no-op and sglang's tokens line up
    with the rendered text. When it doesn't (rare: truncation, a turn
    without `</think>`), the startswith check in `build_training_data`
    falls back to `loss_mask=0` for the whole last turn rather than
    feeding mis-aligned tokens to the optimizer. Counter is exposed via
    `template_drift_turns`.
    """

    def __init__(
        self,
        base_messages: list[dict],
        tokenizer,
        tools: list[dict],
        strip_think: bool = False,
    ):
        self.tokenizer = tokenizer
        self.tools = tools
        self.base_messages = list(base_messages)
        self.turns: list[dict] = []
        self.consistency_report: dict[str, Any] | None = None
        self.template_drift_turns: int = 0
        # When False, every assistant turn keeps its <think> (full format,
        # SFT-matching). When True, historical turns are stripped per the
        # Qwen3 multi-turn convention. See _should_strip_think.
        self.strip_think = strip_think

        self.prompt_text = self._render(self.base_messages, add_generation_prompt=True)

    def _render(self, messages: list[dict], add_generation_prompt: bool) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=self.tools,
        )

    def _materialized_messages(self, keep_last_think: bool) -> list[dict]:
        """Build the message list fed to ``apply_chat_template``.

        Behaviour depends on ``self.strip_think``:

        * ``strip_think=False`` (default) — EVERY assistant turn keeps its
          ``<think>...</think>`` verbatim. The full multi-turn think format
          is preserved end-to-end so the SFT and rollout distributions
          match and there is no ambiguity about whether to think before a
          tool call. ``keep_last_think`` is ignored in this mode.

        * ``strip_think=True`` — historical assistant turns have their
          ``<think>...</think>`` stripped (Qwen3 multi-turn convention;
          the default template's `last_query_index` heuristic otherwise
          keeps every prior think because tool_response user msgs don't
          reset it, bloating prefill each turn). ``keep_last_think``
          controls the most recent turn:
            - False — rollout-side: every turn in ``self.turns`` is
              already historical (the turn being generated isn't appended
              yet), so strip all of them.
            - True  — training-side: the most recent turn IS the trained
              turn, so keep its think for credit assignment; strip
              earlier turns.
        """
        msgs = list(self.base_messages)
        n = len(self.turns)
        for idx, turn in enumerate(self.turns):
            if not self.strip_think:
                content = turn["model_text"]
            else:
                is_trained_last = keep_last_think and (idx == n - 1)
                content = (
                    turn["model_text"] if is_trained_last
                    else _strip_think_block(turn["model_text"])
                )
            msgs.append({"role": "assistant", "content": content})
            if turn["search_result_full"] is not None:
                # Use user-role with already-wrapped <tool_response>...</tool_response>
                # content. The custom template renders this as
                # "<|im_start|>user\n<tool_response>\n...\n</tool_response><|im_end|>\n",
                # byte-equal to what the upstream Qwen3 'tool' role would
                # produce.
                msgs.append({
                    "role": "user",
                    "content": (
                        f"<tool_response>\n{turn['search_result_full']}\n</tool_response>"
                    ),
                })
        return msgs

    # ---- turn accumulation ----
    def add_turn(
        self,
        model_text: str,
        model_token_ids: list[int] | None = None,
        model_log_probs: list[float] | None = None,
        search_result: str | None = None,
    ):
        # Defense-in-depth: callers in `generate()` already run
        # `_truncate_at_first_endoftext` before action parsing, but apply
        # it here too so direct callers (tests, alternate pipelines) stay
        # safe. Idempotent — if the caller already truncated, no-op.
        model_text, model_token_ids, model_log_probs = _truncate_at_first_endoftext(
            self.tokenizer, model_text, model_token_ids, model_log_probs
        )
        self.turns.append(
            {
                "model_text": model_text,
                "model_token_ids": model_token_ids or [],
                "model_log_probs": model_log_probs or [],
                "search_result_full": search_result,
            }
        )

    # ---- text-side context (fed to sglang or used to derive sample.response) ----
    def build_context(
        self,
        add_final_generation_prompt: bool = True,
        keep_last_think: bool = False,
    ) -> str:
        """Render the conversation.

        ``keep_last_think`` should be:
          * False (default) during the rollout loop — every entry in
            ``self.turns`` is a past turn from sglang's POV (the
            about-to-be-generated turn isn't in the list yet), so all
            thinks are stripped, matching the Qwen3 inference convention.
          * True when producing the final ``sample.response`` text at the
            end of a rollout — there the LAST turn is the trained turn
            and its think must be present so the saved response text
            matches the trained token sequence built by
            ``build_training_data``.
        """
        return self._render(
            self._materialized_messages(keep_last_think=keep_last_think),
            add_final_generation_prompt,
        )

    # ---- token-side training data: DIRECT CONCATENATION of sglang's
    # streamed per-turn token ids + fixed pre-tokenized glue. No re-render /
    # re-tokenize of assistant text, so there is no "alignment" to fail. ----
    def build_training_data(
        self,
        return_logprob: bool,
        prompt_token_ids: list[int] | None = None,
    ) -> tuple[list[int], list[int], list[float] | None]:
        """Build (response_token_ids, loss_mask, rollout_log_probs) by DIRECT
        CONCATENATION — replaces the old render → re-tokenize → exact-match-
        per-turn scheme.

        Why the three-case aligner was removed
        --------------------------------------
        The previous implementation rebuilt the trained sequence by rendering
        the whole conversation back to text via ``apply_chat_template`` and
        re-tokenizing it, then demanded the re-tokenized assistant span equal
        sglang's streamed ids per turn (cases a/b) or else zeroed the WHOLE
        turn (case c). Three template/tokenizer divergences broke that match
        on most turns:

          * the Qwen3 template INJECTS an empty ``<think>\\n\\n</think>\\n\\n``
            before the LAST assistant turn when the model didn't emit a
            ``<think>`` (60-77% of turns), so the rendered span carried tokens
            sglang never produced;
          * ``<think>`` canonicalization of non-canonical spacing;
          * BPE boundary re-merges between rendered text and the stream.

        Result: 50-64% of samples ended fully masked (median trainable tokens
        == 0); the policy gradient saw almost none of its own behaviour and
        raw_reward drifted *down*.

        The fix
        -------
        The assistant body of every turn is taken VERBATIM from sglang's
        streamed token ids (``turn["model_token_ids"]``) with ``loss_mask=1``
        and the streamed ``model_log_probs`` as ``rollout_log_prob`` — the
        trained tokens ARE exactly the tokens sglang generated, so the IS
        ratio numerator/denominator refer to the same token. Everything
        between turns (the trailing ``<|im_end|>``, the
        ``<|im_start|>user\\n<tool_response>...</tool_response><|im_end|>\\n``
        block, and the next ``<|im_start|>assistant\\n``) is fixed glue,
        tokenized in isolation, with ``loss_mask=0`` / ``rollout_log_prob=0.0``.
        Every glue boundary marker is an atomic added special token
        (id >= 151644), so isolated tokenization is byte-identical to what
        sglang prefilled — no new drift is introduced.

        ``prompt_token_ids`` already ends with ``<|im_start|>assistant\\n``
        (the generation prompt), so turn 0's body concatenates directly onto
        it; no leading assistant-open glue precedes turn 0.

        strip_think faithfulness
        ------------------------
        With ``strip_think`` ON, sglang prefilled turn i+1 over PAST
        assistants whose ``<think>`` was stripped, so past-turn bodies are
        sliced to their post-``</think>`` tail (same slice the old case (b)
        computed) to keep the trained prefix faithful; the LAST/trained turn
        always uses its full streamed body. With ``strip_think`` OFF
        (recommended) every turn uses its full body and the concatenation is
        byte-exact to sglang's prefill — no IS bias on any turn.

        The trailing ``<|im_end|>`` of each turn is emitted with
        ``loss_mask=0`` / ``rollout_log_prob=0.0``: sglang stops on
        ``</tool_call>``/``</answer>`` and does not itself generate
        ``<|im_end|>``, and the no_drift TIS wrapper force-masks any position
        whose rollout_log_prob == 0.0 anyway, so marking it trainable would be
        a silent inconsistency.
        """
        response_token_ids: list[int] = []
        loss_mask: list[int] = []
        rollout_log_probs: list[float] | None = [] if return_logprob else None

        n = len(self.turns)
        if n == 0:
            return response_token_ids, loss_mask, rollout_log_probs

        endoftext_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        def _tok(text: str) -> list[int]:
            return self.tokenizer(text, add_special_tokens=False)["input_ids"]

        # Fixed, context-independent glue. Every marker (<|im_start|>=151644,
        # <|im_end|>=151645, <tool_response>=151665, ...) is an atomic added
        # special token that always splits the BPE stream, so isolated
        # tokenization == what sglang prefilled.
        nl_ids = _tok("\n")                          # the lone "\n" after <|im_end|>
        user_open_ids = _tok("<|im_start|>user\n")
        asst_open_ids = _tok("<|im_start|>assistant\n")

        def _emit(ids, mask, logprobs=None):
            for k, tid in enumerate(ids):
                response_token_ids.append(tid)
                loss_mask.append(mask)
                if return_logprob:
                    lp = logprobs[k] if (logprobs is not None and k < len(logprobs)) else 0.0
                    rollout_log_probs.append(lp)

        for turn_idx, turn in enumerate(self.turns):
            body = list(turn["model_token_ids"])
            blps = list(turn["model_log_probs"]) if return_logprob else None
            model_text = turn["model_text"]

            # Drop any natural-EOS tokens sglang may have leaked at the end of
            # the stream; the turn terminator <|im_end|> is appended explicitly.
            while body and body[-1] in (endoftext_id, im_end_id):
                body.pop()
                if blps is not None:
                    blps.pop()

            # strip_think compression faithfulness: for PAST turns under
            # strip_think, sglang prefilled the stripped (post-</think>) tail,
            # so train on that same tail. The last turn always keeps full body.
            if self.strip_think and turn_idx < n - 1 and "</think>" in model_text:
                tail = model_text.split("</think>")[-1].lstrip("\n")
                kept_byte_len = len(tail)
                stripped_prefix_text = (
                    model_text[: len(model_text) - kept_byte_len]
                    if kept_byte_len > 0
                    else model_text
                )
                n_stripped = len(_tok(stripped_prefix_text))
                if n_stripped <= len(body):
                    body = body[n_stripped:]
                    if blps is not None:
                        blps = blps[n_stripped:]

            # (a) assistant body — trained, with sglang's own logprobs.
            _emit(body, 1, blps)
            # (b) turn terminator <|im_end|> — emitted, not trained.
            _emit([im_end_id], 0)

            # (c) inter-turn glue (only when another turn follows).
            if turn_idx < n - 1:
                _emit(nl_ids, 0)  # the "\n" after <|im_end|>
                if turn["search_result_full"] is not None:
                    tool_text = (
                        f"<tool_response>\n{turn['search_result_full']}\n</tool_response>"
                    )
                    _emit(user_open_ids, 0)
                    _emit(_tok(tool_text), 0)
                    _emit([im_end_id], 0)
                    _emit(nl_ids, 0)
                _emit(asst_open_ids, 0)

        return response_token_ids, loss_mask, rollout_log_probs

    def get_stats(self) -> dict:
        num_turns = len(self.turns)
        num_search_turns = sum(
            1 for t in self.turns if t["search_result_full"] is not None
        )
        return {
            "num_turns": num_turns,
            "obs_full": num_search_turns,
            "obs_compressed": 0,  # always 0 — kept for wandb schema parity
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

    raw_prompt: str = sample.prompt
    initial_messages = parse_prompt_to_messages(raw_prompt)
    if not initial_messages:
        logger.warning("Failed to parse prompt into messages, falling back to raw text.")
        initial_messages = [{"role": "user", "content": raw_prompt}]

    initial_messages = clean_instruction_in_messages(initial_messages)

    manager = ChatTemplateConversationManager(
        initial_messages, state.tokenizer, tools,
        strip_think=_should_strip_think(args),
    )
    prompt_text = manager.prompt_text
    prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Track why the rollout loop ended. One of:
    #   "answer"     — model emitted <answer> (success)
    #   "length"     — sglang hit max_response_len mid-turn
    #   "abort"      — sglang aborted (already returned earlier)
    #   "parse_fail" — model emitted neither <tool_call> nor <answer>
    #   "max_turns"  — exhausted max_turns without reaching <answer>
    # Anything but "answer" is treated as TRUNCATED at the end.
    exit_reason: str | None = None
    last_finish_reason = None

    action_stops = ["</tool_call>", "</answer>"]
    turn_sampling = dict(sampling_params)
    turn_sampling["stop"] = list(turn_sampling.get("stop") or []) + action_stops
    turn_sampling["no_stop_trim"] = True

    # Optional, OPT-IN via --enforce-total-response-budget (default OFF, so existing
    # scripts keep their original per-turn behavior unchanged): treat max_new_tokens as a
    # TOTAL response budget across the search turns by decrementing it each turn. Without
    # this, the loop applies max_new_tokens PER TURN, so total length can reach
    # max_turns * max_new_tokens. The response-length curriculum needs this to bind length.
    enforce_total_budget = getattr(args, "enforce_total_response_budget", False)
    total_budget = int(turn_sampling.get("max_new_tokens") or 0) if enforce_total_budget else 0
    tokens_used = 0

    for _turn_idx in range(config["max_turns"]):
        if total_budget > 0:
            remaining = total_budget - tokens_used
            if remaining <= 0:
                exit_reason = "length"
                break
            turn_sampling["max_new_tokens"] = remaining
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

        # Sanitize the model output BEFORE any semantic parsing: if sglang
        # leaked <|endoftext|> mid-stream, drop everything after it in
        # text, tokens, and log_probs simultaneously. Otherwise
        # `postprocess_predictions` could see an <answer> / <tool_call>
        # past the EOT that won't survive into the saved trajectory,
        # producing a "phantom answer" sample that's marked COMPLETED but
        # has no parseable answer in sample.response.
        cur_response, cur_token_ids, cur_log_probs = _truncate_at_first_endoftext(
            state.tokenizer, cur_response, cur_token_ids, cur_log_probs
        )
        tokens_used += len(cur_token_ids)

        if last_finish_reason == "length":
            manager.add_turn(
                cur_response, cur_token_ids, cur_log_probs, search_result=None
            )
            exit_reason = "length"
            break

        action, content = postprocess_predictions(cur_response)

        if action == "answer":
            manager.add_turn(
                cur_response, cur_token_ids, cur_log_probs, search_result=None
            )
            exit_reason = "answer"
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
            # No break — model gets to consume the tool_response next turn.
        else:
            # Fix #5: parse-failure (neither <tool_call> nor <answer>).
            # Don't open another assistant retry — that wastes rollout
            # budget and produces weird back-to-back assistant blocks.
            # Terminate now and mark as TRUNCATED downstream.
            manager.add_turn(
                cur_response, cur_token_ids, cur_log_probs, search_result=None
            )
            exit_reason = "parse_fail"
            break

    # Fix #2: max_turns reached without an <answer> (last action was
    # likely "search"). last_finish_reason will be "stop" because the
    # model emitted </tool_call>, but semantically the trajectory is
    # incomplete and should NOT count as COMPLETED.
    if exit_reason is None:
        exit_reason = "max_turns"

    response_token_ids, loss_mask, rollout_log_probs = manager.build_training_data(
        return_logprob, prompt_token_ids=prompt_token_ids
    )

    stats = manager.get_stats()
    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["num_turns"] = stats["num_turns"]
    sample.metadata["template_drift_turns"] = manager.template_drift_turns
    if manager.consistency_report is not None:
        sample.metadata["bpe_drift"] = manager.consistency_report.get("mismatch", -1)
        sample.metadata["bpe_drift_len_diff"] = manager.consistency_report.get(
            "len_diff", 0
        )

    # For the final response text we want the SAME rendering as
    # build_training_data (last turn keeps think), so sample.response is
    # the text view of the trained token sequence.
    full_context = manager.build_context(
        add_final_generation_prompt=False, keep_last_think=True
    )
    response = full_context[len(prompt_text):]

    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    sample.prompt = prompt_text

    if return_logprob:
        sample.rollout_log_probs = rollout_log_probs if rollout_log_probs else None

    sample.metadata["exit_reason"] = exit_reason
    if exit_reason == "answer":
        sample.status = Sample.Status.COMPLETED
    else:
        # length / parse_fail / max_turns all map to TRUNCATED so the
        # reward function's truncated-shaping branch fires and the
        # completed-rate metric isn't polluted. (abort returned earlier.)
        sample.status = Sample.Status.TRUNCATED

    return sample


# ---------------------------------------------------------------------------
# Reward function — identical scoring; just emits one extra metric.
# ---------------------------------------------------------------------------

async def reward_func(args, sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    solution_str = sample.prompt + sample.response

    status_str = getattr(sample.status, "value", None) or str(sample.status or "")

    score = compute_score_em_sft(
        solution_str=solution_str,
        ground_truth=sample.label["ground_truth"],
        status=status_str,
        response=sample.response,
    )

    is_valid, _ = is_valid_sequence(solution_str)
    has_tool_call_loose = (
        "<tool_call>" in sample.response
        or _parse_bare_json_tool_call(sample.response)[0] is not None
    )
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


# ---------------------------------------------------------------------------
# Eval generate wrapper — search rollout that ALSO assigns the task reward.
# ---------------------------------------------------------------------------
#
# Why this exists (OPD eval). For a normal search-R1 RL run the launch script
# sets `--custom-rm-path generate_with_search_tools_qwen_sft_no_drift.reward_func`,
# so `generate_and_rm` (in sglang_rollout.py) computes the EM task score via
# `async_rm` for both training rollouts and eval. On-policy distillation runs
# instead point `--custom-rm-path` at the teacher reward
# (`slime.rollout.on_policy_distillation.reward_func`, which returns teacher
# token logprobs / a 0.0 task reward). Because `async_rm` ALWAYS prefers
# `args.custom_rm_path` when it is set, eval in an OPD run would otherwise score
# every trajectory with the teacher reward (and need the teacher server up at
# eval time) — useless as a task-accuracy metric.
#
# `generate_and_rm` only calls `async_rm` when `sample.reward is None`. So this
# wrapper runs the ordinary search rollout via `generate(...)` and then fills in
# `sample.reward` with the real EM score from `reward_func(...)`. With the reward
# already set, `async_rm` (and therefore the OPD teacher reward) is skipped, and
# eval reports genuine search task accuracy without contacting the teacher.
#
# Wire it ONLY for eval via the eval dataset config's
# `custom_generate_function_path` (see examples/on_policy_distillation/
# eval_search.yaml); training rollout keeps using the plain `generate`.
async def generate_eval(args, sample: Sample, sampling_params) -> Sample:
    sample = await generate(args, sample, sampling_params)
    if sample.status != Sample.Status.ABORTED and sample.reward is None:
        sample.reward = await reward_func(args, sample)
    return sample


# ---------------------------------------------------------------------------
# TIS wrapper — drift fix #4.
# ---------------------------------------------------------------------------
#
# `compute_mis_weights_with_cp_no_drift` wraps the standard TIS function and
# zeros out any position whose `rollout_log_prob == 0.0`. Those are by
# construction the system-injected glue tokens (we set them to 0.0 in
# `build_training_data`); even though `loss_mask` is already 0 there, this
# wrapper guards against:
#   - upstream code paths that re-derive a loss_mask of all 1s,
#   - any future change to add_turn / build_training_data that forgets to
#     keep loss_mask aligned with rollout_log_probs.
#
# CP-correctness:
#   `loss_masks` arrives from the caller as FULL-length response masks,
#   while `rollout_log_probs` and `train_log_probs` arrive CP-SLICED
#   (see actor.py:_get_rollout_data + slice_log_prob_with_cp). Element-
#   wise masking on the raw inputs would shape-mismatch under CP > 1.
#   The fix: gather rollout_log_probs to full length BEFORE building the
#   masked loss_masks, then hand the (already-masked) loss_masks plus the
#   ORIGINAL CP-sliced log-probs to the upstream wrapper, which does its
#   own all_gather internally. This double-gather is cheap relative to
#   forward/backward and keeps the wrapper purely additive.
#
# Wire it via the script:
#   --custom-tis-function-path \
#     generate_with_search_tools_qwen_sft_no_drift.compute_mis_weights_with_cp_no_drift
# (PYTHONPATH already contains the search-r1 directory, see the launch
# script.)


def _load_orig_compute_mis_weights_with_cp():
    """Lazy import so this file stays importable without Megatron."""
    mod = importlib.import_module("examples.train_infer_mismatch_helper.mis")
    return mod.compute_mis_weights_with_cp


def compute_mis_weights_with_cp_no_drift(
    args,
    *,
    pg_loss,
    train_log_probs,
    rollout_log_probs,
    loss_masks,
    total_lengths,
    response_lengths,
    **kwargs: Any,
):
    # Gather CP-sliced rollout_log_probs to full response length so we
    # can build a same-shape mask against full-length loss_masks.
    from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

    full_rollout_log_probs = [
        all_gather_with_cp(log_prob, total_length, response_length)
        for log_prob, total_length, response_length in zip(
            rollout_log_probs, total_lengths, response_lengths, strict=False
        )
    ]

    masked_loss_masks = []
    for full_rlp, lm in zip(full_rollout_log_probs, loss_masks, strict=False):
        # full_rlp: shape [response_length], same as lm.
        injected = (full_rlp == 0.0)
        new_lm = lm.float() * (~injected).float()
        masked_loss_masks.append(new_lm.to(lm.dtype))

    orig = _load_orig_compute_mis_weights_with_cp()
    return orig(
        args,
        pg_loss=pg_loss,
        train_log_probs=train_log_probs,           # CP-sliced; upstream gathers
        rollout_log_probs=rollout_log_probs,        # CP-sliced; upstream gathers
        loss_masks=masked_loss_masks,               # FULL length, already masked
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        **kwargs,
    )
