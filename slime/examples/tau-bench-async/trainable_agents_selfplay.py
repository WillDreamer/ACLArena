"""
Self-play variant of trainable_agents.py.

Replaces the external LLM user simulator (litellm/GLM API) with the same
SGLang rollout engine that serves the policy model.  RESPOND actions now
go through an async SGLang call instead of a blocking HTTP call to an
external service, so the rollout engine is no longer idle while waiting
for the user simulator.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from openai_tool_adapter import create_openai_adapter
from tau_bench.agents.tool_calling_agent import RESPOND_ACTION_NAME, ToolCallingAgent
from tau_bench.types import Action, EnvInfo, EnvResponse, RunConfig
from transformers import AutoTokenizer

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post

logger = logging.getLogger(__name__)

STOP_SIGNAL = "###STOP###"


class ContextWindowManager:
    """
    Sliding window over multi-turn conversation history.

    Keeps ALL messages but compresses old observation content (tool/user responses)
    outside the window to prevent context overflow while ensuring rollout and training
    see exactly the same token sequence structure.

    A "turn" is one agent→env/user exchange, i.e.:
      (assistant message, tool/user response message)
    The system prompt and first user message (initial observation) are always
    included in full regardless of k.

    Observations older than the last k turns are truncated to max_obs_chars
    characters. If the total token count still exceeds max_tokens, the oldest
    out-of-window observations are further aggressively truncated until it fits.

    IMPORTANT: both rollout inference and training token-delta computation must
    call all_messages() with the same tokenizer so they see the identical token
    sequence, keeping rollout_log_probs and train_log_probs aligned for TIS.
    """

    def __init__(
        self,
        system_msg: dict,
        initial_user_msg: dict,
        k: int = 10,
        max_obs_chars: int = 300,
        max_tokens: int = 30000,
    ):
        self.k = k
        self.max_obs_chars = max_obs_chars
        self.max_tokens = max_tokens
        # Fixed prefix: [system, initial_user] — never compressed
        self.prefix: list[dict] = [system_msg, initial_user_msg]
        # All messages after the prefix
        self.history: list[dict] = []
        # Cached compressed view, invalidated on each append
        self._cached_messages: list[dict] | None = None

    def append(self, msg: dict) -> None:
        """Append a single message to history and invalidate the cache."""
        self.history.append(msg)
        self._cached_messages = None

    def all_messages(self, tokenizer=None) -> list[dict]:
        """Return prefix + history with out-of-window observations compressed.

        This is the single consistent view used for BOTH rollout inference and
        training token-delta computation. When tokenizer is provided, further
        truncates old observations until the total token count is below max_tokens.

        Assistant messages are never compressed.
        """
        if self._cached_messages is not None and tokenizer is None:
            return self._cached_messages

        messages = self.prefix + self._compressed_history()

        if tokenizer is not None:
            messages = self._apply_token_budget(messages, tokenizer)
            self._cached_messages = messages
        return messages

    def _compressed_history(self) -> list[dict]:
        """Apply soft window-based compression: truncate old tool/user obs to max_obs_chars."""
        assistant_indices = [i for i, m in enumerate(self.history) if m["role"] == "assistant"]

        if len(assistant_indices) <= self.k:
            return list(self.history)

        window_start_idx = assistant_indices[len(assistant_indices) - self.k]

        result = []
        for i, msg in enumerate(self.history):
            if i < window_start_idx and msg["role"] in ("tool", "user"):
                content = msg["content"]
                if len(content) > self.max_obs_chars:
                    content = content[:self.max_obs_chars] + "..."
                result.append({**msg, "content": content})
            else:
                result.append(msg)
        return result

    def _apply_token_budget(self, messages: list[dict], tokenizer) -> list[dict]:
        """Hard budget enforcement: further shrink out-of-window obs until token count fits.

        Iteratively halves the character budget for the oldest compressible messages
        until the rendered token count is below self.max_tokens.  Assistant messages
        and the prefix are never modified.
        """
        def count_tokens(msgs: list[dict]) -> int:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            return len(tokenizer.encode(text, add_special_tokens=False))

        if count_tokens(messages) <= self.max_tokens:
            return messages

        # Find indices of compressible messages (out-of-window tool/user obs)
        # prefix has len(self.prefix) entries; history entries start after that
        prefix_len = len(self.prefix)
        assistant_indices_in_msgs = [
            i for i, m in enumerate(messages[prefix_len:]) if m["role"] == "assistant"
        ]
        if len(assistant_indices_in_msgs) > self.k:
            window_start = prefix_len + assistant_indices_in_msgs[len(assistant_indices_in_msgs) - self.k]
        else:
            window_start = prefix_len

        compressible = [
            i for i in range(prefix_len, window_start)
            if messages[i]["role"] in ("tool", "user")
        ]

        if not compressible:
            logger.warning(
                "Context still exceeds max_tokens=%d after soft compression and no "
                "further compressible messages remain. Returning as-is.",
                self.max_tokens,
            )
            return messages

        messages = [m.copy() for m in messages]
        char_budget = self.max_obs_chars

        while count_tokens(messages) > self.max_tokens and char_budget > 20:
            char_budget = max(20, char_budget // 2)
            for i in compressible:
                content = messages[i]["content"]
                if len(content) > char_budget:
                    messages[i]["content"] = content[:char_budget] + "..."

        if count_tokens(messages) > self.max_tokens:
            logger.warning(
                "Context exceeds max_tokens=%d even after aggressive compression "
                "(char_budget=%d). This turn will likely be rejected by SGLang.",
                self.max_tokens,
                char_budget,
            )

        return messages


class Status(Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


@dataclass
class InteractionResult:
    prompt: str
    reward: float
    messages: list[dict[str, Any]]
    info: dict[str, Any]
    response: str = ""
    loss_mask: list[int] | None = None
    tokens: int | None = None
    status: Status = Status.COMPLETED
    rollout_log_probs: list[float] | None = None


# ---------------------------------------------------------------------------
# Async SGLang-backed user simulator
# ---------------------------------------------------------------------------

class SGLangUserSimulationEnv:
    """
    Drop-in async replacement for LLMUserSimulationEnv.

    Uses the same SGLang router as the policy model so both the agent and
    the user simulator share the same inference engine, eliminating the
    external API round-trip.

    Conversation format (mirrors LLMUserSimulationEnv):
      - system: user-simulator system prompt
      - user:   agent message (the "other side" of the conversation)
      - assistant: user simulator reply
      - user:   next agent message
      - ...
    """

    def __init__(self, tokenizer: AutoTokenizer, url: str, sampling_params: dict[str, Any], k: int = 6) -> None:
        self.tokenizer = tokenizer
        self.url = url
        # Use a separate, calmer sampling config: shorter responses, lower temperature.
        self.sampling_params = {
            **sampling_params,
            "temperature": 0.7,
            "max_new_tokens": 256,
        }
        self.k = k  # keep last k user/assistant turn pairs in history
        self._system_msg: dict[str, Any] | None = None
        self.messages: list[dict[str, Any]] = []

    def _build_system_prompt(self, instruction: str | None) -> str:
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n") if instruction is not None else ""
        )
        return (
            f"You are a user interacting with an agent.{instruction_display}\n"
            "Rules:\n"
            "- Just generate one line at a time to simulate the user's message.\n"
            "- Do not give away all the instruction at once. Only provide the information "
            "that is necessary for the current step.\n"
            "- Do not hallucinate information that is not provided in the instruction.\n"
            "- Only if the instruction goal is satisfied and THE EXECUTION IS CONFIRMED "
            f"generate '{STOP_SIGNAL}' as a standalone message without anything else.\n"
            "- Do not repeat the exact instruction. Use your own words.\n"
            "- Try to make the conversation as natural as possible."
        )

    async def reset(self, instruction: str | None = None) -> str:
        self._system_msg = {"role": "system", "content": self._build_system_prompt(instruction)}
        self.messages = [{"role": "user", "content": "Hi! How can I help you today?"}]
        return await self._generate()

    async def step(self, agent_message: str) -> str:
        """Called each time the agent sends a message to the user."""
        # Strip <think>...</think> blocks from agent message to avoid ballooning context
        import re
        clean_message = re.sub(r"<think>.*?</think>", "", agent_message, flags=re.DOTALL).strip()
        self.messages.append({"role": "user", "content": clean_message or agent_message})
        return await self._generate()

    def _windowed_messages(self) -> list[dict[str, Any]]:
        """Return system msg + last k*2 messages (k user turns + k assistant turns)."""
        cutoff = self.k * 2
        history = self.messages[-cutoff:] if len(self.messages) > cutoff else self.messages
        return [self._system_msg] + history

    async def _generate(self) -> str:
        text = self.tokenizer.apply_chat_template(
            self._windowed_messages(),
            tokenize=False,
            add_generation_prompt=True,
            # No tools — user sim is plain chat
        )
        output = await post(self.url, {"text": text, "sampling_params": self.sampling_params})

        if output.get("meta_info", {}).get("finish_reason", {}).get("type") == "abort":
            logger.warning("SGLang aborted user-sim generation; returning STOP signal.")
            return STOP_SIGNAL

        response = output["text"]
        if response.endswith("<|im_end|>"):
            response = response[:-10]
        response = response.strip()
        self.messages.append({"role": "assistant", "content": response})
        return response

    def get_total_cost(self) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Self-play trainable agent mixin
# ---------------------------------------------------------------------------

TOOL_INSTRUCTION = (
    " At each turn, you are allowed to call one or no function to assist "
    "with task execution using <tools></tools> XML tags.\n"
    "YOU MUST EXECUTE TOOLS TO MAKE ANY MODIFICATIONS OR CANCELLATIONS. "
    "Each tool call leads to a message returned by the system.\n"
    "NEVER confirm execution to the user without seeing confirmation "
    "from the tool system.\n"
)


class SelfPlayAgentMixin:
    """
    Replaces TrainableAgentMixin's asolve() with a version that uses
    SGLangUserSimulationEnv instead of the external LLM user simulator.

    Only RESPOND actions hit the user sim (async SGLang call).
    Tool actions call env.step() directly — they are pure Python DB
    operations with no I/O, so they complete in microseconds.
    """

    def _reformulate_tool_call(self, text: str) -> str:
        return text.replace(
            "You may call one or more functions to assist with the user query.",
            TOOL_INSTRUCTION,
        )

    async def _call_llm(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await post(url, payload)

    def _parse_tool(self, response: str) -> dict[str, Any]:
        return self.openai_adapter.parse_response_to_openai_format(response)

    def _get_token_delta(
        self, tokenizer: AutoTokenizer, messages: list[dict]
    ) -> tuple[list[int], list[int]]:
        curr = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        if messages[-1]["role"] == "assistant":
            prev = tokenizer.apply_chat_template(
                messages[:-1], add_generation_prompt=True, tokenize=False
            )
            new_tokens = tokenizer.encode(curr[len(prev):], add_special_tokens=False)
            return new_tokens, [1] * len(new_tokens)
        else:
            prev = tokenizer.apply_chat_template(
                messages[:-1], add_generation_prompt=False, tokenize=False
            )
            new_tokens = tokenizer.encode(curr[len(prev):], add_special_tokens=False)
            return new_tokens, [0] * len(new_tokens)

    def _build_final_result(
        self,
        res: InteractionResult,
        total_reward: float,
        info: dict[str, Any],
        messages: list[dict[str, Any]],
        loss_masks: list[int],
        prompt_token_ids: list[int],
        response_token_ids: list[int],
        rollout_log_probs: list[float] | None = None,
        agent_turns: int = 0,
        tool_call_turns: int = 0,
    ) -> InteractionResult:
        res.reward = total_reward
        info["has_tool_call"] = tool_call_turns > 0
        info["num_turns"] = agent_turns
        info["tool_call_turn_frac"] = tool_call_turns / agent_turns if agent_turns > 0 else 0.0
        res.info = info
        res.messages = messages
        res.loss_mask = loss_masks
        res.tokens = prompt_token_ids + response_token_ids
        res.response = "".join(
            [msg.get("content", "") for msg in messages if msg["role"] == "assistant"]
        )
        res.response_length = len(loss_masks)
        res.rollout_log_probs = rollout_log_probs  # always a list, even if empty (never None)
        return res

    async def asolve(
        self,
        env,
        rollout_args: dict[str, Any],
        sampling_params: dict[str, Any],
        task_index: int | None = None,
        max_num_steps: int = 30,
        context_window_k: int = 10,
    ) -> InteractionResult:
        state = GenerateState(rollout_args)
        agent_url = (
            f"http://{rollout_args.sglang_router_ip}:"
            f"{rollout_args.sglang_router_port}/generate"
        )

        # --- Reset env state without triggering the external user sim ---
        if task_index is not None:
            env.task_index = task_index
        env.data = env.data_load_func()
        env.task = env.tasks[env.task_index]
        env.actions = []

        # --- Create async SGLang user sim and get initial observation ---
        user_sim = SGLangUserSimulationEnv(
            tokenizer=state.tokenizer,
            url=agent_url,
            sampling_params=sampling_params,
        )
        initial_obs = await user_sim.reset(instruction=env.task.instruction)

        # Build initial agent conversation with sliding window manager.
        # max_tokens budget = model context length minus response headroom, so the
        # prompt fed to SGLang never exceeds the model's context window.
        _max_response = getattr(rollout_args, "rollout_max_response_len", 1024)
        _model_ctx = getattr(rollout_args, "rollout_max_context_len", None) or 32768
        ctx = ContextWindowManager(
            system_msg={"role": "system", "content": self.wiki},
            initial_user_msg={"role": "user", "content": initial_obs},
            k=context_window_k,
            max_tokens=_model_ctx - _max_response,
        )

        prompt_text = state.tokenizer.apply_chat_template(
            ctx.all_messages(), tokenize=False, add_generation_prompt=True, tools=self.tools_info
        )
        prompt_text = self._reformulate_tool_call(prompt_text)
        prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        loss_masks: list[int] = []
        response_token_ids: list[int] = []
        rollout_log_probs: list[float] = []
        total_reward = 0.0
        info: dict[str, Any] = EnvInfo(task=env.task, source="user").model_dump()
        env_response: EnvResponse | None = None
        agent_turns = 0
        tool_call_turns = 0

        res = InteractionResult(prompt=prompt_text, reward=0, messages=[], info={})

        for _ in range(max_num_steps):
            # --- Policy model (agent) generates next action ---
            # all_messages(tokenizer) applies both soft window compression AND hard
            # token-budget enforcement, returning a single consistent view used for
            # both rollout inference and token-delta computation (TIS alignment).
            current_messages = ctx.all_messages(state.tokenizer)
            text_input = state.tokenizer.apply_chat_template(
                current_messages, tokenize=False, add_generation_prompt=True, tools=self.tools_info
            )
            text_input = self._reformulate_tool_call(text_input)
            output = await self._call_llm(agent_url, {"text": text_input, "sampling_params": sampling_params, "return_logprob": True})

            if output["meta_info"]["finish_reason"]["type"] == "abort":
                res.status = Status.ABORTED
                return self._build_final_result(
                    res, total_reward, info, ctx.all_messages(), loss_masks, prompt_token_ids, response_token_ids, rollout_log_probs,
                    agent_turns=agent_turns, tool_call_turns=tool_call_turns,
                )

            response = output["text"]
            if response.endswith("<|im_end|>"):
                response = response[:-10]

            # Parse tool calls
            try:
                openai_result = self._parse_tool(response)
                if not openai_result["success"]:
                    logger.warning(f"Tool parse failed: {openai_result['error']}")
                    res.status = Status.ABORTED
                    return self._build_final_result(
                        res, total_reward, info, ctx.all_messages(), loss_masks, prompt_token_ids, response_token_ids, rollout_log_probs,
                        agent_turns=agent_turns, tool_call_turns=tool_call_turns,
                    )
                parsed = openai_result["parsed_result"]
            except Exception as e:
                logger.warning(f"Tool parse exception: {e}")
                res.status = Status.ABORTED
                return self._build_final_result(
                    res, total_reward, info, ctx.all_messages(), loss_masks, prompt_token_ids, response_token_ids, rollout_log_probs,
                    agent_turns=agent_turns, tool_call_turns=tool_call_turns,
                )

            # Track agent tokens (loss_mask=1). Use all_messages(tokenizer) so the
            # token delta is computed on the same compressed context used for rollout.
            ctx.append({"role": "assistant", "content": response})
            messages_with_assistant = ctx.all_messages(state.tokenizer)
            agent_token_ids, agent_loss_mask = self._get_token_delta(state.tokenizer, messages_with_assistant)
            response_token_ids.extend(agent_token_ids)
            loss_masks.extend(agent_loss_mask)
            # Collect agent log probs for TIS; pad to match token count if SGLang didn't return them
            step_log_probs = (
                [item[0] for item in output["meta_info"].get("output_token_logprobs", [])]
                if output.get("meta_info") else []
            )
            if len(step_log_probs) < len(agent_token_ids):
                step_log_probs += [0.0] * (len(agent_token_ids) - len(step_log_probs))
            rollout_log_probs.extend(step_log_probs[:len(agent_token_ids)])

            # Build action
            agent_content, calls = parsed["normal_text"], parsed["calls"]
            agent_turns += 1
            if calls:
                tool_call_turns += 1
                if len(calls) > 1:
                    logger.debug("Multiple tool calls; using first.")
                tool_call = calls[0]
                params = json.loads(tool_call["parameters"])
                action = Action(name=tool_call["name"], kwargs=params if isinstance(params, dict) else {})
            else:
                action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": agent_content})

            # --- Execute action ---
            if action.name == RESPOND_ACTION_NAME:
                # Self-play: call SGLang user sim asynchronously
                try:
                    user_response = await user_sim.step(action.kwargs["content"])
                except Exception as e:
                    logger.warning(f"User sim step failed: {e}")
                    res.status = Status.ABORTED
                    return self._build_final_result(
                        res, total_reward, info, ctx.all_messages(), loss_masks, prompt_token_ids, response_token_ids, rollout_log_probs,
                        agent_turns=agent_turns, tool_call_turns=tool_call_turns,
                    )

                done = STOP_SIGNAL in user_response
                reward = 0.0
                env_info = EnvInfo(task=env.task, source="user")

                if done:
                    reward_res = env.calculate_reward()
                    reward = reward_res.reward
                    env_info.reward_info = reward_res
                    env_info.user_cost = 0.0

                env_response = EnvResponse(
                    observation=user_response,
                    reward=reward,
                    done=done,
                    info=env_info,
                )
                env.actions.append(action)

                # Add user response to agent's message history (loss_mask=0)
                ctx.append({"role": "user", "content": user_response})
            else:
                # Tool call: pure Python DB operation, no I/O
                try:
                    env_response = env.step(action)
                except Exception as e:
                    logger.warning(f"Tool execution failed: {e}")
                    res.status = Status.ABORTED
                    return self._build_final_result(
                        res, total_reward, info, ctx.all_messages(), loss_masks, prompt_token_ids, response_token_ids, rollout_log_probs,
                        agent_turns=agent_turns, tool_call_turns=tool_call_turns,
                    )

                ctx.append({
                    "role": "tool",
                    "name": action.name,
                    "content": env_response.observation,
                })

            # Track env/tool/user tokens (loss_mask=0). Same compressed context for consistency.
            env_token_ids, env_loss_mask = self._get_token_delta(state.tokenizer, ctx.all_messages(state.tokenizer))
            response_token_ids.extend(env_token_ids)
            loss_masks.extend(env_loss_mask)
            # Pad rollout_log_probs with 0 for non-agent tokens (loss_mask=0, TIS ignores them)
            rollout_log_probs.extend([0.0] * len(env_token_ids))

            total_reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            if env_response.done:
                res.status = Status.COMPLETED
                break

        if env_response is None or not env_response.done:
            res.status = Status.TRUNCATED

        return self._build_final_result(
            res, total_reward, info, ctx.all_messages(), loss_masks, prompt_token_ids, response_token_ids, rollout_log_probs,
            agent_turns=agent_turns, tool_call_turns=tool_call_turns,
        )


# ---------------------------------------------------------------------------
# Concrete agent class
# ---------------------------------------------------------------------------

class TrainableToolCallingAgentSelfPlay(ToolCallingAgent, SelfPlayAgentMixin):
    """
    ToolCallingAgent with self-play user simulation via SGLang.

    Identical to TrainableToolCallingAgent except asolve() comes from
    SelfPlayAgentMixin, which routes user-simulator calls through the
    same SGLang engine instead of an external API.
    """

    def __init__(
        self,
        tools_info: list[dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        rollout_args: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ):
        super().__init__(
            tools_info=tools_info,
            wiki=wiki,
            model=model,
            provider=provider,
            temperature=temperature,
        )
        self.rollout_args = rollout_args or {}
        self.sampling_params = sampling_params or {
            "temperature": 1.0,
            "max_new_tokens": 1024,
        }
        self.openai_adapter = create_openai_adapter(tools_info=self.tools_info, parser_type="qwen25")


def agent_factory(
    tools_info: list[dict[str, Any]],
    wiki,
    config: RunConfig,
    rollout_args: dict[str, Any] | None = None,
    sampling_params: dict[str, Any] | None = None,
):
    if config.agent_strategy == "tool-calling":
        return TrainableToolCallingAgentSelfPlay(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
            rollout_args=rollout_args,
            sampling_params=sampling_params,
        )
    raise NotImplementedError(f"Unsupported agent strategy: {config.agent_strategy}")
