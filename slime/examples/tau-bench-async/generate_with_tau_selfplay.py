"""
Self-play version of generate_with_tau.py.

The external LLM user simulator (GLM / OpenAI API) is replaced by the same
SGLang rollout engine that serves the policy model.  No external API calls
are made during rollout — both agent and user simulator run through the
local SGLang server.

Usage: set --custom-generate-function-path generate_with_tau_selfplay.generate
in the training script.
"""

import logging
from typing import Any

from tau_bench.envs import get_env
from tau_bench.types import RunConfig

from trainable_agents_selfplay import InteractionResult, Status, agent_factory

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Tau-bench configuration — user_model / user_model_provider are unused
# because the user simulator now runs through SGLang directly.
TAU_CONFIGS = {
    "env": "retail",
    "agent": "tool-calling",
    "user_model": "selfplay",          # unused — kept for RunConfig compatibility
    "task_split": "train",
    # Use "human" so HumanUserSimulationEnv is created: its __init__ does NOT call
    # any LLM API (unlike LLMUserSimulationEnv which calls litellm in __init__).
    # asolve() bypasses env.reset()/env.user entirely, so stdin is never blocked.
    "user_strategy": "human",
    "model_provider": "auto_router",   # unused, required by RunConfig
    "model": "qwen3-30b",              # unused, required by RunConfig
    "user_model_provider": "openai",   # unused
    # Sliding context window: only the last k turns are sent to SGLang for inference.
    # Prevents context-length overflow in long episodes (32768 token limit).
    "context_window_k": 3,
}
CONTEXT_WINDOW_K = TAU_CONFIGS.pop("context_window_k")
tau_config = RunConfig(**TAU_CONFIGS)


def res_to_sample(res: InteractionResult, task_index: int) -> Sample:
    status_mapping = {
        Status.COMPLETED: Sample.Status.COMPLETED,
        Status.TRUNCATED: Sample.Status.TRUNCATED,
        Status.ABORTED: Sample.Status.ABORTED,
    }
    sample = Sample(
        index=task_index,
        prompt=res.prompt,
        tokens=res.tokens,
        response=res.response,
        reward=res.reward,
        loss_mask=res.loss_mask,
        status=status_mapping.get(res.status, Sample.Status.FAILED),
        metadata=res.info,
        rollout_log_probs=res.rollout_log_probs,
    )
    if hasattr(res, "response_length"):
        sample.response_length = res.response_length
    elif res.loss_mask:
        sample.response_length = len(res.loss_mask)
    elif res.tokens:
        sample.response_length = len(res.tokens)
    else:
        sample.response_length = 0
    return sample


async def generate(args: dict[str, Any], sample: Sample, sampling_params: dict) -> Sample:
    """
    Entry point for slime training pipeline.

    Both the policy model (agent) and the user simulator call the same
    SGLang router — no external API is involved.
    """
    assert not args.partial_rollout, "Partial rollout is not supported for tau-bench."

    task_index = int(sample.prompt)
    logger.info(f"Starting self-play interaction for task {task_index}")

    # user_strategy="human" → HumanUserSimulationEnv, no LLM API call on init.
    # asolve() bypasses env.reset()/env.user completely and drives the user sim
    # through SGLang directly, so env.user is never invoked.
    env = get_env(
        env_name=tau_config.env,
        user_strategy=tau_config.user_strategy,  # "human"
        user_model=tau_config.user_model,
        user_provider=tau_config.user_model_provider,
        task_split=tau_config.task_split,
        task_index=task_index,
    )

    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=tau_config,
        rollout_args=args,
        sampling_params=sampling_params,
    )

    interaction_result = await agent.asolve(
        env, agent.rollout_args, agent.sampling_params, task_index,
        context_window_k=CONTEXT_WINDOW_K,
    )

    result_sample = res_to_sample(interaction_result, task_index)
    logger.info(f"Finished self-play interaction for task {task_index}")
    return result_sample
