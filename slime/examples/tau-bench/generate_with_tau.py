"""
Tau-Bench Integration for slime Training

This module provides the main interface for training agents in tau-bench environments
using the slime framework. It handles agent-environment interactions and converts
results to the format expected by slime's training pipeline.
"""

import logging
import os
from typing import Any

from async_env import get_async_env
from tau_bench.types import RunConfig
from trainable_agents import InteractionResult, Status, agent_factory

from slime.utils.types import Sample
from slime.rollout.sglang_rollout import get_model_url

# Set up logger for this module
logger = logging.getLogger(__name__)

# Placeholder token id for the degenerate-sample no-op (see res_to_sample). Any
# in-vocab id works: the synthetic sample's loss_mask is ALL ZERO, so these
# tokens never contribute a gradient — they exist only to give get_batch a
# structurally valid (prompt_length >= 1) tensor instead of an empty one.
_ABORT_PLACEHOLDER_TOKEN_ID = 0

# ─────────────────────────────────────────────────────────────────────────────
# Tau-bench configuration (ACLArena / Cluster).
#
# The user simulator is selected by TAU_USER_STRATEGY:
#   "claude" — Bedrock Claude (boto3, cross-region; no external API/key). The
#              model id / region come from TAU_USER_MODEL_ID / TAU_BEDROCK_REGION
#              (read in tau_bench/envs/user.py::BedrockClaudeUserSimulationEnv).
#   "local"  — a model served IN-CLUSTER over an OpenAI-compatible HTTP endpoint
#              (e.g. GLM-flash on a dedicated node, launched by slime as a second
#              entry in --sglang-config). No external egress at all. The endpoint
#              is resolved at run time from the live SGLang router and published
#              via TAU_USER_SIM_URL (see generate() below); the served model name
#              comes from TAU_USER_MODEL_ID. Read in user.py::LocalUserSimulationEnv.
#
# Everything here is overridable via env so the run script is the single source
# of truth (no code edits to switch env/split/user-model):
#   TAU_ENV            retail | airline                       (default retail)
#   TAU_TASK_SPLIT     train | test | dev                     (default train)
#   TAU_USER_STRATEGY  claude | local | llm | react | ...     (default claude)
#   TAU_USER_MODEL_ID  Bedrock model id (claude) OR served model name (local);
#                      mirrored into RunConfig.user_model.
#   TAU_USER_SIM_MODEL named model in --sglang-config to route to (local; default
#                      "user_sim"); used to resolve the router URL from args.
#   TAU_USER_SIM_URL   explicit endpoint override (local). If unset, generate()
#                      fills it from the live router; if set, it wins (single-node
#                      tests / self-launched servers).
# model / model_provider are UNUSED by our SGLang rollout path but RunConfig
# requires them, so we keep placeholder values.
# ─────────────────────────────────────────────────────────────────────────────
TAU_CONFIGS = {
    "env": os.environ.get("TAU_ENV", "retail"),
    "agent": "tool-calling",  # only tool-calling is implemented for training
    "user_model": os.environ.get("TAU_USER_MODEL_ID", "us.anthropic.claude-opus-4-7"),
    "task_split": os.environ.get("TAU_TASK_SPLIT", "train"),
    "user_strategy": os.environ.get("TAU_USER_STRATEGY", "claude"),
    "model_provider": "auto_router",  # Unused, required by RunConfig
    "model": "qwen3.5-4b",  # Unused, required by RunConfig
    "user_model_provider": "bedrock",  # informational; CLAUDE/LOCAL strategies ignore it
}
tau_config = RunConfig(**TAU_CONFIGS)


def _ensure_user_sim_url(args) -> None:
    """For the LOCAL user-sim, publish the live SGLang router URL via env.

    slime assigns the user-sim model's router port dynamically at launch, so we
    cannot hardcode it. The named-model router map lives on `args`
    (args.sglang_model_routers, populated by start_rollout_servers); resolve it
    once here with get_model_url and stash it in TAU_USER_SIM_URL so the env
    construction chain (which has no access to `args`) can read it. An explicit
    TAU_USER_SIM_URL set by the operator always wins (idempotent: only fills it
    when empty)."""
    if tau_config.user_strategy != "local":
        return
    if os.environ.get("TAU_USER_SIM_URL"):
        return
    model_name = os.environ.get("TAU_USER_SIM_MODEL", "user_sim")
    url = get_model_url(args, model_name, "/v1/chat/completions")
    os.environ["TAU_USER_SIM_URL"] = url
    logger.info(f"Resolved local user-sim endpoint for model '{model_name}': {url}")


def res_to_sample(res: InteractionResult, task_index: int) -> Sample:
    """
    Convert InteractionResult to Sample format for slime training.

    This function transforms the tau-bench interaction result into the format
    expected by slime's training pipeline, handling status mapping and response
    length calculation.

    Args:
        res: InteractionResult from tau-bench agent
        task_index: Index of the task being processed

    Returns:
        Sample object for slime training
    """
    # Map tau-bench status to slime status
    status_mapping = {
        Status.COMPLETED: Sample.Status.COMPLETED,
        Status.TRUNCATED: Sample.Status.TRUNCATED,
        Status.ABORTED: Sample.Status.ABORTED,
    }
    status = status_mapping.get(res.status, Sample.Status.ABORTED)

    # ── Degenerate-sample guard (Option 1: drop from training) ──
    # asolve's env-reset failure path early-returns an EMPTY result
    # (prompt_token_ids=[], response_token_ids=[]), so res.tokens == [] and
    # res.response_length == 0. That sample is ABORTED, but the tau custom
    # generate path bypasses slime's standard ABORTED filtering, so it reaches
    # the trainer. There, get_batch (megatron_utils/data.py) computes
    #   prompt_length = total_length - response_length
    # and does F.pad(loss_mask, (prompt_length - 1, 1)). With an empty sample
    # prompt_length == 0 -> negative left-pad -> the job-killing
    #   RuntimeError: narrow(): length must be non-negative
    # (Cluster job <JOB_ID>, first train step.)
    #
    # We cannot physically drop the sample: generate_rollout_async asserts every
    # group has exactly n_samples_per_prompt samples (GRPO needs the full group),
    # so removing one breaks the group. Instead we emit a STRUCTURALLY VALID but
    # ZERO-GRADIENT no-op: a 1-token prompt + 1-token response with loss_mask all
    # zero. prompt_length == 1 keeps F.pad non-negative, and the all-zero
    # loss_mask means this sample contributes nothing to the loss — equivalent to
    # dropping it. If the whole task's group degenerates (env fully failed to
    # reset for all n_samples), every sample carries the same reward, so the
    # dynamic_sampling_filter (check_reward_nonzero_std) drops the entire group
    # and the task is skipped — the desired outcome.
    # Guard the EXACT crash condition: get_batch needs prompt_length >= 1, where
    # prompt_length = total_length - response_length = len(res.tokens) - response_length
    # (res.tokens = prompt_token_ids + response_token_ids; response_length =
    # len(response_token_ids); so this == len(prompt_token_ids)). This single
    # expression catches the empty early-return (tokens=[]) AND any pathological
    # state where the prompt half is empty while the response half is not.
    n_tokens = len(res.tokens) if res.tokens else 0
    resp_len = res.response_length if (getattr(res, "response_length", None) is not None) else 0
    prompt_length = n_tokens - resp_len
    is_degenerate = (n_tokens == 0) or (resp_len <= 0) or (prompt_length <= 0)
    if is_degenerate:
        logger.warning(
            f"res_to_sample: degenerate trajectory for task {task_index} "
            f"(status={res.status}, tokens_len={n_tokens}, response_length={resp_len}, "
            f"prompt_length={prompt_length}); emitting a zero-gradient no-op sample "
            f"(loss_mask all 0) so it is excluded from training without breaking the "
            f"GRPO group."
        )
        noop = Sample(
            index=task_index,
            prompt=res.prompt if res.prompt else str(task_index),
            tokens=[_ABORT_PLACEHOLDER_TOKEN_ID, _ABORT_PLACEHOLDER_TOKEN_ID],
            response="",
            reward=0.0,
            loss_mask=[0],
            status=Sample.Status.ABORTED,
            metadata=res.info,
            rollout_log_probs=[0.0],
            # Also flag it the slime-native way: rollout.py zeroes the loss_mask of
            # any remove_sample=True sample (and the loss reducer clamps the
            # per-sample token denominator with clamp_min(.,1), so an all-zero mask
            # is div-by-zero safe). Belt-and-suspenders with our explicit [0] mask.
            remove_sample=True,
        )
        # total_length=len(tokens)=2, response_length=1 -> prompt_length=1 in
        # get_batch (data.py:139); len(loss_mask)==response_length==1 satisfies the
        # rollout.py:753 assert.
        noop.response_length = 1
        return noop

    # Debug logging for response tracking
    logger.debug(
        f"res_to_sample: response_length="
        f"{res.response_length if hasattr(res, 'response_length') else 'None'}, "
        f"loss_mask_len={len(res.loss_mask) if res.loss_mask else 'None'}, "
        f"tokens_len={len(res.tokens) if res.tokens else 'None'}"
    )

    # Create sample with basic information
    sample = Sample(
        index=task_index,
        prompt=res.prompt,
        tokens=res.tokens,
        response=res.response,
        reward=res.reward,
        loss_mask=res.loss_mask,
        status=status,
        metadata=res.info,
        rollout_log_probs=res.rollout_log_probs,
    )

    # Truncated trajectories: keep them in the batch with a clearly negative
    # reward instead of dropping. Magnitude (-0.2) is set so truncation is
    # *worse* than a format-bad failure (-0.1); otherwise the model finds it
    # cheaper to keep thinking until the length cap fires than to emit an
    # imperfect tool_call, which directly drives CoT longer.
    if status == Sample.Status.TRUNCATED:
        sample.reward = -0.2

    # ABORTED trajectories MUST be penalized at least as hard as truncation,
    # otherwise RL collapses. An abort = the agent emitted a broken/empty turn
    # (e.g. wrote a <think> block then `<|im_end|>` with NO action, or an
    # unparseable tool_call, or a user-sim error). The env returns reward 0.0 for
    # these. With aborted==0.0 but truncated==-0.2, on a HARD GRPO group (16
    # samples, mostly 0.0/-0.2) an early ABORT scores HIGHER than a genuine
    # attempt that hits the turn cap, and ties a failed-but-complete attempt — so
    # "give up immediately" gets a POSITIVE within-group advantage and is
    # reinforced. That is exactly what collapsed job <JOB_ID>: aborts rose from
    # ~rollout 100, the policy mode-collapsed to emitting only `<|im_end|>` by
    # rollout 119 (eval 0.80 -> 0.00). Set ABORTED to -0.3 (strictly WORSE than
    # truncation -0.2) so quitting is never cheaper than trying. Env-tunable.
    # NOTE: the degenerate-sample no-op path above (remove_sample, loss_mask all
    # 0) handles the EMPTY-prompt abort and returns early, so this only re-scores
    # NON-empty aborts that actually carry trainable tokens.
    if status == Sample.Status.ABORTED:
        sample.reward = float(os.environ.get("TAU_ABORTED_REWARD", "-0.3"))

    # Ensure response_length is set correctly
    if hasattr(res, "response_length"):
        sample.response_length = res.response_length
    else:
        # Fallback: calculate from loss_mask if available
        if res.loss_mask:
            # loss_mask only contains response part, so length equals response_length
            sample.response_length = len(res.loss_mask)
        elif res.tokens:
            # If no loss_mask available, use total tokens as fallback
            sample.response_length = len(res.tokens)
        else:
            sample.response_length = 0
            logger.debug(f"res_to_sample: Set response_length={sample.response_length}")

    return sample


async def generate(args: dict[str, Any], sample: Sample, sampling_params: dict) -> Sample:
    """
    Generate a complete agent-environment interaction trajectory for tau-bench.

    This is the main entry point for slime training. It creates a tau-bench
    environment, initializes a trainable agent, and executes a full interaction
    trajectory. The result is converted to slime's Sample format for training.

    Args:
        args: Rollout arguments from slime training pipeline
        sample: Sample containing task index in prompt field
        sampling_params: LLM sampling parameters

    Returns:
        Sample object containing the complete interaction trajectory

    Raises:
        AssertionError: If partial rollout is requested (not supported)
    """
    # Validate arguments
    assert not args.partial_rollout, "Partial rollout is not supported for tau-bench interactions."

    # Extract task index from sample prompt
    task_index = int(sample.prompt)
    logger.info(f"Starting agent-environment interaction for task {task_index}")

    # For the local user-sim, resolve the in-cluster SGLang router URL from args
    # and publish it via TAU_USER_SIM_URL (no-op for the Bedrock/claude path).
    _ensure_user_sim_url(args)

    # Initialize tau-bench environment (async wrapper around the sync env so the
    # blocking user-sim call runs in a worker thread, not on the loop).
    env = get_async_env(
        env_name=tau_config.env,
        user_strategy=tau_config.user_strategy,
        user_model=tau_config.user_model,
        user_provider=tau_config.user_model_provider,
        task_split=tau_config.task_split,
        task_index=task_index,
    )

    # Create trainable agent
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=tau_config,
        rollout_args=args,
        sampling_params=sampling_params,
    )

    # Execute agent-environment interaction
    # Note: The sample.prompt field contains the task index for repeatability
    interaction_result = await agent.asolve(env, agent.rollout_args, agent.sampling_params, task_index)

    # Convert to slime Sample format
    result_sample = res_to_sample(interaction_result, task_index)

    logger.info(f"Finished agent-environment interaction for task {task_index}")
    return result_sample
