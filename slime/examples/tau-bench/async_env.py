"""
Async wrapper around tau-bench's synchronous Env.

WHY THIS EXISTS
    slime's rollout (examples/tau-bench/trainable_agents.py) drives the
    environment with `await env.reset(...)` / `await env.step(...)`, because a
    single GPU rollout batch fans out *many* trajectories concurrently via
    asyncio.gather. tau-bench's upstream `Env` (vendored under ./tau_bench) is
    fully synchronous, and its user simulator now makes a BLOCKING Bedrock
    invoke_model call (see tau_bench/envs/user.py::BedrockClaudeUserSimulationEnv).

    If we called the sync env directly from the event loop, every Bedrock call
    (hundreds of ms, cross-region) would block the loop and serialize the whole
    batch. Wrapping each blocking call in `asyncio.to_thread(...)` hands it to a
    worker thread; boto3 releases the GIL on socket I/O, so N user-sim calls run
    truly concurrently and the rollout batch overlaps with SGLang generation.

    The env's reward computation (calculate_reward -> replays gt actions) stays
    inside the same worker thread, so no tau-bench internals need to be async.

This keeps the vendored tau_bench package untouched (pure upstream + the Claude
user-sim addition) and confines all async glue to the example layer.
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from tau_bench.envs import get_env as _get_sync_env
from tau_bench.envs.base import Env
from tau_bench.types import Action, EnvResetResponse, EnvResponse

# Dedicated thread pool for the blocking env calls (Bedrock user-sim invoke).
# We do NOT use asyncio.to_thread / the loop's default executor: that pool caps
# at min(32, cpu+4) workers and is shared with the rest of slime, so a rollout
# batch of hundreds of concurrent trajectories would serialize on ~32 threads
# (each Bedrock call is ~1-3s). A dedicated, larger pool lets the user-sim calls
# fan out to match the rollout concurrency. Sized via env; threads are cheap
# here because they spend ~all their time blocked on a socket (GIL released).
_TAU_ENV_WORKERS = int(os.environ.get("TAU_ENV_THREAD_WORKERS", "256"))
_EXECUTOR = ThreadPoolExecutor(
    max_workers=_TAU_ENV_WORKERS, thread_name_prefix="tau-env"
)


class AsyncTauEnv:
    """Thin async facade over a synchronous tau-bench Env.

    Exposes the exact attributes/methods the rollout touches:
        .tools_info, .wiki   (read straight off the sync env)
        await reset(task_index=...) -> EnvResetResponse
        await step(action)          -> EnvResponse
    Every blocking call is offloaded to a thread so the asyncio event loop that
    drives the rollout batch stays responsive.
    """

    def __init__(self, env: Env):
        self._env = env

    # Pass-through attributes the agent reads synchronously (no I/O).
    @property
    def tools_info(self):
        return self._env.tools_info

    @property
    def wiki(self):
        return self._env.wiki

    @property
    def task(self):
        return self._env.task

    async def reset(self, task_index: Optional[int] = None) -> EnvResetResponse:
        # env.reset triggers the user-simulator's first turn (a Bedrock call).
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_EXECUTOR, self._env.reset, task_index)

    async def step(self, action: Action) -> EnvResponse:
        # RESPOND actions trigger a user-sim Bedrock call; tool actions are local
        # (pure-python tool .invoke); the terminal step also replays gt actions
        # for reward. All of it runs in the worker thread.
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_EXECUTOR, self._env.step, action)


def get_async_env(
    env_name: str,
    user_strategy: str,
    user_model: str,
    task_split: str,
    user_provider: Optional[str] = None,
    task_index: Optional[int] = None,
) -> AsyncTauEnv:
    """Build the sync tau-bench env, then wrap it for async use.

    Signature mirrors tau_bench.envs.get_env so call sites read identically.
    """
    sync_env = _get_sync_env(
        env_name=env_name,
        user_strategy=user_strategy,
        user_model=user_model,
        user_provider=user_provider,
        task_split=task_split,
        task_index=task_index,
    )
    return AsyncTauEnv(sync_env)
