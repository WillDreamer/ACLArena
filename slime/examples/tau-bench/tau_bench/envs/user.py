# Copyright Sierra

import abc
import enum
from typing import Optional, List, Dict, Any, Union
import logging
import os
import threading
import time

# Set up logger for this module
logger = logging.getLogger(__name__)

# NOTE (ACLArena vendored copy): `litellm` is NOT imported at module top-level.
# The Cluster slime image ships neither litellm nor the external-LLM SDKs,
# and the training path uses the Bedrock Claude user simulator below
# (UserStrategy.CLAUDE), which needs only boto3. The original litellm-backed
# simulators (LLM/REACT/VERIFY/REFLECTION) keep working IF litellm is installed,
# via the lazy `_litellm()` helper. This keeps `import tau_bench` working in the
# container while preserving the upstream eval/user-sim strategies.


def _litellm():
    """Lazily import litellm so the package imports without it (Cluster image)."""
    import litellm  # noqa: F401
    from litellm import completion

    return litellm, completion


class BaseUserSimulationEnv(abc.ABC):
    metadata = {}

    @abc.abstractmethod
    def reset(self, instruction: Optional[str] = None) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, content: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_total_cost(self) -> float:
        raise NotImplementedError


class HumanUserSimulationEnv(BaseUserSimulationEnv):
    def reset(self, instruction: str) -> str:
        return input(f"{instruction}\n")

    def step(self, content: str) -> str:
        return input(f"{content}\n")

    def get_total_cost(self) -> float:
        return 0

MAX_RETRIES = 10
RETRY_DELAY_SECONDS = 1
class LLMUserSimulationEnv(BaseUserSimulationEnv):
    def __init__(self, model: str, provider: str) -> None:
        super().__init__()
        self.messages: List[Dict[str, Any]] = []
        self.model = model
        self.provider = provider
        self.total_cost = 0.0
        self.reset()

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        litellm, completion = _litellm()
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                res = completion(
                    model=self.model, custom_llm_provider=self.provider, messages=messages
                )
                break
            except (litellm.ServiceUnavailableError, litellm.InternalServerError) as e:
                error_message = f"LLM Service Unavailable: Provider={e.llm_provider}, Model={e.model}. Details: {e}"
                
                if attempt == MAX_RETRIES:
                    logger.error(f"Maximum retries ({MAX_RETRIES}) reached. Raising final error.")
                    # Raise the serializable error for Ray to forward
                    raise RuntimeError(error_message) from e 
                
                logger.error(
                    f"LLM call failed (Attempt {attempt}/{MAX_RETRIES}). Error: {e.__class__.__name__}. "
                    f"Retrying in {RETRY_DELAY_SECONDS} seconds..."
                )
                time.sleep(RETRY_DELAY_SECONDS)

            except Exception as e:
                # Catch other unexpected exceptions (e.g., connection errors, model errors)
                logger.error(f"An unexpected error occurred during LLM call on attempt {attempt}: {e}")
                # Since Ray can usually serialize standard exceptions, we just re-raise it
                raise e

        message = res.choices[0].message
        self.messages.append(message.model_dump())
        self.total_cost = res._hidden_params["response_cost"]
        return message.content

    def build_system_prompt(self, instruction: Optional[str]) -> str:
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n")
            if instruction is not None
            else ""
        )
        return f"""You are a user interacting with an agent.{instruction_display}
Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- Only if the instruction goal is satisified and THE EXECUTION IS CONFIRMED generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversatin. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible by using natual language only, and stick to the personalities in the instruction."""

    def reset(self, instruction: Optional[str] = None) -> str:
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


class ReactUserSimulationEnv(LLMUserSimulationEnv):
    def __init__(self, model: str, provider: str) -> None:
        super().__init__(model=model, provider=provider)
        self.reset()

    def build_system_prompt(self, instruction: Optional[str]) -> str:
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n")
            if instruction is not None
            else ""
        )
        return f"""You are a user interacting with an agent.{instruction_display}
Rules:
- First, generate a Thought about what to do next (this message will not be sent to the agent).
- Then, generate a one line User Response to simulate the user's message (this message will be sent to the agent).
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as the User Response without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction.

Format:

Thought:
<the thought>

User Response:
<the user response (this will be parsed and sent to the agent)>"""

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        _, completion = _litellm()
        res = completion(
            model=self.model, custom_llm_provider=self.provider, messages=messages
        )
        message = res.choices[0].message
        self.messages.append(message.model_dump())
        self.total_cost = res._hidden_params["response_cost"]
        return self.parse_response(message.content)

    def reset(self, instruction: Optional[str] = None) -> str:
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def parse_response(self, response: str) -> str:
        if "###STOP###" in response:
            return "###STOP###"
        elif "Thought:" in response:
            _, user_response = response.split("Thought:")
            return user_response.strip()
        elif "User Response:" in response:
            _, user_response = response.split("User Response:")
            return user_response.strip()
        else:
            raise ValueError(f"Invalid response format: {response}")

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


class VerifyUserSimulationEnv(LLMUserSimulationEnv):
    def __init__(self, model: str, provider: str, max_attempts: int = 3) -> None:
        self.model = model
        self.provider = provider
        self.max_attempts = max_attempts
        self.reset()

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        _, completion = _litellm()
        attempts = 0
        cur_message = None
        while attempts < self.max_attempts:
            res = completion(
                model=self.model, custom_llm_provider=self.provider, messages=messages
            )
            cur_message = res.choices[0].message
            self.total_cost = res._hidden_params["response_cost"]
            if verify(self.model, self.provider, cur_message, messages):
                self.messages.append(cur_message.model_dump())
                return cur_message.content
            attempts += 1
        assert cur_message is not None
        return cur_message.content

    def reset(self, instruction: Optional[str] = None) -> str:
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


def map_role_label(role: str) -> str:
    if role == "user":
        return "Customer"
    elif role == "assistant":
        return "Agent"
    else:
        return role.capitalize()


def verify(
    model: str, provider: str, response: str, messages: List[Dict[str, Any]]
) -> bool:
    transcript = "\n".join(
        [
            f"{map_role_label(message['role'])}: {message['content']}"
            for message in messages
        ]
    )
    prompt = f"""You are a supervisor of the Agent in the conversation. You are given a Transcript of a conversation between a Customer and an Agent. The Customer has generated a Response, and you need to verify if it is satisfactory (true) or not (false).
Your answer will be parsed, so do not include any other text than the classification (true or false).
    
# Transcript:
{transcript}

# Response:
{response}

-----

Classification:"""
    _, completion = _litellm()
    res = completion(
        model=model,
        custom_llm_provider=provider,
        messages=[{"role": "user", "content": prompt}],
    )
    return "true" in res.choices[0].message.content.lower()


def reflect(
    model: str, provider: str, response: str, messages: List[Dict[str, Any]]
) -> str:
    transcript = "\n".join(
        [
            f"{map_role_label(message['role'])}: {message['content']}"
            for message in messages
        ]
    )
    prompt = f"""You are a supervisor of the Agent in the conversation. You are given a Transcript of a conversation between a (simulated) Customer and an Agent. The Customer generated a Response that was marked as unsatisfactory by you.
You need to generate a Reflection on what went wrong in the conversation, and propose a new Response that should fix the issues.
Your answer will be parsed, so do not include any other text than the classification (true or false).
    
# Transcript:
{transcript}

# Response:
{response}

# Format:

Reflection:
<the reflection>

Response:
<the response (this will be parsed and sent to the agent)>"""
    _, completion = _litellm()
    res = completion(
        model=model,
        custom_llm_provider=provider,
        messages=[{"role": "user", "content": prompt}],
    )
    _, response = res.choices[0].message.content.split("Response:")
    return response.strip()


class ReflectionUserSimulationEnv(LLMUserSimulationEnv):
    def __init__(self, model: str, provider: str, max_attempts: int = 2) -> None:
        self.model = model
        self.provider = provider
        self.max_attempts = max_attempts
        self.reset()

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        cur_messages = messages.copy()
        initial_response = super().generate_next_message(cur_messages)
        if verify(self.model, self.provider, initial_response, cur_messages):
            return initial_response
        attempts = 1
        while attempts < self.max_attempts:
            new_message = reflect(
                self.model, self.provider, initial_response, cur_messages
            )
            cur_messages.append({"role": "user", "content": new_message})
            new_response = super().generate_next_message(cur_messages)
            if verify(self.model, self.provider, new_response, cur_messages):
                return new_response
            attempts += 1
        return initial_response

    def reset(self, instruction: Optional[str] = None) -> str:
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


# ─────────────────────────────────────────────────────────────────────────────
# ACLArena / Cluster: Bedrock Claude user simulator (no external API, boto3 only).
#
# Replaces the litellm-backed OpenAI/Gemini user simulator. Inside the Cluster
# container the default boto3 session already assumes CLUSTER_DEV_ROLE (acct
# ACCOUNT_ID) via /root/.aws/config (credential_source=EcsContainer); on a dev
# box, set AWS_PROFILE=cluster-dev. We therefore build a plain boto3 client
# with NO explicit profile and let the standard credential chain resolve it.
#
# Mirrors the proven invoke_model + temperature-fallback path from
# cluster_utils/api_usage.py (boto3 1.33 in the image has no .converse(), and
# claude-opus-4-7/4-8 reject `temperature`). Region defaults to us-east-1 (the
# Bedrock region those model ids are enabled in — note the JOB runs in
# ap-south-1, so this is a cross-region call; see run script's egress self-test).
#
# Config via env (all optional, with sane defaults), read at construction time:
#   TAU_USER_MODEL_ID   Bedrock model id   (default us.anthropic.claude-opus-4-7)
#   TAU_BEDROCK_REGION  Bedrock region     (default us-east-1)
#   TAU_USER_MAX_TOKENS max_tokens         (default 1000)
#   TAU_USER_TEMP       temperature        (default 0.0; auto-dropped on 4.7/4.8)
# ─────────────────────────────────────────────────────────────────────────────

# One Bedrock client per (region) per process, created lazily and shared across
# the many concurrent rollout coroutines. boto3 clients are thread-safe for
# invoke_model; we still guard creation with a lock so the asyncio.to_thread
# workers don't race to build duplicates.
_BEDROCK_CLIENTS: Dict[str, Any] = {}
_BEDROCK_LOCK = threading.Lock()


def _get_bedrock_client(region: str):
    cli = _BEDROCK_CLIENTS.get(region)
    if cli is not None:
        return cli
    with _BEDROCK_LOCK:
        cli = _BEDROCK_CLIENTS.get(region)
        if cli is not None:
            return cli
        import boto3
        from botocore.config import Config

        # Default credential chain: in-container ECS creds -> assume
        # CLUSTER_DEV_ROLE (via /root/.aws/config); on a dev box, AWS_PROFILE.
        # Retries/backoff handle transient Bedrock throttling under high rollout
        # concurrency (read_timeout generous: cross-region call from ap-south-1).
        #
        # CRITICAL — max_pool_connections (root cause of a stalled live run, job
        # 376533): the rollout fans out up to TAU_ENV_THREAD_WORKERS (default 256)
        # concurrent threads, each making a blocking invoke_model call through THIS
        # shared client. botocore's default HTTP pool is only 10 connections, so
        # 200+ threads thrashed it — the log filled with "Connection pool is full,
        # discarding connection: bedrock-runtime..." (377 times), every discarded
        # connection paying a fresh cross-region TLS handshake. Throughput
        # collapsed (49/256 trajectories in 14 min, then 0), no training step ever
        # ran, and the Cluster stuck-job detector killed it. Size the pool to the
        # worker count so connections are reused, not churned.
        pool = int(os.environ.get("TAU_BEDROCK_POOL", os.environ.get("TAU_ENV_THREAD_WORKERS", "256")))
        cfg = Config(
            retries={"total_max_attempts": 8, "mode": "adaptive"},
            max_pool_connections=pool,
            read_timeout=60,
            connect_timeout=10,
        )
        cli = boto3.client("bedrock-runtime", region_name=region, config=cfg)
        _BEDROCK_CLIENTS[region] = cli
    return cli


class BedrockClaudeUserSimulationEnv(BaseUserSimulationEnv):
    """tau-bench user simulator backed by Bedrock Claude (invoke_model).

    Drop-in replacement for LLMUserSimulationEnv: same system prompt, same
    one-line-at-a-time behaviour, same '###STOP###' termination contract. Only
    the transport differs (boto3 invoke_model instead of litellm.completion).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,  # accepted & ignored (kept for load_user signature parity)
        region: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.messages: List[Dict[str, Any]] = []
        self.model = model or os.environ.get(
            "TAU_USER_MODEL_ID", "us.anthropic.claude-opus-4-7"
        )
        self.region = region or os.environ.get("TAU_BEDROCK_REGION", "us-east-1")
        self.max_tokens = int(os.environ.get("TAU_USER_MAX_TOKENS", "1000"))
        self.temperature = float(os.environ.get("TAU_USER_TEMP", "0.0"))
        self.total_cost = 0.0  # Bedrock invoke_model returns no cost; kept for API parity
        self.reset()

    def _client(self):
        return _get_bedrock_client(self.region)

    def _invoke(self, system_prompt: str, conv_messages: List[Dict[str, Any]]) -> str:
        """Call Bedrock Claude. conv_messages is the litellm-style list MINUS the
        system message (Anthropic takes system as a top-level field).

        Retries (app-level, on TOP of botocore's adaptive retries): each rollout
        trajectory is multi-turn, and a single user-sim failure makes
        trainable_agents.asolve mark the WHOLE trajectory ABORTED and drop it —
        so one transient blip wastes a full multi-turn rollout. boto3's adaptive
        retry covers throttling/5xx within a single call, but NOT: connection
        resets/read-timeouts that escape it under high concurrency, nor an empty
        completion (Claude occasionally returns no content block under load,
        which would feed an empty user turn into the agent). We wrap the whole
        call in an exponential-backoff+jitter loop that retries on any exception
        AND on empty text, so transient issues are absorbed and the trajectory
        keeps going. Tunables (env): TAU_USER_MAX_RETRIES (default 8),
        TAU_USER_RETRY_BASE secs (default 1.0), TAU_USER_RETRY_CAP secs (60)."""
        import json as _json
        import random

        # Anthropic requires content to be a string (or block list); our messages
        # already use plain-string content, and roles alternate user/assistant.
        body: Dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": m["role"], "content": m["content"]} for m in conv_messages
            ],
        }
        if system_prompt:
            body["system"] = system_prompt

        client = self._client()

        def _call(payload: Dict[str, Any]):
            return client.invoke_model(
                modelId=self.model,
                body=_json.dumps(payload),
                contentType="application/json",
                accept="application/json",
            )

        max_retries = int(os.environ.get("TAU_USER_MAX_RETRIES", "8"))
        base = float(os.environ.get("TAU_USER_RETRY_BASE", "1.0"))
        cap = float(os.environ.get("TAU_USER_RETRY_CAP", "60"))
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                try:
                    resp = _call(body)
                except Exception as e:
                    # opus-4-7 / opus-4-8 reject `temperature`; drop it and retry
                    # in-place (same fallback as cluster_utils/api_usage.py).
                    # Mutating `body` is sticky, so subsequent attempts skip it.
                    if "temperature" in str(e).lower() and "temperature" in body:
                        body.pop("temperature", None)
                        resp = _call(body)
                    else:
                        raise
                payload = _json.loads(resp["body"].read())
                text = "".join(b.get("text", "") for b in payload.get("content", []))
                if text.strip():
                    return text
                # Empty completion — treat as transient and retry.
                last_err = RuntimeError("empty completion from Bedrock")
            except Exception as e:  # noqa: BLE001 — retry on ANY transient failure
                last_err = e

            if attempt < max_retries:
                # Exponential backoff with full jitter, capped.
                delay = min(cap, base * (2 ** (attempt - 1)))
                delay = random.uniform(0, delay)
                logger.warning(
                    f"Bedrock user-sim call failed (attempt {attempt}/{max_retries}): "
                    f"{type(last_err).__name__}: {last_err}. Retrying in {delay:.1f}s."
                )
                time.sleep(delay)

        # Exhausted retries — re-raise so asolve marks this trajectory ABORTED
        # (it will be dropped from the batch, same as before, but only after we
        # genuinely could not get a user turn).
        raise RuntimeError(
            f"Bedrock user-sim failed after {max_retries} attempts: "
            f"{type(last_err).__name__}: {last_err}"
        ) from last_err

    def build_system_prompt(self, instruction: Optional[str]) -> str:
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n") if instruction is not None else ""
        )
        return f"""You are a user interacting with an agent.{instruction_display}
Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- Only if the instruction goal is satisified and THE EXECUTION IS CONFIRMED generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversatin. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible by using natual language only, and stick to the personalities in the instruction."""

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        # messages[0] is the system message; Bedrock takes it separately.
        system_prompt = ""
        conv: List[Dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
            else:
                conv.append(m)
        content = self._invoke(system_prompt, conv)
        self.messages.append({"role": "assistant", "content": content})
        return content

    def reset(self, instruction: Optional[str] = None) -> str:
        self.messages = [
            {"role": "system", "content": self.build_system_prompt(instruction=instruction)},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


# ─────────────────────────────────────────────────────────────────────────────
# ACLArena / Cluster: LOCAL user simulator over an OpenAI-compatible HTTP endpoint.
#
# Replaces the cross-region Bedrock Claude call with an in-cluster model (e.g.
# GLM-flash) served by SGLang on a dedicated node. slime launches that model as a
# second entry in --sglang-config and exposes its router; the rollout entry point
# (generate_with_tau.py) resolves the router URL at runtime and publishes it via
# the TAU_USER_SIM_URL env var, which this sim reads. Keeping the URL in env (not
# a threaded constructor arg) confines the change to user.py — the vendored
# get_env / Env / domain-env chain stays untouched, exactly like the Bedrock sim.
#
# Same system prompt, same one-line-at-a-time behaviour, same '###STOP###'
# termination contract as LLM/Claude sims — only the transport differs (a plain
# requests POST to {base_url}/chat/completions instead of boto3 invoke_model).
#
# Config via env (all optional, read at construction time):
#   TAU_USER_SIM_URL    full OpenAI-compatible endpoint, e.g.
#                       http://<ip>:<port>/v1/chat/completions  (required at run
#                       time; generate_with_tau.py sets it from the live router)
#   TAU_USER_MODEL_ID   served model name SGLang reports (default "user_sim")
#   TAU_USER_MAX_TOKENS max_tokens         (default 1000)
#   TAU_USER_TEMP       temperature        (default 0.0)
# ─────────────────────────────────────────────────────────────────────────────

# One requests.Session per process, shared across the many concurrent rollout
# threads. CRITICAL — pool size (same lesson as the Bedrock client, job <JOB_ID>):
# the rollout fans out up to TAU_ENV_THREAD_WORKERS (default 256) concurrent
# threads, each doing a blocking POST through THIS shared session. urllib3's
# default pool is only 10 connections, so 200+ threads would thrash it ("Connection
# pool is full, discarding connection"), each discard paying a fresh TCP/TLS
# handshake → throughput collapse → no training step → stuck-job kill. Size the
# pool to the worker count so connections are reused, not churned.
_HTTP_SESSION = None
_HTTP_LOCK = threading.Lock()


def _get_http_session():
    global _HTTP_SESSION
    if _HTTP_SESSION is not None:
        return _HTTP_SESSION
    with _HTTP_LOCK:
        if _HTTP_SESSION is not None:
            return _HTTP_SESSION
        import requests
        from requests.adapters import HTTPAdapter

        pool = int(
            os.environ.get(
                "TAU_USER_SIM_POOL", os.environ.get("TAU_ENV_THREAD_WORKERS", "256")
            )
        )
        sess = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=pool,
            pool_maxsize=pool,
            max_retries=0,  # we do our own app-level retries below
        )
        sess.mount("http://", adapter)
        sess.mount("https://", adapter)
        _HTTP_SESSION = sess
    return _HTTP_SESSION


class LocalUserSimulationEnv(BaseUserSimulationEnv):
    """tau-bench user simulator backed by a local OpenAI-compatible HTTP server.

    Drop-in replacement for BedrockClaudeUserSimulationEnv: same system prompt,
    same one-line-at-a-time behaviour, same '###STOP###' termination contract.
    Only the transport differs (requests POST to a local SGLang /chat/completions
    instead of boto3 invoke_model). Unlike the Bedrock sim, the system message is
    passed inline in the messages array (OpenAI schema), so no system split.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,  # accepted & ignored (load_user parity)
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.messages: List[Dict[str, Any]] = []
        self.base_url = base_url or os.environ.get("TAU_USER_SIM_URL")
        if not self.base_url:
            raise ValueError(
                "LocalUserSimulationEnv requires a base URL: set TAU_USER_SIM_URL "
                "(e.g. http://<ip>:<port>/v1/chat/completions) or pass base_url."
            )
        self.model = model or os.environ.get("TAU_USER_MODEL_ID", "user_sim")
        self.max_tokens = int(os.environ.get("TAU_USER_MAX_TOKENS", "1000"))
        # DEFAULT 0.7 (was 0.0). At temperature 0 the GLM user-sim decodes greedily,
        # so once the dialogue state stabilizes it emits the SAME line every turn —
        # observed as a verbatim-echo death loop (114/256 trajectories repeated one
        # reply, up to 25x; 38% of consecutive user-sim turns byte-identical), which
        # never advances the task and burns the 30-turn cap -> reward -0.2 on ~44%
        # of trajectories. Sampling at 0.7 lets the user-sim break out of the loop.
        self.temperature = float(os.environ.get("TAU_USER_TEMP", "0.7"))
        self.timeout = float(os.environ.get("TAU_USER_HTTP_TIMEOUT", "60"))
        # GLM-4.7-Flash is a HYBRID reasoning model (thinking ON by default). For a
        # user simulator we want a terse one-line user turn, NOT a chain of thought:
        # thinking wastes tokens/latency and — if the server folds reasoning into
        # `content` — would pollute the user message fed to the agent. Default to
        # no-think via the OpenAI `chat_template_kwargs.enable_thinking=false`
        # (honored by SGLang's GLM chat template). Disable with TAU_USER_NO_THINK=0.
        self.no_think = os.environ.get("TAU_USER_NO_THINK", "1") not in ("0", "false", "False")
        # HOW to request no-think (the chat_template_kwargs KEY). GLM-4.7 may 400
        # on the Qwen-style "enable_thinking" key, so default to "off" (send no
        # kwarg; reasoning_parser glm45 keeps any CoT out of `content` anyway).
        # Override: TAU_USER_THINK_KWARG=enable_thinking|thinking|off.
        self._think_kwarg_mode = os.environ.get("TAU_USER_THINK_KWARG", "off").strip()
        self.total_cost = 0.0  # local model has no $ cost; kept for API parity
        self.reset()

    def _invoke(self, messages: List[Dict[str, Any]]) -> str:
        """POST the full OpenAI-style messages list to the local server.

        Retry rationale is identical to the Bedrock sim: a single user-sim
        failure makes trainable_agents.asolve mark the WHOLE multi-turn
        trajectory ABORTED and drop it, so one transient blip wastes a full
        rollout. We wrap the call in an exponential-backoff+jitter loop that
        retries on any exception AND on an empty completion. Tunables (env):
        TAU_USER_MAX_RETRIES (8), TAU_USER_RETRY_BASE secs (1.0),
        TAU_USER_RETRY_CAP secs (60)."""
        import random

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": m["role"], "content": m["content"]} for m in messages
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.no_think and self._think_kwarg_mode != "off":
            # SGLang passes chat_template_kwargs through to the model's chat
            # template. NOTE: `enable_thinking` is the QWEN convention; GLM-4.7's
            # template may NOT accept it and can 400 the whole request (suspected
            # root cause of a 26.5k-deterministic-400 run against the GLM user-sim).
            # So the KEY is configurable via TAU_USER_THINK_KWARG:
            #   off              -> send nothing (let GLM default; reasoning is
            #                       split into reasoning_content by reasoning_parser
            #                       glm45, so it won't pollute the user message)
            #   enable_thinking  -> {"enable_thinking": False}   (Qwen-style)
            #   thinking         -> {"thinking": False}          (some GLM builds)
            # Default below stays enable_thinking for backward-compat, but the run
            # script defaults the ENV to "off" (GLM-safe). One key only.
            payload["chat_template_kwargs"] = {self._think_kwarg_mode: False}
        session = _get_http_session()

        max_retries = int(os.environ.get("TAU_USER_MAX_RETRIES", "8"))
        base = float(os.environ.get("TAU_USER_RETRY_BASE", "1.0"))
        cap = float(os.environ.get("TAU_USER_RETRY_CAP", "60"))
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = session.post(self.base_url, json=payload, timeout=self.timeout)
                # A 4xx (except 429 Too-Many-Requests) is a DETERMINISTIC reject:
                # the request is malformed/unacceptable, so retrying it 8x is pure
                # waste and floods the logs (observed: 26.5k identical 400s, every
                # call burning all 8 attempts then aborting the whole trajectory).
                # Surface the server's response BODY (the real reason — SGLang does
                # not log it in CloudWatch) and fail fast instead of retrying.
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    body = ""
                    try:
                        body = resp.text[:500]
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"local user-sim {resp.status_code} (non-retryable client error) "
                        f"from {self.base_url}; body={body!r}"
                    )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"] or ""
                if text.strip():
                    return text
                # Empty completion — treat as transient and retry.
                last_err = RuntimeError("empty completion from local user-sim")
            except RuntimeError as e:
                # Non-retryable client error raised above — re-raise immediately so
                # asolve aborts this trajectory without burning the retry budget.
                if "non-retryable client error" in str(e):
                    logger.warning(f"Local user-sim NON-RETRYABLE: {e}")
                    raise
                last_err = e
            except Exception as e:  # noqa: BLE001 — retry on ANY transient failure
                last_err = e

            if attempt < max_retries:
                # Exponential backoff with full jitter, capped.
                delay = min(cap, base * (2 ** (attempt - 1)))
                delay = random.uniform(0, delay)
                logger.warning(
                    f"Local user-sim call failed (attempt {attempt}/{max_retries}): "
                    f"{type(last_err).__name__}: {last_err}. Retrying in {delay:.1f}s."
                )
                time.sleep(delay)

        # Exhausted retries — re-raise so asolve marks this trajectory ABORTED.
        raise RuntimeError(
            f"Local user-sim failed after {max_retries} attempts: "
            f"{type(last_err).__name__}: {last_err}"
        ) from last_err

    def build_system_prompt(self, instruction: Optional[str]) -> str:
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n") if instruction is not None else ""
        )
        return f"""You are a user interacting with an agent.{instruction_display}
Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- Only if the instruction goal is satisified and THE EXECUTION IS CONFIRMED generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversatin. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible by using natual language only, and stick to the personalities in the instruction."""

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        # OpenAI schema keeps the system message inline in the array — no split.
        content = self._invoke(messages)
        self.messages.append({"role": "assistant", "content": content})
        return content

    def reset(self, instruction: Optional[str] = None) -> str:
        self.messages = [
            {"role": "system", "content": self.build_system_prompt(instruction=instruction)},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


class UserStrategy(enum.Enum):
    HUMAN = "human"
    LLM = "llm"
    REACT = "react"
    VERIFY = "verify"
    REFLECTION = "reflection"
    CLAUDE = "claude"  # ACLArena: Bedrock Claude user simulator (boto3, no external API)
    LOCAL = "local"  # ACLArena: local OpenAI-compatible user simulator (e.g. GLM-flash)


def load_user(
    user_strategy: Union[str, UserStrategy],
    model: Optional[str] = "gpt-4o",
    provider: Optional[str] = None,
) -> BaseUserSimulationEnv:
    if isinstance(user_strategy, str):
        user_strategy = UserStrategy(user_strategy)
    if user_strategy == UserStrategy.HUMAN:
        return HumanUserSimulationEnv()
    elif user_strategy == UserStrategy.CLAUDE:
        # Bedrock Claude user sim. `model`/`provider` here come from RunConfig
        # (user_model / user_model_provider); the Bedrock model id + region are
        # taken from env (TAU_USER_MODEL_ID / TAU_BEDROCK_REGION) so the run
        # script controls them. If user_model looks like a Bedrock id, honor it.
        bedrock_model = model if (model and "anthropic" in model) else None
        return BedrockClaudeUserSimulationEnv(model=bedrock_model, provider=provider)
    elif user_strategy == UserStrategy.LOCAL:
        # Local OpenAI-compatible user sim (e.g. GLM-flash on a dedicated node).
        # The served model name comes from env (TAU_USER_MODEL_ID) unless the
        # RunConfig user_model looks like a real served name; the endpoint URL is
        # read from TAU_USER_SIM_URL (published by generate_with_tau.py at run
        # time from the live SGLang router). Both default inside the sim.
        local_model = model if (model and "anthropic" not in model and model != "gpt-4o") else None
        return LocalUserSimulationEnv(model=local_model, provider=provider)
    elif user_strategy == UserStrategy.LLM:
        if model is None:
            raise ValueError("LLM user strategy requires a model")
        if provider is None:
            raise ValueError("LLM user strategy requires a model provider")
        return LLMUserSimulationEnv(model=model, provider=provider)
    elif user_strategy == UserStrategy.REACT:
        if model is None:
            raise ValueError("React user strategy requires a model")
        if provider is None:
            raise ValueError("React user strategy requires a model provider")
        return ReactUserSimulationEnv(model=model, provider=provider)
    elif user_strategy == UserStrategy.VERIFY:
        if model is None:
            raise ValueError("Verify user strategy requires a model")
        if provider is None:
            raise ValueError("Verify user strategy requires a model provider")
        return VerifyUserSimulationEnv(model=model, provider=provider)
    elif user_strategy == UserStrategy.REFLECTION:
        if model is None:
            raise ValueError("Reflection user strategy requires a model")
        if provider is None:
            raise ValueError("Reflection user strategy requires a model provider")
        return ReflectionUserSimulationEnv(model=model, provider=provider)
    raise ValueError(f"Unknown user strategy {user_strategy}")
