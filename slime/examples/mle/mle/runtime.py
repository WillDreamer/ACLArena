"""Classify the first request and keep one expert per conversation."""

import asyncio
import json
import sqlite3
import time
from pathlib import Path

import httpx

from mle.core import (
    TASK_TO_EXPERT,
    canonical,
    classification_regex,
    digest,
    first_user_input,
    parse_prediction,
    router_messages,
)
from mle.rules import routing_rule


def load_config(path):
    return json.loads(Path(path).read_text())


class Router:
    def __init__(self, config_path, experts_path, state_directory, service_manager=None):
        from transformers import AutoTokenizer

        self.config = load_config(config_path)
        self.classifier = self.config["classifier"]
        self.experts = load_config(experts_path)
        self.mapping = TASK_TO_EXPERT
        self.service_manager = service_manager
        self.service_lock = asyncio.Lock()
        if set(self.experts) != set(self.mapping.values()):
            raise ValueError("Exactly four explicit expert services are required")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.classifier["model_id"],
            trust_remote_code=False,
        )
        self.prompt = Path(__file__).with_name("prompt.txt").read_text()
        self.router_id = digest({"config": self.config, "experts": self.experts, "prompt": self.prompt})
        self.client = httpx.AsyncClient(timeout=self.config["timeout_seconds"], trust_env=False)
        state_directory = Path(state_directory)
        state_directory.mkdir(parents=True, exist_ok=True)
        self.events = state_directory / "events.jsonl"
        self.db = sqlite3.connect(state_directory / "conversations.sqlite3", isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS routes (session TEXT PRIMARY KEY, decision TEXT NOT NULL)")
        self.locks = {}

    async def close(self):
        if self.service_manager is not None:
            await asyncio.to_thread(self.service_manager.stop)
        await self.client.aclose()
        self.db.close()

    async def ensure_service(self, name):
        if self.service_manager is None:
            return
        loading = asyncio.create_task(asyncio.to_thread(self.service_manager.ensure, name))
        try:
            await asyncio.shield(loading)
        except asyncio.CancelledError as cancelled:
            # Finish a running load before releasing the request's switch lock.
            try:
                await loading
            finally:
                raise cancelled

    def log(self, event):
        with self.events.open("a") as f:
            f.write(canonical({"time": time.time(), **event}) + "\n")

    async def check_services(self):
        if self.service_manager is not None:
            self.service_manager.check()
            if self.service_manager.exclusive:
                return
        async def check(name, service):
            response = await self.client.get(service["base_url"] + "/models")
            response.raise_for_status()
            models = {m["id"] for m in response.json()["data"]}
            if service["model_name"] not in models:
                raise ValueError(f"Wrong or unavailable model service: {name}")

        services = {"classifier": self.classifier, **self.experts}
        await asyncio.gather(*(check(n, s) for n, s in services.items()))

    async def classify(self, payloads, *, fallback=False):
        """Apply tool rules, then classify remaining inputs with constrained decoding."""
        results = [None] * len(payloads)
        pending, rendered, lengths = [], [], []
        for i, payload in enumerate(payloads):
            start = time.monotonic()
            base = {"attempt": 1, "fallback": False, "model_called": False, "decision_source": "router_model", "rule_id": None}
            results[i] = base
            try:
                decision = routing_rule(payload)
                if decision:
                    label, rule = decision
                    base.update(status="ok", predicted=label, rule_id=rule, decision_source="tool_rule", raw_output=canonical({"task_type": label}), latency_seconds=time.monotonic() - start)
                    continue
                messages = router_messages(payload, self.prompt)
                prompt = self.config["completion_template"].format(
                    policy=messages[0]["content"], canonical_task_data=messages[1]["content"],
                )
                length = len(self.tokenizer.encode(prompt, add_special_tokens=False))
                if length + self.config["max_new_tokens"] > self.config["serving"]["context_length"]:
                    raise ValueError("Input exceeds context window; truncation is forbidden")
                pending.append(i)
                rendered.append(prompt)
                lengths.append(length)
            except Exception as exc:
                base.update(status="input_error", error=str(exc), raw_output=None, latency_seconds=time.monotonic() - start)
        if pending:
            start = time.monotonic()
            for i in pending:
                results[i]["model_called"] = True
            try:
                await self.ensure_service("classifier")
                start = time.monotonic()
                response = await self.client.post(
                    self.classifier["base_url"] + "/completions",
                    json={"model": self.classifier["model_name"], "prompt": rendered, "temperature": 0, "seed": self.config["serving"]["random_seed"], "n": 1, "max_tokens": self.config["max_new_tokens"], "regex": classification_regex()},
                )
                response.raise_for_status()
                choices = sorted(response.json()["choices"], key=lambda x: x["index"])
                if [c["index"] for c in choices] != list(range(len(pending))):
                    raise ValueError("Classifier returned duplicate or missing choices")
                for i, length, choice in zip(pending, lengths, choices, strict=True):
                    result = results[i]
                    result.update(raw_output=choice["text"], input_tokens=length, output_tokens=len(self.tokenizer.encode(choice["text"], add_special_tokens=False)))
                    try:
                        if choice["finish_reason"] != "stop":
                            raise ValueError("Incomplete classification output")
                        result.update(predicted=parse_prediction(choice["text"]), status="ok")
                    except (ValueError, TypeError) as exc:
                        result.update(status="format_error", error=str(exc))
            except Exception as exc:
                status = "timeout" if isinstance(exc, httpx.TimeoutException) else "call_error"
                for i in pending:
                    results[i].update(status=status, error=str(exc), raw_output=None)
            elapsed = time.monotonic() - start
            for i in pending:
                results[i]["latency_seconds"] = elapsed
                if elapsed > self.config["timeout_seconds"]:
                    results[i].update(status="timeout", error="Classification batch exceeded timeout")
        for result in results:
            if result["status"] != "ok" and fallback:
                result.update(predicted="instruction_following", fallback=True)
            if result.get("predicted"):
                result["expert"] = self.mapping[result["predicted"]]
        return results

    async def complete(self, body, session_id):
        if self.service_manager is not None and self.service_manager.exclusive:
            async with self.service_lock:
                return await self._complete(body, session_id)
        return await self._complete(body, session_id)

    async def _complete(self, body, session_id):
        # Each process has one event loop; a conversation lock covers its full turn.
        # Different conversations can use independent, fixed model services concurrently.
        async with self.locks.setdefault(session_id, asyncio.Lock()):
            row = self.db.execute("SELECT decision FROM routes WHERE session=?", (session_id,)).fetchone()
            if row:
                decision = json.loads(row[0])
                if decision.get("router_id") != self.router_id:
                    raise ValueError("Conversation belongs to a different router configuration; use a new conversation ID")
            else:
                try:
                    payload = first_user_input(body["messages"], body.get("tools", []))
                except (ValueError, TypeError, KeyError) as exc:
                    decision = {"predicted": "instruction_following", "expert": "if_expert", "fallback": True, "status": "input_error", "error": str(exc)}
                else:
                    decision = (await self.classify([payload], fallback=True))[0]
                decision["router_id"] = self.router_id
                self.db.execute("INSERT INTO routes VALUES (?, ?)", (session_id, canonical(decision)))
                self.log({"event": "route", "session_id": session_id, **decision})
            expert = self.experts[decision["expert"]]
            forwarded = {**self.config["answer_defaults"], **body, "model": expert["model_name"]}
            start = time.monotonic()
            try:
                await self.ensure_service(decision["expert"])
                response = await self.client.post(expert["base_url"] + "/chat/completions", json=forwarded, timeout=self.config["answer_timeout_seconds"])
                response.raise_for_status()
                output = response.json()
            except Exception as exc:
                self.log({"event": "answer_error", "session_id": session_id, "expert": decision["expert"], "error": str(exc)})
                raise
            self.log({"event": "answer", "session_id": session_id, "expert": decision["expert"], "latency_seconds": time.monotonic() - start, "usage": output.get("usage")})
            return output, decision
