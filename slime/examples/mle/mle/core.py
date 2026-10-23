"""Input validation, task labels, and JSON serialization for routing."""

import hashlib
import json
import re

TASK_TO_EXPERT = {
    "reasoning": "math_expert",
    "search": "search_expert",
    "customer_service": "ecommerce_expert",
    "instruction_following": "if_expert",
}
TASK_TYPES = tuple(TASK_TO_EXPERT)
INPUT_KEYS = {
    "original_system_prompt",
    "tools",
    "conversation_history",
    "current_user_message",
}


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def classification_regex():
    return "(" + "|".join(re.escape(canonical({"task_type": t})) for t in TASK_TYPES) + ")"


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


def parse_prediction(text):
    obj = strict_json(text)
    if not isinstance(obj, dict) or set(obj) != {"task_type"}:
        raise ValueError("Expected exactly the task_type field")
    if not isinstance(obj["task_type"], str) or obj["task_type"] not in TASK_TYPES:
        raise ValueError("Illegal task_type")
    return obj["task_type"]


def validate_input(payload):
    if not isinstance(payload, dict) or set(payload) != INPUT_KEYS:
        raise ValueError(
            "Router input must contain exactly the four public context fields"
        )
    if not isinstance(payload["original_system_prompt"], str):
        raise TypeError("System prompt must be text")
    if (
        not isinstance(payload["current_user_message"], str)
        or not payload["current_user_message"].strip()
    ):
        raise ValueError("A non-empty first user message is required")
    if not isinstance(payload["tools"], list) or any(
        not isinstance(t, dict) for t in payload["tools"]
    ):
        raise ValueError("Tools must be a list of schema objects")
    history = payload["conversation_history"]
    if not isinstance(history, list):
        raise TypeError("History must be a list")
    # At first-user routing, only an environment's initial assistant greeting
    # may precede the request. No previous user/tool turns or future messages.
    for message in history:
        if (
            set(message) != {"role", "content"}
            or message["role"] != "assistant"
            or not isinstance(message["content"], str)
        ):
            raise ValueError(
                "Only initial assistant greetings may precede first-user routing"
            )
    canonical(payload)
    return payload


def first_user_input(messages, tools=None):
    """Cut at the first non-empty user message; never inspect later turns."""
    if not isinstance(messages, list):
        raise TypeError("Expected a chat message list")
    system, history = [], []
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("content"), str):
            raise TypeError("Expected text chat messages")
        role, content = m.get("role"), m["content"]
        if role == "system":
            system.append(content)
        elif role == "assistant":
            history.append({"role": role, "content": content})
        elif role == "user":
            if content.strip():
                return validate_input(
                    {
                        "original_system_prompt": "\n\n".join(system),
                        "tools": tools if tools is not None else [],
                        "conversation_history": history,
                        "current_user_message": content,
                    }
                )
        else:
            raise ValueError(f"Unexpected role before first user: {role}")
    raise ValueError("No effective user request")


def router_messages(payload, prompt):
    # Send only the visible context fields to the classifier.
    validate_input(payload)
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": canonical(payload)},
    ]
