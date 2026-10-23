"""Route recognizable tool environments; defer ambiguous capabilities to the classifier."""

import re

from mle.core import validate_input

_RETRIEVAL = {"search", "web_search", "search_web", "search_engine", "retrieve"}
_UTILITY = {"calculate", "calculator", "think"}
_BUSINESS_NOUNS = {
    "account",
    "bill",
    "bills",
    "booking",
    "customer",
    "customers",
    "inventory",
    "item",
    "items",
    "line",
    "order",
    "payment",
    "product",
    "reservation",
    "roaming",
    "task",
    "user",
    "users",
}
_IDENTIFIERS = {
    "account_id",
    "bill_id",
    "booking_id",
    "customer_id",
    "item_id",
    "item_ids",
    "line_id",
    "order_id",
    "payment_method_id",
    "product_id",
    "reservation_id",
    "task_id",
    "user_id",
}
_MUTATIONS = {
    "cancel",
    "create",
    "delete",
    "disable",
    "enable",
    "exchange",
    "modify",
    "refuel",
    "refund",
    "resume",
    "return",
    "send",
    "suspend",
    "update",
}

def routing_rule(payload):
    """Use available tool schemas only; defer other requests to the classifier."""
    return tool_environment(payload)


def tool_environment(payload):
    """Return (task_type, rule_id) or None, using only supplied function schemas.

    A business environment needs both an identity/support capability and a
    state-changing operation tied to a business identifier. Search must expose
    a query parameter. Unknown or mixed capabilities defer to model judgment.
    Names merely mentioned in prose are never interpreted as available tools.
    """
    validate_input(payload)
    if not payload["tools"]:
        return None
    retrieval = business = identity = mutation = False
    for tool in payload["tools"]:
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            return None
        function = tool["function"]
        name = function.get("name", "")
        parameters = function.get("parameters", {})
        if not isinstance(name, str) or not isinstance(parameters, dict):
            return None
        properties = parameters.get("properties", {})
        if not isinstance(properties, dict):
            return None
        words = set(re.findall(r"[a-z]+", name.lower()))
        fields = set(properties)
        if name in _UTILITY:
            continue
        if name in _RETRIEVAL and fields & {"query", "queries", "search_query"}:
            retrieval = True
        elif {"transfer", "human"} <= words:
            business = identity = True
        elif words & _BUSINESS_NOUNS or fields & _IDENTIFIERS:
            business = True
            identity |= bool(
                words & {"get", "find", "lookup", "list"}
                and words & {"user", "users", "customer", "customers", "account"}
            )
            mutation |= bool(words & _MUTATIONS and fields & _IDENTIFIERS)
        elif words & {"get", "lookup"} and "id" in fields:
            # A generic ID lookup is not sufficient evidence by itself.
            continue
        else:
            return None
    if retrieval and not business:
        return "search", "query_tools_only_v1"
    if business and identity and mutation and not retrieval:
        return "customer_service", "identity_and_business_mutation_v1"
    return None
