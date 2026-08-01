"""SFT-friendly reward scoring for Search-R1.

Differences vs qa_em_format_qwen.py (see analysis of step-48 reward collapse):

1. has_tool_call -> executed_tool_call
   The old check just looked for the literal "<tool_call>" substring in the
   response. That created a free-loading attractor: a model could embed a
   fake tool_call JSON inside <think>, never actually invoke the tool, get
   truncated, and still receive +0.2. We now require that the trajectory
   contains a real <tool_response>...</tool_response> block (which can only
   appear if the environment actually executed the tool).

2. Truncated trajectories never receive a positive shaping reward.
   Handled in reward_func (generate_with_search_tools_qwen_sft.py). The
   scoring helper here exposes a `status` arg so the same scorer can be
   reused offline.

3. Reshuffled scoring table to widen the gap between "real search behaviour"
   and "format-only gaming":

      answer_correct + valid_format + executed_tool   ->  1.0
      answer_correct + valid_format + no_tool         ->  0.3
      answer_correct + format_bad                     ->  0.2
      answer_wrong   + valid_format + retrieved       ->  0.4
      answer_wrong   + valid_format + executed (no hit)-> 0.1
      answer_wrong   + valid_format + no_tool         -> -0.05
      answer_wrong   + format_bad                     -> -0.2
      truncated (executed tool, no answer extracted)  -> -0.05
      truncated (otherwise)                            -> -0.2
"""

import json
import random
import re
import string


# ---------------------------------------------------------------------------
# Parsers (unchanged from qa_em_format_qwen.py, kept here so this file is
# self-contained and can be imported on its own).
# ---------------------------------------------------------------------------

def _parse_bare_json_tool_call(text: str):
    """Fallback: match bare {"name": "search", "arguments": {...}} without <tool_call> tags."""
    pattern = r'\{[^{}]*"name"\s*:\s*"search"[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
    match = re.search(pattern, text)
    if not match:
        pattern = r'\{[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*"name"\s*:\s*"search"[^{}]*\}'
        match = re.search(pattern, text)
    if not match:
        return None, ""
    try:
        data = json.loads(match.group(0))
    except Exception:
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


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def is_valid_sequence(text):
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)
    if not assistant_match:
        return False, "Missing assistant marker"

    start_pos = assistant_match.end()
    content = text[start_pos:]

    tags_to_check = ["tool_call", "tool_response", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return (
                False,
                f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags",
            )

    split_pattern = r"(</?(?:tool_call|tool_response|answer)>)"
    parts = re.split(split_pattern, content)

    state = "start"
    for _i, part in enumerate(parts):
        if not part.strip():
            continue
        if re.match(r"</?(?:tool_call|tool_response|answer)>", part):
            if part == "<tool_call>" and state in ["start", "tool_response"]:
                state = "in_tool_call"
            elif part == "</tool_call>" and state == "in_tool_call":
                state = "after_tool_call"
            elif part == "<tool_response>" and state == "after_tool_call":
                state = "in_tool_response"
            elif part == "</tool_response>" and state == "in_tool_response":
                state = "tool_response"
            elif part == "<answer>" and state in ["start", "after_tool_call", "tool_response"]:
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            if state in ["in_tool_call", "in_tool_response", "in_answer"]:
                pass
            elif state in ["start", "after_tool_call", "tool_response", "end"]:
                pass
            else:
                return False, f"Unexpected content in state {state}"

    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"


def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if len(matches) <= 1:
        return None
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<tool_response>(.*?)</tool_response>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> bool:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


# ---------------------------------------------------------------------------
# Strict tool-call execution check (Fix #1)
# ---------------------------------------------------------------------------

def executed_tool_call(text: str) -> bool:
    """Return True iff the trajectory contains a *real* tool execution.

    A real execution is identified by a NON-EMPTY balanced
    <tool_response>...</tool_response> block. Two important guards:

    1. The string must be the assistant trajectory (sample.response), NOT
       sample.prompt + sample.response — the chat-template instruction in the
       prompt itself contains the literal "<tool_response></tool_response>"
       as a teaching example, which would otherwise make this always-True.
    2. We require the block to be non-empty: empty pairs (i.e. the prompt
       example) do not count even if the caller forgets guard #1.
    """
    if "<tool_response>" not in text or "</tool_response>" not in text:
        return False
    # Match a non-empty balanced block. The body must contain at least one
    # non-whitespace character — empty <tool_response></tool_response> pairs
    # (as found in the instruction prompt) do not count.
    return bool(re.search(r"<tool_response>\s*\S[\s\S]*?</tool_response>", text))


# ---------------------------------------------------------------------------
# Reward scorer (Fix #3 — new table; Fix #2 — `status` short-circuit)
# ---------------------------------------------------------------------------

def compute_score_em_sft(
    solution_str,
    ground_truth,
    status: str | None = None,
    response: str | None = None,
    do_print_prob: int = 64,
):
    """New scoring function for Search-R1 SFT-then-RL pipeline.

    Args:
        solution_str:  full prompt + response text (used by is_valid_sequence
                       which itself anchors on the assistant marker).
        ground_truth:  dict with key 'target' (list of acceptable answers).
        status:        optional sample status string. If equal to "truncated"
                       (case-insensitive substring), apply Fix #2 short-circuit.
        response:      assistant-only text. REQUIRED for a correct executed-tool
                       check, because the instruction prompt itself contains a
                       literal '<tool_response></tool_response>' example that
                       would otherwise always satisfy the substring test.
                       Falls back to solution_str if not provided (legacy).
        do_print_prob: 1/N probability of debug-printing this sample.

    Returns:
        float reward in [-0.2, 1.0].
    """
    is_valid_format, _ = is_valid_sequence(solution_str)
    answer = extract_solution(solution_str)
    answer_correct = bool(answer and em_check(answer, ground_truth["target"]))
    # Critical: only the assistant trajectory may contribute a "real" tool
    # execution. The prompt's instruction block contains the literal tag
    # pair as documentation, so checking solution_str would always be True.
    exec_text = response if response is not None else solution_str
    executed = executed_tool_call(exec_text)
    retrieval_correct = (
        is_retrieval_correct(exec_text, ground_truth["target"]) if executed else False
    )

    # ---- Fix #2: truncated trajectories receive no positive shaping reward.
    is_truncated = isinstance(status, str) and "truncated" in status.lower()
    if is_truncated:
        # Tiny relief if the model at least managed to drive a real search
        # but ran out of budget before emitting the final <answer>.
        if executed and answer is None:
            score = -0.05
        else:
            score = -0.2
        _maybe_print(
            do_print_prob, ground_truth, answer, solution_str,
            tag="TRUNC", score=score,
        )
        return score

    # ---- Fix #3: new scoring table.
    if answer_correct:
        if is_valid_format and executed:
            score = 1.0
        elif is_valid_format and not executed:
            # Correct answer without ever calling the tool — neutral reward.
            # Loss-mask gating in the dynamic filter handles gradient flow,
            # so the reward only needs to avoid creating a positive signal
            # toward the "answer directly, skip search" shortcut.
            score = 0.0
        else:
            # Correct answer, format broken.
            score = 0.0
    else:
        if is_valid_format and retrieval_correct:
            score = 0.4
        elif is_valid_format and executed:
            score = 0.1
        elif is_valid_format and not executed:
            # Format OK but never called the tool: tiny negative to discourage
            # the "format-only gaming" attractor.
            score = -0.05
        else:
            score = -0.2

    _maybe_print(
        do_print_prob, ground_truth, answer, solution_str, tag="OK", score=score
    )
    return score


def _maybe_print(do_print_prob, ground_truth, answer, solution_str, tag, score):
    if do_print_prob > 0 and random.randint(1, do_print_prob) == 1:
        print("--------------------------------")
        print(f"[{tag}] score={score}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
