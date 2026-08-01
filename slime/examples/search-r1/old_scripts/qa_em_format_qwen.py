import json
import random
import re
import string


def _parse_bare_json_tool_call(text: str):
    """Fallback: match bare {"name": "search", "arguments": {...}} without <tool_call> tags.

    Returns (action, content) where action is 'search' or None.
    """
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
    # Find the position of "<|im_start|>assistant" with potential whitespace
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)

    if not assistant_match:
        return False, "Missing assistant marker"

    # Extract the content after the assistant marker
    start_pos = assistant_match.end()
    content = text[start_pos:]

    # Check for balanced tags (no longer require think tags)
    tags_to_check = ["tool_call", "tool_response", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return (
                False,
                f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags",
            )

    # Now check for proper sequence pattern and no extraneous content

    # 1. First split the content by any tags we recognize (without think)
    split_pattern = r"(</?(?:tool_call|tool_response|answer)>)"
    parts = re.split(split_pattern, content)

    # 2. Keep track of the current position in the expected sequence
    # start -> [tool_call -> tool_response ->]* answer -> end
    state = "start"

    # 3. Check each part
    for _i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue

        # Check if this is a tag
        if re.match(r"</?(?:tool_call|tool_response|answer)>", part):
            # This is a tag, check if it's valid in the current state
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
            # This is content, check if it's valid in the current state
            if state in ["in_tool_call", "in_tool_response", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_tool_call", "tool_response", "end"]:
                # Allow free-form content outside tags (e.g. reasoning without <think>)
                # Also allow trailing content after </answer> (e.g. <|endoftext|>)
                pass
            else:
                return False, f"Unexpected content in state {state}"

    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"


def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<tool_response>(.*?)</tool_response>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def compute_score_em(
    solution_str,
    ground_truth,
    method="strict",
    structure_format_score=0,
    final_format_score=0,
    retrieval_score=0,
    format_score=0,
    score=1.0,
):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    is_valid_format, _ = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth["target"])
    answer = extract_solution(solution_str=solution_str)
    has_tool_call = (
        "<tool_call>" in solution_str
        or _parse_bare_json_tool_call(solution_str)[0] is not None
    )
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        if is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score  # 0.3
            else:
                return structure_format_score  # 0.2
        else:
            return final_format_score # -0.1
    else:
        if em_check(answer, ground_truth["target"]):
            if is_valid_format:
                if has_tool_call:
                    return score  # 1
                else:
                    return 0  # no tool_call → same as worst (-0.1)
            else:
                return 0.6 * score - structure_format_score  # 0.2
        elif is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score  # 0.4
            elif has_tool_call:
                return structure_format_score + 0.1 # 0.3
            else:
                return 0  # 降低到0，不给纯<answer>的序列打分
        else:
            if has_tool_call:
                return structure_format_score # 鼓励调用工具
            else:   
                return final_format_score  # -0.1

