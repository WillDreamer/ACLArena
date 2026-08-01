"""8B-specific reward scorer for Search-R1.

This exists ONLY so the 8B run can use a relaxed scoring table without
touching ``qa_em_format_qwen_sft.py`` (which the 30B run still imports via
``generate_with_search_tools_qwen_sft.py``). The format / parsing helpers are
imported from ``qa_em_format_qwen_sft`` rather than duplicated, so "valid
format", "executed tool", "answer correct" and the metrics in ``reward_func``
stay byte-for-byte consistent between the two runs.

Difference vs ``qa_em_format_qwen_sft.compute_score_em_sft``
------------------------------------------------------------
Correctness is the PRIMARY signal and is checked FIRST, so a correct final
``<answer>`` is NEVER vetoed by truncation or by the strict ``valid_format``
gate. In the 8B search RL run ``valid_format`` was only ~4-8%, and the
truncated short-circuit forced ``-0.2`` on correct-but-truncated samples, so
together they discarded the positive signal from ~90% of correct answers
(only 8/512 correct answers earned reward > 0 by late training). Here a
correct answer always earns positive reward, graduated by how clean the
trajectory was; the executed-tool ordering keeps a real-search incentive, and
the "answer directly, skip search" shortcut is still handled by the
dynamic-sampling loss-mask filter (``no_tool_loss_mask_filter``), not by
zeroing the reward.

Scoring table (values intentionally simple — tune as needed):
   answer_correct + valid_format + executed_tool   ->  1.0
   answer_correct + executed_tool (format bad)     ->  0.7
   answer_correct + valid_format  (no tool)        ->  0.5
   answer_correct (format bad, no tool)            ->  0.3
   answer_wrong   + valid_format + retrieved       ->  0.4
   answer_wrong   + valid_format + executed (no hit)-> 0.1
   answer_wrong   + valid_format + no_tool         -> -0.05
   answer_wrong   + format_bad                     -> -0.2
   truncated (executed tool, no answer extracted)  -> -0.05
   truncated (otherwise)                            -> -0.2
"""

import random

from qa_em_format_qwen_sft import (  # type: ignore
    em_check,
    executed_tool_call,
    extract_solution,
    is_retrieval_correct,
    is_valid_sequence,
)


def _maybe_print(do_print_prob, ground_truth, answer, solution_str, tag, score):
    if do_print_prob > 0 and random.randint(1, do_print_prob) == 1:
        print("--------------------------------")
        print(f"[{tag}] score={score}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")


def compute_score_em_sft(
    solution_str,
    ground_truth,
    status: str | None = None,
    response: str | None = None,
    do_print_prob: int = 64,
):
    """8B relaxed scorer. Same signature as
    ``qa_em_format_qwen_sft.compute_score_em_sft`` so it is a drop-in for the
    8B ``reward_func``.

    Returns a float reward in [-0.2, 1.0].
    """
    is_valid_format, _ = is_valid_sequence(solution_str)
    answer = extract_solution(solution_str)
    answer_correct = bool(answer and em_check(answer, ground_truth["target"]))
    # Only the assistant trajectory may contribute a "real" tool execution
    # (the prompt's instruction block contains a literal tag pair as docs).
    exec_text = response if response is not None else solution_str
    executed = executed_tool_call(exec_text)
    retrieval_correct = (
        is_retrieval_correct(exec_text, ground_truth["target"]) if executed else False
    )

    # ---- Correctness is PRIMARY and never vetoed by truncation/format.
    if answer_correct:
        if is_valid_format and executed:
            score = 1.0   # correct + valid format + real search
        elif executed:
            score = 0.7   # correct + real search, format imperfect
        elif is_valid_format:
            score = 0.5   # correct + clean format, but no search
        else:
            score = 0.3   # correct answer only (format broken, no search)
        _maybe_print(
            do_print_prob, ground_truth, answer, solution_str, tag="OK", score=score
        )
        return score

    # ---- Truncated with NO correct answer: no positive shaping reward.
    is_truncated = isinstance(status, str) and "truncated" in status.lower()
    if is_truncated:
        # Tiny relief if the model at least drove a real search but ran out of
        # budget before emitting the final <answer>.
        if executed and answer is None:
            score = -0.05
        else:
            score = -0.2
        _maybe_print(
            do_print_prob, ground_truth, answer, solution_str,
            tag="TRUNC", score=score,
        )
        return score

    # ---- Wrong answer, not truncated.
    if is_valid_format and retrieval_correct:
        score = 0.4
    elif is_valid_format and executed:
        score = 0.1
    elif is_valid_format and not executed:
        # Format OK but never called the tool: tiny negative to discourage the
        # "format-only gaming" attractor.
        score = -0.05
    else:
        score = -0.2

    _maybe_print(
        do_print_prob, ground_truth, answer, solution_str, tag="OK", score=score
    )
    return score
