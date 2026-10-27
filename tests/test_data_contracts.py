"""Data-contract tests for the checked-in benchmark files.

These are not unit tests of application logic: they pin the row counts and
key fields that the generation / review scripts silently assume. A changed
dataset should fail here before it produces misaligned reviews.
"""
import json
from collections import Counter


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_flask_evaluation_set_contract(root):
    rows = load_jsonl(root / "FLASK/evaluation_set/flask_evaluation.jsonl")

    assert len(rows) == 1740
    assert [row["idx"] for row in rows] == list(range(1, 1741))
    assert all(
        {"idx", "instruction", "answer", "skill", "domain", "difficulty"} <= set(row)
        for row in rows
    )
    assert all(isinstance(row["skill"], list) and 2 <= len(row["skill"]) <= 3 for row in rows)
    assert all(isinstance(row["domain"], list) and row["domain"] for row in rows)

    difficulty = Counter(str(row["difficulty"]) for row in rows)
    assert difficulty == {
        "formal education knowledge": 444,
        "major level knowledge": 442,
        "simple lifestyle knowledge": 388,
        "advanced lifestyle knowledge": 278,
        "expert level knowledge": 186,
        "-1": 2,
    }
    # The two sentinel rows are integer -1, not the string "-1".
    assert [(row["idx"], row["difficulty"]) for row in rows if row["difficulty"] == -1] == [
        (1706, -1),
        (1716, -1),
    ]


def test_flask_sources_and_mt_bench_contract(root):
    raw = load_jsonl(root / "FLASK/input_data/flask_evaluation_raw.jsonl")
    assert len(raw) == 1700
    assert [row["question_id"] for row in raw] == list(range(1, 1701))

    hard = load_jsonl(root / "FLASK/evaluation_set/flask_hard_evaluation.jsonl")
    assert len(hard) == 89
    assert len({row["idx"] for row in hard}) == 89

    questions = load_jsonl(
        root / "FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl"
    )
    assert len(questions) == 80
    assert [row["question_id"] for row in questions] == list(range(81, 161))
    assert all(len(row["turns"]) == 2 for row in questions)

    judge_prompts = load_jsonl(
        root / "FastChat/fastchat/llm_judge/data/judge_prompts.jsonl"
    )
    assert len(judge_prompts) == 8
    assert {row["name"] for row in judge_prompts} == {
        "pair-v2",
        "pair-v2-multi-turn",
        "pair-math-v1",
        "pair-math-v1-multi-turn",
        "single-v1",
        "single-math-v1",
        "single-v1-multi-turn",
        "single-math-v1-multi-turn",
    }

    reviewers = load_jsonl(root / "FLASK/gpt_review/src/reviewer.jsonl")
    assert [row["prompt_id"] for row in reviewers] == [1, 2]
