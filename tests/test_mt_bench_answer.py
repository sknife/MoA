"""Characterization tests for generate_for_mt_bench.get_answer.

The module reads a module-global ``args`` object that is only created under
``__main__``; tests install a minimal namespace, which is what the script does
at runtime. The None-aggregator test pins defect B7 from
docs/consistency-report.md: a failed aggregator call raises AttributeError
instead of being recorded as a failed answer.
"""
import json
from types import SimpleNamespace

import pytest

import generate_for_mt_bench as mt


def test_get_answer_writes_fastchat_answer_schema_and_appends(monkeypatch, tmp_path):
    mt.args = SimpleNamespace(force_temperature=None)
    calls = []

    def fake_generate(**kwargs):
        # get_answer keeps appending to the same messages list after the call,
        # so snapshot it to observe the arguments as they were at call time.
        calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return f"answer-{len(calls)}"

    monkeypatch.setattr(mt, "generate_with_references", fake_generate)

    question = {
        "question_id": 81,
        "category": "writing",
        "turns": ["first turn", "second turn"],
    }
    answer_file = tmp_path / "answers.jsonl"

    mt.get_answer(
        question=question,
        model="test-model",
        reference_models=[],
        num_choices=1,
        max_tokens=64,
        answer_file=str(answer_file),
        rounds=1,
        provider="together",
    )
    mt.get_answer(
        question={**question, "question_id": 82},
        model="test-model",
        reference_models=[],
        num_choices=1,
        max_tokens=64,
        answer_file=str(answer_file),
        rounds=1,
        provider="together",
    )

    lines = [json.loads(line) for line in answer_file.read_text().splitlines()]
    assert len(lines) == 2
    first = lines[0]
    assert set(first) == {"question_id", "answer_id", "model_id", "choices", "tstamp"}
    assert first["question_id"] == 81
    assert first["model_id"] == "test-model"
    assert len(first["answer_id"]) == 22
    assert first["choices"] == [
        {"index": 0, "turns": ["answer-1", "answer-2"]}
    ]
    assert lines[1]["question_id"] == 82
    assert lines[1]["choices"][0]["turns"] == ["answer-3", "answer-4"]

    assert calls[0]["temperature"] == 0.7  # "writing" category
    assert calls[0]["max_tokens"] == 64
    assert calls[0]["messages"] == [{"role": "user", "content": "first turn"}]
    assert calls[1]["messages"] == [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "answer-1"},
        {"role": "user", "content": "second turn"},
    ]


def test_get_answer_crashes_when_aggregator_returns_none(monkeypatch, tmp_path):
    mt.args = SimpleNamespace(force_temperature=None)
    monkeypatch.setattr(mt, "generate_with_references", lambda **kwargs: None)

    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'strip'"):
        mt.get_answer(
            question={"question_id": 81, "category": "writing", "turns": ["q"]},
            model="test-model",
            reference_models=[],
            num_choices=1,
            max_tokens=64,
            answer_file=str(tmp_path / "answers.jsonl"),
            rounds=1,
            provider="together",
        )
