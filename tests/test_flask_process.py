"""Characterization tests for generate_for_flask.process_fn."""
import pytest

import generate_for_flask as flask


def test_provider_selects_generate_fn_and_output_has_only_text(monkeypatch):
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return "answer"

    monkeypatch.setattr(flask, "generate_with_references", fake_generate)

    together_result = flask.process_fn(
        {"question_id": 7, "text": "question"},
        model="agg",
        reference_models=[],
        provider="together",
    )
    openai_result = flask.process_fn(
        {"question_id": 8, "text": "question"},
        model="agg",
        reference_models=[],
        provider="openai",
    )

    assert together_result == {"text": "answer"}
    assert openai_result == {"text": "answer"}
    assert calls[0]["generate_fn"] is flask.generate_together
    assert calls[1]["generate_fn"] is flask.generate_openai
    assert calls[0]["messages"] == [{"role": "user", "content": "question"}]


def test_invalid_provider_asserts_and_none_references_are_filtered(monkeypatch):
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        if kwargs["model"] == "ref1":
            return None
        if kwargs["model"] == "ref2":
            return "usable"
        return "final"

    monkeypatch.setattr(flask, "generate_with_references", fake_generate)

    with pytest.raises(AssertionError):
        flask.process_fn(
            {"text": "question"},
            model="agg",
            reference_models=[],
            provider="anthropic",
        )
    assert calls == []

    result = flask.process_fn(
        {"text": "question"},
        model="agg",
        reference_models=["ref1", "ref2"],
        provider="together",
    )

    assert result == {"text": "final"}
    assert calls[0]["model"] == "ref1"
    assert calls[1]["model"] == "ref2"
    assert calls[2]["model"] == "agg"
    assert calls[2]["references"] == ["usable"]
