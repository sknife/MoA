"""Characterization tests for generate_for_alpaca_eval.process_fn.

These tests assert the current call sequence. Two details are intentionally
pinned because they are easy to break:
- the final aggregator call does not forward temperature/max_tokens;
- an item with a pre-existing ``references`` list skips reference models.
"""
import generate_for_alpaca_eval as alpaca


def test_process_fn_calls_reference_models_then_aggregator(monkeypatch):
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return "final" if kwargs["model"] == "agg" else "out-" + kwargs["model"]

    monkeypatch.setattr(alpaca, "generate_with_references", fake_generate)

    result = alpaca.process_fn(
        {"instruction": "question"},
        model="agg",
        reference_models=["ref1", "ref2"],
        temperature=0.3,
        max_tokens=123,
        rounds=1,
    )

    assert result == {"output": "final", "generator": "agg-together"}
    assert [call["model"] for call in calls] == ["ref1", "ref2", "agg"]
    assert calls[0]["references"] == []
    assert calls[1]["references"] == []
    assert calls[2]["references"] == ["out-ref1", "out-ref2"]
    assert calls[0]["messages"] == [{"role": "user", "content": "question"}]
    assert calls[0]["temperature"] == 0.3
    assert calls[0]["max_tokens"] == 123
    # Current behavior: the aggregator call relies on generate_with_references
    # defaults and does not forward the caller's sampling parameters.
    assert set(calls[2]) == {"model", "messages", "references"}


def test_process_fn_uses_preset_references_and_skips_reference_models(monkeypatch):
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return "final"

    monkeypatch.setattr(alpaca, "generate_with_references", fake_generate)

    result = alpaca.process_fn(
        {"instruction": "question", "references": ["preset"]},
        model="agg",
        reference_models=["ref1"],
        rounds=2,
    )

    assert result == {"output": "final", "generator": "agg-together"}
    assert len(calls) == 1
    assert calls[0]["model"] == "agg"
    assert calls[0]["references"] == ["preset"]
