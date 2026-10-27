"""Characterization tests for the MoA prompt/dispatch layer.

Assertions describe what utils.py does today, including the exact reference
numbering and the fact that the aggregator call receives the caller's
messages unchanged when there are no references.
"""
import copy

import utils


def test_inject_references_prepends_system_message_and_keeps_input_immutable():
    messages = [{"role": "user", "content": "hello"}]
    original = copy.deepcopy(messages)

    result = utils.inject_references_to_messages(messages, ["first answer", "second answer"])

    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert result[0]["content"].startswith(
        "You have been provided with a set of responses"
    )
    assert result[0]["content"].endswith(
        "Responses from models:\n1. first answer\n2. second answer"
    )
    assert result[1] == {"role": "user", "content": "hello"}
    assert messages == original
    assert result is not messages


def test_inject_references_appends_to_existing_system_message():
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hello"},
    ]

    result = utils.inject_references_to_messages(messages, ["only answer"])

    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert result[0]["content"].startswith("SYS\n\nYou have been provided")
    assert result[0]["content"].endswith("Responses from models:\n1. only answer")


def test_generate_with_references_forwards_kwargs_and_only_injects_when_needed():
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return "aggregated"

    messages = [{"role": "user", "content": "hello"}]
    output = utils.generate_with_references(
        "aggregator",
        messages,
        references=[],
        max_tokens=77,
        temperature=0.25,
        generate_fn=fake_generate,
    )

    assert output == "aggregated"
    assert calls[0]["model"] == "aggregator"
    assert calls[0]["messages"] is messages
    assert calls[0]["max_tokens"] == 77
    assert calls[0]["temperature"] == 0.25

    output = utils.generate_with_references(
        "aggregator",
        messages,
        references=["ref"],
        generate_fn=fake_generate,
    )

    assert output == "aggregated"
    assert calls[1]["messages"] is not messages
    assert calls[1]["messages"][0]["role"] == "system"
    assert "1. ref" in calls[1]["messages"][0]["content"]
    assert calls[1]["max_tokens"] == 2048
    assert calls[1]["temperature"] == 0.7
