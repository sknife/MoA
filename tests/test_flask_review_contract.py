"""Characterization tests for FLASK/gpt_review/gpt4_eval.py.

parse_score lowercases keys and normalizes "na"/"n/a"/"not applicable" to
"N/A". gen_prompt always uses reviewer index 1 (prompt_id 2 in the checked-in
data), regardless of what reviewer 0 says.
"""
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "FLASK" / "gpt_review" / "gpt4_eval.py"

spec = importlib.util.spec_from_file_location("moa_gpt4_eval", SCRIPT)
gpt4_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpt4_eval)


def load_jsonl(rel):
    return [
        json.loads(line)
        for line in (REPO / rel).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_parse_score_lowercases_keys_and_normalizes_na():
    review = (
        "Reasoning here.\n"
        '{"Logical_Robustness": 5, "Factuality": "na", '
        '"Readability": "not applicable"}'
    )

    assert gpt4_eval.parse_score(review, 3) == {
        "logical robustness": 5,
        "factuality": "N/A",
        "readability": "N/A",
    }


def test_parse_score_uses_last_json_block_and_returns_empty_on_bad_input():
    assert gpt4_eval.parse_score(
        '{"First": 1} trailing {"Comprehension": 4}', 3
    ) == {"comprehension": 4}
    assert gpt4_eval.parse_score("no dictionary at all", 3) == {}
    assert gpt4_eval.parse_score('{"Comprehension": 4', 3) == {}


def test_gen_prompt_uses_reviewer_index_1_and_embeds_skills_and_response():
    reviewers = load_jsonl("FLASK/gpt_review/src/reviewer.jsonl")
    prompts = load_jsonl("FLASK/gpt_review/src/prompt.jsonl")
    skills = json.loads(
        (REPO / "FLASK/metadata_annotation/skillset/src/skillset_description.json")
        .read_text(encoding="utf-8")
    )
    item = load_jsonl("FLASK/evaluation_set/flask_evaluation.jsonl")[0]

    system_prompt, prompt = gpt4_eval.gen_prompt(
        reviewers, prompts, skills, "ASSISTANT RESPONSE", item
    )

    expected = prompts[reviewers[1]["prompt_id"] - 1]
    assert system_prompt == expected["system_prompt"]
    assert "ASSISTANT RESPONSE" in prompt
    assert item["instruction"] in prompt
    assert item["answer"] in prompt
    # At least one skill in the checked-in item is expanded with its rubric.
    assert "Score 1:" in prompt
    assert "Score 5:" in prompt
