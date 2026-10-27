"""End-to-end characterization of FLASK/gpt_review/aggregate_skill.py.

The script is import-time executable, so it is run as a subprocess in a
temporary working directory. It writes outputs/stats/<name>_skill.csv and
silently drops score keys outside its 12-skill key_order.
"""
import csv
import json
import subprocess
import sys


def test_aggregate_skill_writes_expected_csv(root, tmp_path):
    review_path = tmp_path / "review.jsonl"
    rows = [
        {"score": {"comprehension": "5", "factuality": "N/A", "unknown": "4"}},
        {"score": {"comprehension": "3", "factuality": "1"}},
    ]
    review_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    script = root / "FLASK" / "gpt_review" / "aggregate_skill.py"
    result = subprocess.run(
        [sys.executable, str(script), "-m", str(review_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    csv_path = tmp_path / "outputs" / "stats" / "review_skill.csv"
    assert csv_path.exists()
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {"skill": "factuality", "score": "1.0", "count": "1", "avg": "1.0"},
        {"skill": "comprehension", "score": "8.0", "count": "2", "avg": "4.0"},
    ]
