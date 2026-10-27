"""CLI-surface smoke tests.

This repository has no HTTP server, so the public interface is the CLI. Each
entry point must at least import its dependencies and render help without a
network call or API key.
"""
import subprocess
import sys

import pytest

CASES = [
    ("generate_for_alpaca_eval.py", "root", ["--help"]),
    ("generate_for_flask.py", "root", ["--help"]),
    ("generate_for_mt_bench.py", "root", ["--help"]),
    ("eval_mt_bench.py", "root", ["--help"]),
    ("show_mt_bench_result.py", "root", ["--help"]),
    ("bot.py", "root", ["--help"]),
    ("FLASK/gpt_review/gpt4_eval.py", "gpt_review", ["-h"]),
]


@pytest.mark.parametrize("script,cwd_name,args", CASES)
def test_entrypoint_help_exits_zero(root, script, cwd_name, args):
    cwd = root if cwd_name == "root" else root / "FLASK" / "gpt_review"
    result = subprocess.run(
        [sys.executable, str(root / script), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )

    combined = (result.stdout + result.stderr).lower()
    assert result.returncode == 0, combined
    assert "usage" in combined or "synopsis" in combined, combined
