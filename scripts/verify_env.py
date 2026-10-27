#!/usr/bin/env python3
"""Check the local MoA test environment without making network calls.

Exit code 0 means: every module needed by the offline test suite imports,
all data files used by contract tests exist, and their row counts match the
documented baseline. API keys are reported but never required.
"""
import argparse
import importlib
import importlib.metadata as md
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODULES = [
    ("openai", True),
    ("requests", True),
    ("loguru", True),
    ("fire", True),
    ("datasets", True),
    ("typer", True),
    ("rich", True),
    ("together", True),
    ("shortuuid", True),
    ("pytest", True),
    ("torch", True),
    ("transformers", True),
    ("fastchat", True),
    ("pandas", True),
    ("numpy", True),
]

DATA_FILES = [
    ("FLASK/evaluation_set/flask_evaluation.jsonl", 1740),
    ("FLASK/input_data/flask_evaluation_raw.jsonl", 1700),
    ("FLASK/evaluation_set/flask_hard_evaluation.jsonl", 89),
    ("FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl", 80),
    ("FastChat/fastchat/llm_judge/data/judge_prompts.jsonl", 8),
    ("FLASK/gpt_review/src/reviewer.jsonl", 2),
    ("FLASK/gpt_review/src/prompt.jsonl", 2),
]

KEYS = ["TOGETHER_API_KEY", "OPENAI_API_KEY", "OPENAI_API_KEYS"]


def check_modules():
    rows = []
    for name, required in MODULES:
        row = {"name": name, "required": required, "ok": False,
               "version": None, "error": None}
        try:
            importlib.import_module(name)
            row["ok"] = True
            try:
                row["version"] = md.version(name)
            except md.PackageNotFoundError:
                row["version"] = "unknown"
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def count_lines(path):
    with open(path, "rb") as handle:
        return sum(1 for line in handle if line.strip())


def check_data():
    rows = []
    for rel, expected in DATA_FILES:
        path = ROOT / rel
        row = {"path": rel, "expected": expected, "ok": False,
               "actual": None, "error": None}
        try:
            row["actual"] = count_lines(path)
            row["ok"] = row["actual"] == expected
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="print JSON only")
    parser.add_argument("--write", default=str(ROOT / ".deps" / "env.json"),
                        help="write a JSON report to this path ('' disables)")
    args = parser.parse_args()

    modules = check_modules()
    data = check_data()
    keys = {name: bool(os.environ.get(name)) for name in KEYS}
    failed_modules = [r["name"] for r in modules if r["required"] and not r["ok"]]
    failed_data = [r["path"] for r in data if not r["ok"]]
    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "venv": sys.prefix,
        "modules": modules,
        "data": data,
        "api_keys_present": keys,
        "ok": not failed_modules and not failed_data,
        "failed_modules": failed_modules,
        "failed_data": failed_data,
    }

    if args.write:
        out = Path(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"python {report['python']}  ({report['executable']})")
        print(f"venv   {report['venv']}")
        print("\nmodules:")
        for row in modules:
            state = "ok" if row["ok"] else "FAIL"
            version = row["version"] or ""
            extra = f"  {row['error']}" if row["error"] else ""
            print(f"  [{state:4}] {row['name']:14} {version}{extra}")
        print("\ndata files:")
        for row in data:
            state = "ok" if row["ok"] else "FAIL"
            print(f"  [{state:4}] {row['path']}  "
                  f"expected={row['expected']} actual={row['actual']}")
        print("\napi keys (presence only):")
        for name, present in keys.items():
            print(f"  {'set' if present else 'unset':5} {name}")
        print("\nRESULT:", "PASS" if report["ok"] else "FAIL")
        if failed_modules:
            print("missing modules:", ", ".join(failed_modules))
        if failed_data:
            print("bad data files:", ", ".join(failed_data))

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
