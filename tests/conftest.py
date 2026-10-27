import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FLASK = ROOT / "FLASK"
FLASK_REVIEW = FLASK / "gpt_review"

# gpt4_eval.py does sys.path.append("../") relative to its cwd; tests import
# it from the repo root, so add FLASK explicitly for openai_concurrent.
for path in (str(ROOT), str(FLASK), str(FLASK_REVIEW)):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture
def root():
    return ROOT
