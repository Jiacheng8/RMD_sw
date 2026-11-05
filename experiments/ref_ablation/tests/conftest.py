"""Make the repo root (``experiments``) and the repo's tests/ (``sw_lmmd_fixtures``) importable."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for path in (REPO, os.path.join(REPO, "tests")):
    if path not in sys.path:
        sys.path.insert(0, path)
