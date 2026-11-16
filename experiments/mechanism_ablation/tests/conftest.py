"""Make the repo root, tests/ (sw_lmmd_fixtures) and the reference-ablation tests (its toy store and
trainer builders) importable."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for path in (REPO, os.path.join(REPO, "tests"), os.path.join(REPO, "experiments", "ref_ablation", "tests")):
    if path not in sys.path:
        sys.path.insert(0, path)
