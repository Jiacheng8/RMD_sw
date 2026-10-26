"""Make the repo root (``rdm``), the repo's tests/ (``sw_lmmd_fixtures``) and ``z-image/``
(``zimage_distill``) importable."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ZIMAGE = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ZIMAGE, ".."))
for path in (REPO, os.path.join(REPO, "tests"), ZIMAGE):
    if path not in sys.path:
        sys.path.insert(0, path)
