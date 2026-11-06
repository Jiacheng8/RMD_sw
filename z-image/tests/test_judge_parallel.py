"""eval_qib/judge_parallel.py: telling answers judged with thinking from ones judged without."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval_qib"))
from judge_parallel import thinking_stats, thought  # noqa: E402

JSON = '{"Resolution": {"Resolution": {"score": 1}}}'


def test_an_empty_think_block_is_no_thinking():
    assert not thought("<think>\n\n</think>\n\n" + JSON)          # ms-swift 4.x without the switch
    assert not thought(JSON)                                       # no think block at all
    assert thought("<think>\n**Resolution**: sharp -> Score: 1\n</think>\n\n" + JSON)


def test_stats_count_every_pillar_answer():
    rows = [{"ID": 1, "quality_judge_output": "<think>\nok\n</think>" + JSON,
             "alignment_judge_output": "<think>\n\n</think>" + JSON,
             "creative_generation_judge_output": None},
            {"ID": 2, "aesthetics_judge_output": "<think>\nok\n</think>" + JSON}]
    assert thinking_stats(rows) == (3, 1)
