#!/usr/bin/env python
"""Tabulate Qwen-Image-Bench runs next to the paper's leaderboard.

    PYTHONPATH=z-image python -m eval_qib.report <run dir> [<run dir> ...] [--out report.md]

A run dir is what run_eval.sh writes: ``generation.json`` (what rendered the images) and the
judge's ``input_bench_scores.json`` (L1 pillars, L2 sub-capabilities, total, 0-100).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_qib.bench import LEADERBOARD, PILLARS

SHORT = {"Quality": "Quality", "Aesthetics": "Aesth.", "Alignment": "Align.",
         "Real-world Fidelity": "RW Fid.", "Creative Generation": "Creative"}


def fmt(v) -> str:
    return "–" if v is None else f"{v:.2f}"


def load(run: Path) -> dict:
    scores = json.loads((run / "input_bench_scores.json").read_text())
    gen = json.loads((run / "generation.json").read_text()) if (run / "generation.json").exists() \
        else {}
    judged = run / "input_judged.jsonl"
    parse_fail = None
    if judged.exists():
        rows = [json.loads(line) for line in open(judged, encoding="utf-8") if line.strip()]
        parse_fail = sum(1 for r in rows if not r.get("judge_model_output"))
    return {"name": run.name, "scores": scores, "gen": gen, "unparsed_images": parse_fail}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--leaderboard", default="Qwen Image 2512,FLUX 2 Pro,HunyuanImage 3.0,Qwen Image,GPT Image 2",
                    help="paper rows to show for reference (comma list; 'all' for every row)")
    a = ap.parse_args()
    runs = [load(r) for r in a.runs]
    head = "| model | " + " | ".join(SHORT[p] for p in PILLARS) + " | **Total** |"
    lines = ["# Qwen-Image-Bench", "",
             "Judge: Q-Judger (Qwen3.6-27B), official `judge.py`; 1 image per prompt, 1000 prompts.",
             "Scores 0-100 (facet Fail/Pass/Excel = 0/60/100, averaged facet -> L2 -> L1 -> total).",
             "", head, "|---" * (len(PILLARS) + 2) + "|"]
    for r in runs:
        l1 = r["scores"]["level1"]
        lines.append(f"| {r['name']} | " + " | ".join(fmt(l1.get(p)) for p in PILLARS)
                     + f" | **{fmt(r['scores']['total'])}** |")
    rows = list(LEADERBOARD) if a.leaderboard == "all" else \
        [x.strip() for x in a.leaderboard.split(",") if x.strip()]
    for name in rows:
        v = LEADERBOARD[name]
        lines.append(f"| *{name} (paper)* | " + " | ".join(f"{x:.2f}" for x in v[:5])
                     + f" | *{v[5]:.2f}* |")
    lines += ["", "## Runs", ""]
    for r in runs:
        g = r["gen"]
        what = g.get("ckpt") or (f"Z-Image-Turbo, {g.get('teacher_steps')} steps"
                                 if g.get("teacher_steps") else "?")
        sig = g.get("sampling_args", {})
        lines.append(f"- **{r['name']}**: {what}; sigmas {[round(s, 4) for s in sig.get('sigmas', [])]}"
                     f", renoise {sig.get('renoise')}; prompts {g.get('lang', '?')}, seed "
                     f"{g.get('seed', '?')}; images without a parsed judgement: "
                     f"{r['unparsed_images'] if r['unparsed_images'] is not None else '?'}")
    lines += ["", "## Sub-capabilities (L2)", ""]
    for p in PILLARS:
        l2_names = sorted({k for r in runs for k in (r["scores"]["level2"].get(p) or {})})
        if not l2_names:
            continue
        lines += [f"**{p}**", "", "| model | " + " | ".join(l2_names) + " |",
                  "|---" * (len(l2_names) + 1) + "|"]
        for r in runs:
            l2 = r["scores"]["level2"].get(p) or {}
            lines.append(f"| {r['name']} | " + " | ".join(fmt(l2.get(n)) for n in l2_names) + " |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    if a.out:
        a.out.write_text(text, encoding="utf-8")
        print(f"[report] -> {a.out}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
