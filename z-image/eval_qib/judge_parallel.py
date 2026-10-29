#!/usr/bin/env python
"""Run the official judge on several model copies at once: split the input, merge the results.

    python judge_parallel.py split --input <run>/input.jsonl --parts N --metadata <bench>/metadata/bench_metadata.json
    (one official ``judge.py --input <run>/shards/part_K.jsonl`` per model copy, concurrently)
    python judge_parallel.py merge --run <run> --parts N --bench <bench>

The judge is 54.7 GB: one copy fits an 80 GB card, so N cards hold N copies that judge disjoint
images at the same time (``judge.py`` alone splits one copy over all visible cards layer by
layer, which runs them one after another). Every image is judged on its own with greedy
decoding, so the split does not change what the judge sees. ``split`` balances the parts by
judge calls (an image costs one call per level-1 pillar it tests, 3-5), not by image count.

``merge`` concatenates the parts' ``*_judged.jsonl`` in ID order into ``<run>/input_judged.jsonl``
and recomputes the bench scores with ``judge.py``'s own ``compute_bench_scores`` from each row's
``judge_model_output`` -- the same per-dimension scores ``judge.py`` aggregates in one run: a pillar
whose output did not parse is absent there, and a skipped image has none -- so ``<run>/
input_bench_scores.{json,xlsx}`` are what a single ``judge.py`` run over the whole input writes.
Run it with the judge env's python (it imports ``judge.py`` from ``--bench``).

    python judge_parallel.py check-thinking --judged <run>/input_judged.jsonl

fails (exit 1) when the judge answered without thinking -- an empty ``<think></think>`` before its
JSON, as ms-swift 4.x does unless thinking is switched on (eval_qib/run_eval.sh --setup) -- in more
than ``--max-empty`` of its answers. The leaderboard's answers all reason facet by facet first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_jsonl(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def calls_per_id(metadata_path) -> dict:
    """ID -> judge calls (distinct level-1 pillars in its ``dims_en``)."""
    rows = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return {int(r["ID"]): len({d.split("/")[0].strip() for d in r["dims_en"].split(";") if d.strip()})
            for r in rows}


def split(rows: list, parts: int, cost: dict) -> list:
    """Greedy longest-first: each row to the part with the fewest calls so far; ID order kept."""
    loads, assign = [0] * parts, [[] for _ in range(parts)]
    for r in sorted(rows, key=lambda r: (-cost.get(int(r["ID"]), 1), int(r["ID"]))):
        k = loads.index(min(loads))
        assign[k].append(r)
        loads[k] += cost.get(int(r["ID"]), 1)
    return [sorted(a, key=lambda r: int(r["ID"])) for a in assign]


def bench_scores(judged_rows: list, judge_module) -> dict:
    """``judge.py``'s aggregation over per-row ``judge_model_output``."""
    per_row = []
    for r in judged_rows:
        out = r.get("judge_model_output")
        per_row.append(json.loads(out) if isinstance(out, str) and out else {})
    return judge_module.compute_bench_scores(per_row)


def thought(answer: str) -> bool:
    """Whether a raw judge answer reasons before its JSON (a non-empty ``<think>`` block)."""
    head, sep, _ = answer.partition("</think>")
    return bool(sep) and len(head.replace("<think>", "").strip()) > 0


def thinking_stats(judged_rows: list) -> tuple:
    """(answers, answers without thinking) over every raw ``*_judge_output`` of the rows."""
    answers = [v for r in judged_rows for k, v in r.items()
               if k.endswith("_judge_output") and isinstance(v, str) and v]
    return len(answers), sum(1 for a in answers if not thought(a))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--input", type=Path, required=True)
    s.add_argument("--parts", type=int, required=True)
    s.add_argument("--metadata", type=Path, required=True)
    m = sub.add_parser("merge")
    m.add_argument("--run", type=Path, required=True)
    m.add_argument("--parts", type=int, required=True)
    m.add_argument("--bench", type=Path, required=True)
    c = sub.add_parser("check-thinking")
    c.add_argument("--judged", type=Path, required=True)
    c.add_argument("--max-empty", type=float, default=0.05)
    a = ap.parse_args()

    if a.cmd == "check-thinking":
        if not a.judged.exists():
            print(f"[thinking] {a.judged} missing")
            return 1
        n, empty = thinking_stats(read_jsonl(a.judged))
        ok = n > 0 and empty <= a.max_empty * n
        print(f"[thinking] {n - empty}/{n} judge answers reason before scoring"
              f"{'' if ok else ' -- judged WITHOUT thinking'}")
        return 0 if ok else 1

    if a.cmd == "split":
        rows = read_jsonl(a.input)
        cost = calls_per_id(a.metadata)
        shards = a.input.parent / "shards"
        shards.mkdir(exist_ok=True)
        for k, part in enumerate(split(rows, a.parts, cost)):
            path = shards / f"part_{k}.jsonl"
            if path.exists() and read_jsonl(path) == part:
                continue                          # unchanged: keep its finished judgement
            with open(path, "w", encoding="utf-8") as f:
                for r in part:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for stale in shards.glob(f"part_{k}_*"):
                stale.unlink()
            print(f"[split] part {k}: {len(part)} images, "
                  f"{sum(cost.get(int(r['ID']), 1) for r in part)} judge calls -> {path}")
        return 0

    sys.path.insert(0, str(a.bench.resolve()))
    import judge as judge_module                   # the release's judge.py (its aggregation)
    shards = a.run / "shards"
    judged = []
    for k in range(a.parts):
        path = shards / f"part_{k}_judged.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} missing: part {k} has not been judged")
        judged += read_jsonl(path)
    judged.sort(key=lambda r: int(r["ID"]))
    expected = read_jsonl(a.run / "input.jsonl")
    if [int(r["ID"]) for r in judged] != sorted(int(r["ID"]) for r in expected):
        raise SystemExit("the judged parts do not cover input.jsonl exactly once")
    with open(a.run / "input_judged.jsonl", "w", encoding="utf-8") as f:
        for r in judged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    bench = bench_scores(judged, judge_module)
    judge_module.save_bench_scores(bench, str(a.run / "input.jsonl"))
    judge_module.print_bench_scores(bench)
    unparsed = sum(1 for r in judged if not r.get("judge_model_output"))
    print(f"[merge] {len(judged)} images from {a.parts} parts; {unparsed} without a parsed judgement "
          f"-> {a.run / 'input_bench_scores.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
