"""Check our judge setup against the leaderboard: re-judge a release model's own images, compare.

The release stores, for each of its 18 models, the 1000 images and the judge's raw answers that
the leaderboard was computed from. Re-judging some of those images with our setup (judge env,
ms-swift backend as patched by ``run_eval.sh --setup``, batch, GPUs) must reproduce the stored
verdicts closely; a judge set up differently (e.g. without thinking) drifts by points.

    python -m eval_qib.calibrate prepare --bench <bench> --model Qwen-Image --limit 100 --out <run>
    (run_eval.sh judges <run>/input.jsonl as for any run)
    <judge env>/python -m eval_qib.calibrate compare --bench <bench> --model Qwen-Image --run <run>

``prepare`` downloads the model's images of the first ``limit`` IDs into ``<run>/release_images``
and writes ``<run>/input.jsonl`` with the Chinese prompts, as the leaderboard was judged.
``compare`` scores both sides with the release's own ``judge.py`` / ``score_utils.py`` over the
same images: total and pillars, facet verdict agreement, Fail / Pass / Excel shares.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

#: |our total - the release's| on the same images above which the setup is not trusted
MAX_TOTAL_GAP = 2.5


def release_rows(bench) -> list:
    from eval_qib.bench import PROMPTS_FILE
    with open(Path(bench) / PROMPTS_FILE, encoding="utf-8") as f:
        return sorted((json.loads(line) for line in f if line.strip()), key=lambda r: int(r["ID"]))


def prepare(bench, model: str, limit: int, out: Path) -> None:
    from huggingface_hub import snapshot_download

    from eval_qib.bench import BENCH_REPO, BENCH_REVISION
    rows = release_rows(bench)
    if model not in rows[0]:
        models = [k for k, v in rows[0].items() if isinstance(v, str) and v.startswith("images/")]
        raise SystemExit(f"unknown model {model!r}; the release has: {', '.join(models)}")
    rows = rows[:limit] if limit else rows
    images = out / "release_images"
    snapshot_download(BENCH_REPO, repo_type="dataset", revision=BENCH_REVISION,
                      allow_patterns=[r[model] for r in rows], local_dir=str(images))
    with open(out / "input.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            path = images / r[model]
            if not path.is_file():
                raise SystemExit(f"{path} missing after the download")
            f.write(json.dumps({"ID": int(r["ID"]), "prompt": r["prompt_cn"],
                                "image_path": str(path)}, ensure_ascii=False) + "\n")
    (out / "generation.json").write_text(json.dumps(
        {"calibration": model, "images": "the release's own", "lang": "zh", "ids": len(rows)}, indent=2))
    print(f"[calibrate] {len(rows)} images of {model} from the release -> {out / 'input.jsonl'}")


def facets(score_json, prefix="") -> dict:
    """{'L2/L3': raw score} for the scored facets (0/1/2) of one pillar's score JSON."""
    out = {}
    for k, v in (score_json or {}).items():
        if isinstance(v, dict) and "score" in v and not isinstance(v["score"], dict):
            if v["score"] in (0, 1, 2):
                out[prefix + k] = v["score"]
        elif isinstance(v, dict):
            out.update(facets(v, f"{prefix}{k}/"))
    return out


def compare(bench, model: str, run: Path) -> int:
    sys.path.insert(0, str(Path(bench).resolve()))
    import judge                                   # the release's: its parsing and aggregation
    from score_utils import extract_json_from_response, fix_score_json

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from judge_parallel import thinking_stats

    ours_rows = [json.loads(line) for line in open(run / "input_judged.jsonl", encoding="utf-8")
                 if line.strip()]
    release = {int(r["ID"]): r for r in release_rows(bench)}
    ours, theirs = [], []
    for r in ours_rows:
        out = r.get("judge_model_output")
        ours.append(json.loads(out) if isinstance(out, str) and out else {})
        rel, row = {}, release[int(r["ID"])]
        for pillar, col in judge.DIM_OUTPUT_MAP.items():
            text = row.get(f"{col.removesuffix('_judge_output')}_response_{model}")
            parsed = extract_json_from_response(text) if isinstance(text, str) and text else None
            if parsed is not None:
                rel[pillar] = fix_score_json(parsed, pillar)
        theirs.append(rel)
    a, b = judge.compute_bench_scores(ours), judge.compute_bench_scores(theirs)

    same = total = 0
    dist = {"ours": Counter(), "release": Counter()}
    for o, t in zip(ours, theirs):
        for pillar in set(o) & set(t):
            fo, ft = facets(o[pillar]), facets(t[pillar])
            for k in set(fo) & set(ft):
                total += 1
                same += fo[k] == ft[k]
            dist["ours"].update(fo.values())
            dist["release"].update(ft.values())
    n, empty = thinking_stats(ours_rows)

    print(f"[calibrate] {model}, {len(ours_rows)} of the release's images, judged again here\n")
    print(f"    {'':22s} {'ours':>8s} {'release':>8s} {'diff':>7s}")
    for pillar in judge.DIM_OUTPUT_MAP:
        x, y = a["level1"].get(pillar), b["level1"].get(pillar)
        if x is not None and y is not None:
            print(f"    {pillar:22s} {x:8.2f} {y:8.2f} {x - y:+7.2f}")
    gap = a["total"] - b["total"]
    print(f"    {'Total':22s} {a['total']:8.2f} {b['total']:8.2f} {gap:+7.2f}\n")
    for side, c in dist.items():
        m = sum(c.values()) or 1
        print(f"    {side:8s} Fail {c[0] / m:4.0%}  Pass {c[1] / m:4.0%}  Excel {c[2] / m:4.0%}")
    print(f"    facet verdicts identical: {same}/{total} ({same / max(total, 1):.0%})")
    print(f"    answers that reason before scoring: {n - empty}/{n}\n")
    ok = abs(gap) <= MAX_TOTAL_GAP and empty <= 0.05 * n
    print(f"[calibrate] {'OK' if ok else 'MISMATCH'}: total {gap:+.2f} vs the release "
          f"(tolerance {MAX_TOTAL_GAP}){'' if empty <= 0.05 * n else '; judged without thinking'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--bench", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--bench", type=Path, required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--run", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "prepare":
        a.out.mkdir(parents=True, exist_ok=True)
        prepare(a.bench, a.model, a.limit, a.out)
        return 0
    return compare(a.bench, a.model, a.run)


if __name__ == "__main__":
    raise SystemExit(main())
