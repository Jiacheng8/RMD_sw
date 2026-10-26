"""Freeze completed production prompts; fail closed while captioning is incomplete."""
import argparse
import csv
import json
import os
from pathlib import Path

from common import atomic_json, atomic_save, digest, lock


def collect(build, inputs, shards):
    with (build / "manifest.csv").open(encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    known = {r["image_id"]: r for r in manifest}
    if len(known) != len(manifest):
        raise ValueError("manifest.csv contains duplicate image IDs")
    expected = set()
    for shard in shards:
        expected.update((build / "shards" / f"shard_{shard:04d}.txt").read_text().split())
    if expected - known.keys():
        raise ValueError("Shards contain IDs absent from manifest.csv")
    records, origins = {}, {}
    for path in inputs:
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                r = json.loads(line)  # Includes truncated final lines: never silently drop these.
                image_id = r["image_id"]
                if image_id not in expected:
                    continue
                has_prompt = isinstance(r.get("prompt"), str) and bool(r["prompt"].strip())
                skipped = bool(r.get("skipped"))
                if has_prompt == skipped:
                    raise ValueError(f"Invalid terminal record at {path}:{line_no}")
                if has_prompt and r.get("lang") not in ("zh", "en"):
                    raise ValueError(f"Missing/invalid language: {image_id}")
                if image_id in records:
                    old = records[image_id]
                    if any(old.get(k) != r.get(k) for k in ("prompt", "lang", "skipped")):
                        raise ValueError(f"Conflicting records for {image_id}: "
                                         f"{origins[image_id]} and {path}:{line_no}; choose inputs explicitly")
                    continue
                records[image_id], origins[image_id] = r, f"{path}:{line_no}"
    missing = expected - records.keys()
    rows = []
    for m in manifest:  # canonical original manifest order, independent of completion order
        r = records.get(m["image_id"])
        if r is None or "skipped" in r:
            continue
        if r.get("source", m["source"]) != m["source"]:
            raise ValueError(f"Source mismatch for {m['image_id']}")
        rows.append(dict(prompt_id=len(rows), image_id=m["image_id"], source=m["source"],
                         lang=r["lang"], prompt=r["prompt"].strip(),
                         original_image=m["path"], writer=r.get("writer"), origin=origins[m["image_id"]]))
    return rows, dict(expected=len(expected), terminal=len(records), missing=len(missing),
                      missing_examples=sorted(missing)[:20],
                      skipped=sum("skipped" in r for r in records.values()), usable=len(rows))


def parse_shards(spec):
    result = set()
    for part in spec.split(","):
        bounds = [int(x) for x in part.split("-")]
        if len(bounds) == 1:
            result.add(bounds[0])
        elif len(bounds) == 2 and bounds[0] <= bounds[1]:
            result.update(range(bounds[0], bounds[1] + 1))
        else:
            raise ValueError(f"Invalid shard range: {part}")
    if not result or min(result) < 0:
        raise ValueError("Shard numbers must be nonnegative")
    return sorted(result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt-build", type=Path, default=Path(os.environ.get("PROMPT_BUILD", "/data/ironman/jiacheng/RDM/z-image/prompt_build")))
    ap.add_argument("--inputs", type=Path, nargs="+", required=True,
                    help="Explicit production JSONLs, in any order; no rejects/token logs/pilot experiments")
    ap.add_argument("--shards", default="0-259")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-incomplete", action="store_true", help="Only for an isolated pilot output")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.limit < 0:
        ap.error("--limit must be nonnegative")
    rows, report = collect(a.prompt_build, a.inputs, parse_shards(a.shards))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["missing"] and not a.allow_incomplete:
        raise SystemExit("Prompts are still incomplete. Re-run after all selected shards finish.")
    if a.limit:
        rows = rows[:a.limit]
    if not rows:
        raise SystemExit("No usable prompts")
    meta = dict(report=report, prompt_hash=digest(rows), num_prompts=len(rows),
                shards=a.shards, inputs=[str(p.resolve()) for p in a.inputs],
                pilot=bool(a.limit or a.allow_incomplete))
    if a.dry_run:
        return
    with lock(a.out / ".prepare.lock"):
        if (a.out / "prepare.json").exists():
            from common import load_prompts
            old, _ = load_prompts(a.out)
            if old != rows:
                raise SystemExit("Output already has a different prompt snapshot; use a new --out")
            print("Frozen snapshot matches; nothing changed")
            return
        atomic_save(a.out / "prompts.jsonl", lambda f: f.write(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()))
        atomic_json(a.out / "prepare.json", meta)


if __name__ == "__main__":
    main()
