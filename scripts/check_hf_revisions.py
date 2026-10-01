#!/usr/bin/env python
"""Check that the Hugging Face models a run uses are the ones the reference results were made with.

    python scripts/check_hf_revisions.py --record assets/hf_revisions.json   # reference machine
    python scripts/check_hf_revisions.py --check  assets/hf_revisions.json   # new machine, after downloads

Model repos are downloaded at their newest commit, so a repo updated between two machines' downloads
would silently change the run: a new chat template changes the Qwen3 context the student is
conditioned on, a re-uploaded encoder no longer matches the reference store's features. The HF
cache names every file by its content hash (snapshots/<commit>/<file> -> blobs/<etag>); --record
writes each repo's commit and per-file hashes, --check compares the files present on both sides.
A repo that only moved (a README commit) passes; a CHANGED file fails (exit 1). A repo that is not
downloaded at all fails too; single files absent here are only reported (some load lazily).
Uses HF_HUB_CACHE (or HF_HOME/hub): source the root's env.sh first.
"""
import argparse
import json
import os
import sys

# Everything that shapes a training or evaluation number: generator, VAE, prompt encoder, the
# three training encoders (their features are baked into the reference store), PickScore.
REPOS = [
    "black-forest-labs/FLUX.2-klein-4B",
    "black-forest-labs/FLUX.2-dev",
    "Qwen/Qwen3-4B",
    "timm/vit_large_patch16_dinov3.lvd1689m",
    "timm/vit_so400m_patch16_siglip_256.v2_webli",
    "timm/aimv2_huge_patch14_224.apple_pt",
    "yuvalkirstain/PickScore_v1",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
]


def hub_dir() -> str:
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    return os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")


def snapshot(hub: str, repo: str):
    """``(commit, {relative path: blob hash})`` of the cached ``main``, or ``(None, {})``."""
    d = os.path.join(hub, "models--" + repo.replace("/", "--"))
    ref = os.path.join(d, "refs", "main")
    if not os.path.exists(ref):
        return None, {}
    commit = open(ref).read().strip()
    snap = os.path.join(d, "snapshots", commit)
    files = {}
    for root, _, names in os.walk(snap):
        for name in names:
            path = os.path.join(root, name)
            files[os.path.relpath(path, snap)] = os.path.basename(os.path.realpath(path))
    return commit, files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", metavar="JSON")
    mode.add_argument("--check", metavar="JSON")
    args = ap.parse_args()
    hub = hub_dir()

    if args.record:
        out = {}
        for repo in REPOS:
            commit, files = snapshot(hub, repo)
            if commit is None:
                print(f"not in {hub}: {repo}", file=sys.stderr)
                return 1
            out[repo] = {"commit": commit, "files": files}
        with open(args.record, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"recorded {len(out)} repos from {hub} -> {args.record}")
        return 0

    ref = json.load(open(args.check))
    bad = 0
    print(f"    HF models vs {os.path.basename(args.check)} (cache {hub}):")
    for repo, want in ref.items():
        commit, files = snapshot(hub, repo)
        if commit is None:
            print(f"    MISSING  {repo}: not downloaded")
            bad += 1
            continue
        changed = sorted(f for f, h in want["files"].items() if f in files and files[f] != h)
        absent = sorted(f for f in want["files"] if f not in files)
        if changed:
            bad += 1
            print(f"    CHANGED  {repo}: {', '.join(changed)} (commit {want['commit'][:10]} -> {commit[:10]})")
        else:
            moved = "" if commit == want["commit"] else f", repo moved {want['commit'][:10]} -> {commit[:10]}"
            note = f", {len(absent)} file(s) not downloaded here" if absent else ""
            print(f"    same     {repo}{moved}{note}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
