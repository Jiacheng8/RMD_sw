"""Per-run report of the mechanism ablation: the new arm at step 3400 plus the anchor on the same machine.

    python -m experiments.mechanism_ablation.report --run-name window-nodiag --run-dir <training run dir> \
        --run-eval <eval dir> --anchor-eval <eval dir> --work <WORK_DIR> --out <dir> \
        [--step0 '<glob>'] [--calibration calibration.json]

Writes ``scores.json``, ``report.md`` and ``run_info.json`` into ``--out`` and prints the summary lines.
Numbers only; the readings are the author's.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import subprocess
import time

CATS = [("single_object", "single"), ("two_object", "two"), ("counting", "count"),
        ("colors", "colors"), ("position", "position"), ("color_attr", "attr")]
PAPER = {  # Table 2 at 3,400 updates (6.8 H100 GPU-h), the paper's evaluation machine
    "main": {"geneval": 0.854, "pickscore": 21.90, "checkpoint": "jiachengcui888/w128_s32_2000steps_w-gan:step_0003400.pth"},
    "global": {"geneval": 0.746, "pickscore": 21.81, "checkpoint": "jiachengcui888/global-ablation-nogroup:step_0003400.pth"},
}
ANCHOR_TOL = {"geneval": 0.005, "pickscore": 0.02}


def _sh(*cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def scores_of(eval_dir: str) -> dict | None:
    p = os.path.join(eval_dir, "summary.json")
    if not os.path.exists(p):
        return None
    s = json.load(open(p))
    g = s.get("geneval") or {}
    return {**{k: g.get(k) for k, _ in CATS}, "geneval": g.get("overall"), "geneval_images": g.get("images"),
            "pickscore": s.get("pickscore"), "pickscore_ctx_len": s.get("pickscore_ctx_len"),
            "geneval_ctx_len": s.get("geneval_ctx_len"), "checkpoint": s.get("checkpoint")}


def fmt(v, spec):
    return "–" if v is None else format(v, spec)


def main() -> None:
    from .probe import table as step0_table

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-eval", required=True)
    ap.add_argument("--anchor-eval", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step0", default=None, help="glob of Step 0 jsons (machine 2)")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--repo-dir", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    run, anchor = scores_of(args.run_eval), scores_of(args.anchor_eval)
    offset = None if anchor is None else {k: anchor[k] - PAPER["main"][k] for k in ANCHOR_TOL}
    within = None if offset is None else all(abs(offset[k]) <= ANCHOR_TOL[k] for k in ANCHOR_TOL)

    # training facts: the log (step times, last step) and run_info (segments, config, commit)
    log = os.path.join(args.run_dir, "train_log.jsonl")
    rows = [json.loads(l) for l in open(log) if l.strip()] if os.path.exists(log) else []
    steps = [r for r in rows if "step" in r and "seconds" in r]
    ri_path = os.path.join(args.run_dir, "run_info.json")
    ri = json.load(open(ri_path)) if os.path.exists(ri_path) else {}
    stages = []
    sp = os.path.join(args.work, "stages.jsonl")
    if os.path.exists(sp):
        stages = [json.loads(l) for l in open(sp) if l.strip()]
    train_wall = sum(s["seconds"] for s in stages if s["stage"].startswith("train"))
    gpu_names = (_sh("nvidia-smi", "--query-gpu=name", "--format=csv,noheader") or "").splitlines()
    driver = (_sh("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader") or "").splitlines()[:1]
    last = steps[-1] if steps else {}
    train = {"steps_reached": last.get("step"), "gpus": (last["active_rows"] // last["active_rows_per_rank"])
             if last else None, "train_hours_sum_of_steps": sum(r["seconds"] for r in steps) / 3600,
             "train_hours_wall": train_wall / 3600 if train_wall else None}
    cal = json.load(open(args.calibration)) if args.calibration and os.path.exists(args.calibration) else None
    step0 = sorted(glob.glob(args.step0)) if args.step0 else []

    scores = {"run": args.run_name, "step": 3400, "scores": run, "anchor": {
        "checkpoint": PAPER["main"]["checkpoint"], "scores": anchor, "paper": PAPER["main"],
        "offset": offset, "tolerance": ANCHOR_TOL, "within_tolerance": within},
        "table2_paper": PAPER, "train": train,
        "beta_star": None if cal is None else cal.get("beta_star")}
    json.dump(scores, open(os.path.join(args.out, "scores.json"), "w"), indent=2)

    L = [f"# {args.run_name} at 3,400 updates (mechanism ablation)", "",
         f"Scored on {socket.gethostname()} ({', '.join(gpu_names) or '?'}), the same machine that trained it, with "
         "scripts/eval_checkpoint.sh --steps 1 --no-diversity: GenEval 553 prompts x 4 images, shared noise, official "
         "detector, ctx_len 48; PickScore on the 499 Pick-a-Pic prompts, 1 image each, ctx_len 232.", "",
         "| checkpoint | steps | train h (sum of steps) | train h (wall) | GPUs | single | two | count | colors | "
         "position | attr | GenEval | PickScore |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def row(name, s, steps_, th, tw, g):
        cats = " | ".join(fmt(None if (s or {}).get(k) is None else 100 * s[k], ".1f") for k, _ in CATS)
        return (f"| {name} | {steps_} | {fmt(th, '.2f')} | {fmt(tw, '.2f')} | {g} | {cats} | "
                f"**{fmt((s or {}).get('geneval'), '.4f')}** | {fmt((s or {}).get('pickscore'), '.3f')} |")
    L.append(row(f"**{args.run_name}**", run, train["steps_reached"], train["train_hours_sum_of_steps"],
                 train["train_hours_wall"], train["gpus"]))
    L.append(row("anchor: main run step 3400 (this machine)", anchor, 3400, None, None, "–"))
    L += ["", "| anchor | GenEval | PickScore |", "|---|---:|---:|",
          f"| this machine | {fmt((anchor or {}).get('geneval'), '.4f')} | {fmt((anchor or {}).get('pickscore'), '.3f')} |",
          f"| paper | {PAPER['main']['geneval']:.3f} | {PAPER['main']['pickscore']:.2f} |",
          f"| offset (this − paper) | {fmt((offset or {}).get('geneval'), '+.4f')} | "
          f"{fmt((offset or {}).get('pickscore'), '+.3f')} |", "",
          ("**Anchor within tolerance** (≤ 0.005 GenEval and ≤ 0.02 PickScore): the run's numbers can go into "
           "Table 2 as they are." if within else
           "**Anchor OUTSIDE tolerance** (> 0.005 GenEval or > 0.02 PickScore): report both; the author re-scores "
           "the new checkpoint on the paper's evaluation machine before anything goes into the paper.")
          if within is not None else "Anchor not scored.", "",
          f"Table 2 at the same budget (paper machine): Local {PAPER['main']['geneval']:.3f} / "
          f"{PAPER['main']['pickscore']:.2f}, Global {PAPER['global']['geneval']:.3f} / {PAPER['global']['pickscore']:.2f}.", ""]
    if step0:
        L += ["## Step 0: own-term share of the L_MMD feature gradient (main run)", "", step0_table(step0), "",
              "g_own = -(2/(BK)) d/du_i k(X_i, Y_i), g_total = d L_MMD / du over the active image features, each "
              "encoder weighted 1/3, 'all' = concatenated. Mean ± std over 8 consecutive windows after 3 warm-up "
              "updates, weights-only checkpoints (step 2000 = step_0002000.pth), critic off "
              "(tools/measure_cache_drift.py protocol).", ""]
    if cal:
        L += ["## beta* calibration", "",
              f"Target |g_own|/|g_total| = {cal['target']['share']:.4f} (main run, step {cal['target']['step']}); "
              f"beta_0 = 1/(K-1) = {cal['beta_0']:.6g}; tolerance {100 * cal['tol']:.0f} %; {cal['steps']}-step runs "
              "of the Global arm + beta L_self from the base model (lr of the real arm's first steps).", "",
              "| round | beta | measured share | rel. error | accepted |", "|---:|---:|---:|---:|---|"]
        for r in cal["rounds"]:
            L.append(f"| {r['round']} | {r['beta']:.6g} | {r['measured_share']:.4f} | "
                     f"{100 * r['relative_error']:+.1f} % | {'yes' if r['accepted'] else 'no'} |")
        L += ["", f"**beta\\* = {fmt(cal.get('beta_star'), '.6g')}**", ""]
    if stages:
        L += ["## Wall-clock per stage", "", "| stage | minutes |", "|---|---:|"] + \
             [f"| {s['stage']} | {s['seconds'] / 60:.1f} |" for s in stages] + [""]
    open(os.path.join(args.out, "report.md"), "w").write("\n".join(L) + "\n")

    info = {"run": args.run_name, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "commit": _sh("git", "-C", args.repo_dir, "rev-parse", "HEAD"),
            "dirty": bool(_sh("git", "-C", args.repo_dir, "status", "--porcelain", "--untracked-files=no")),
            "machine": {"host": socket.gethostname(), "gpus": gpu_names, "gpu_count": len(gpu_names),
                        "driver": driver[0] if driver else None,
                        "cuda_runtime": _sh("bash", "-c", "nvidia-smi | grep -o 'CUDA Version: [0-9.]*'")},
            "steps_reached": train["steps_reached"], "train": train, "stages": stages,
            "training_segments": ri.get("segments"), "config": ri.get("config"),
            "training_machine": ri.get("machine"), "calibration": cal,
            "seeds": {"seed": 0, "window.order_seed": 3407, "row_order_hash": "6559fefb9a872952"},
            "evaluation": {"protocol": "scripts/eval_checkpoint.sh --steps 1 --no-diversity",
                           "run": args.run_eval, "anchor": args.anchor_eval}}
    json.dump(info, open(os.path.join(args.out, "run_info.json"), "w"), indent=2, default=str)

    print("\n".join(L))
    print(f"\n==> SUMMARY {args.run_name} @ step {train['steps_reached']}: GenEval {fmt((run or {}).get('geneval'), '.4f')}, "
          f"PickScore {fmt((run or {}).get('pickscore'), '.3f')} | anchor (main @3400, this machine) GenEval "
          f"{fmt((anchor or {}).get('geneval'), '.4f')} ({fmt((offset or {}).get('geneval'), '+.4f')} vs paper 0.854), "
          f"PickScore {fmt((anchor or {}).get('pickscore'), '.3f')} ({fmt((offset or {}).get('pickscore'), '+.3f')} vs "
          f"21.90) -> {'within tolerance' if within else 'OUTSIDE tolerance: re-score on the paper machine'}")


if __name__ == "__main__":
    main()
