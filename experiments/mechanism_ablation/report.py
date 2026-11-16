"""Reports of the mechanism ablation, in two places:

* ``train`` -- on the rented H100 box, after training (run_machine*.sh): ``report.md`` + ``run_info.json``
  (commit, full config, GPUs, driver, wall-clock per stage, steps reached; machine 2: Step 0 table and
  beta* calibration). Nothing is scored there.

      python -m experiments.mechanism_ablation.report train --run-name window-nodiag --run-dir <run dir> \
          --work <WORK_DIR> --out <dir> [--step0 '<glob>'] [--calibration calibration.json]

* ``scored`` -- on the 6 x 4090 paper evaluation box (eval_4090.sh): the new arm's step 3400 next to
  Table 2 (paper machine: Local 0.854 / 21.90, Global 0.746 / 21.81), plus an anchor when one was scored.
  ``--train-side`` is the folder the H100 box uploaded.

      python -m experiments.mechanism_ablation.report scored --run-name window-nodiag --train-side <dir> \
          --run-eval <eval dir> [--anchor-eval <eval dir>] --out <dir>

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
PAPER = {  # Table 2 at 3,400 updates (6.8 H100 GPU-h), scored on the paper's evaluation machine
    "main": {"geneval": 0.854, "pickscore": 21.90, "checkpoint": "jiachengcui888/w128_s32_2000steps_w-gan:step_0003400.pth"},
    "global": {"geneval": 0.746, "pickscore": 21.81, "checkpoint": "jiachengcui888/global-ablation-nogroup:step_0003400.pth"},
}
ANCHOR_TOL = {"geneval": 0.005, "pickscore": 0.02}
PROTOCOL = ("scripts/eval_checkpoint.sh --steps 1 --no-diversity: GenEval 553 prompts x 4 images, shared noise, "
            "official detector, ctx_len 48; PickScore on the 499 Pick-a-Pic prompts, 1 image each, ctx_len 232")
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _sh(*cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def fmt(v, spec):
    return "–" if v is None else format(v, spec)


def machine() -> dict:
    gpus = (_sh("nvidia-smi", "--query-gpu=name", "--format=csv,noheader") or "").splitlines()
    driver = (_sh("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader") or "").splitlines()[:1]
    return {"host": socket.gethostname(), "gpus": gpus, "gpu_count": len(gpus),
            "driver": driver[0] if driver else None,
            "cuda": _sh("bash", "-c", "nvidia-smi | grep -o 'CUDA Version: [0-9.]*'")}


def train_facts(train_log: str, stages: list) -> dict:
    rows = [json.loads(l) for l in open(train_log) if l.strip()] if os.path.exists(train_log) else []
    steps = [r for r in rows if "step" in r and "seconds" in r]
    last = steps[-1] if steps else {}
    wall = sum(s["seconds"] for s in stages if s["stage"].startswith("train"))
    return {"steps_reached": last.get("step"),
            "gpus": (last["active_rows"] // last["active_rows_per_rank"]) if last else None,
            "train_hours_sum_of_steps": sum(r["seconds"] for r in steps) / 3600 if steps else None,
            "train_hours_wall": wall / 3600 if wall else None}


def step0_section(paths) -> list:
    from .probe import table as step0_table
    if not paths:
        return []
    return ["## Step 0: own-term share of the L_MMD feature gradient (main run)", "", step0_table(paths), "",
            "g_own = -(2/(BK)) d/du_i k(X_i, Y_i), g_total = d L_MMD / du over the active image features, each "
            "encoder weighted 1/3, 'all' = concatenated. Mean ± std over 8 consecutive windows after 3 warm-up "
            "updates, weights-only checkpoints (step 2000 = step_0002000.pth), critic off "
            "(tools/measure_cache_drift.py protocol).", ""]


def calibration_section(cal) -> list:
    if not cal:
        return []
    L = ["## beta* calibration", "",
         f"Target |g_own|/|g_total| = {cal['target']['share']:.4f} (main run, step {cal['target']['step']}); "
         f"beta_0 = 1/(K-1) = {cal['beta_0']:.6g}; tolerance {100 * cal['tol']:.0f} %; {cal['steps']}-step runs of the "
         "Global arm + beta L_self from the base model (lr of the real arm's first steps).", "",
         "| round | beta | measured share | rel. error | accepted |", "|---:|---:|---:|---:|---|"]
    for r in cal["rounds"]:
        L.append(f"| {r['round']} | {r['beta']:.6g} | {r['measured_share']:.4f} | "
                 f"{100 * r['relative_error']:+.1f} % | {'yes' if r['accepted'] else 'no'} |")
    return L + ["", f"**beta\\* = {fmt(cal.get('beta_star'), '.6g')}**", ""]


def stages_section(stages) -> list:
    if not stages:
        return []
    return ["## Wall-clock per stage (training machine)", "", "| stage | minutes |", "|---|---:|"] + \
           [f"| {s['stage']} | {s['seconds'] / 60:.1f} |" for s in stages] + [""]


# ------------------------------------------------------------------------------- train (H100)
def report_train(args) -> None:
    os.makedirs(args.out, exist_ok=True)
    sp = os.path.join(args.work, "stages.jsonl")
    stages = [json.loads(l) for l in open(sp) if l.strip()] if os.path.exists(sp) else []
    train = train_facts(os.path.join(args.run_dir, "train_log.jsonl"), stages)
    ri_path = os.path.join(args.run_dir, "run_info.json")
    ri = json.load(open(ri_path)) if os.path.exists(ri_path) else {}
    cal = json.load(open(args.calibration)) if args.calibration and os.path.exists(args.calibration) else None
    step0 = sorted(glob.glob(args.step0)) if args.step0 else []
    m = machine()
    L = [f"# {args.run_name}: training (mechanism ablation)", "",
         f"Trained on {m['host']} ({', '.join(m['gpus']) or '?'}, driver {m['driver']}). Not scored here: the step-3400 "
         "checkpoint is scored on the 6 x 4090 paper evaluation box (experiments/mechanism_ablation/eval_4090.sh).", "",
         "| run | steps reached | train h (sum of steps) | train h (wall) | GPUs |", "|---|---:|---:|---:|---:|",
         f"| {args.run_name} | {train['steps_reached']} | {fmt(train['train_hours_sum_of_steps'], '.2f')} | "
         f"{fmt(train['train_hours_wall'], '.2f')} | {train['gpus']} |", ""]
    L += step0_section(step0) + calibration_section(cal) + stages_section(stages)
    open(os.path.join(args.out, "report.md"), "w").write("\n".join(L) + "\n")
    info = {"run": args.run_name, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "commit": _sh("git", "-C", REPO_DIR, "rev-parse", "HEAD"),
            "dirty": bool(_sh("git", "-C", REPO_DIR, "status", "--porcelain", "--untracked-files=no")),
            "machine": m, "steps_reached": train["steps_reached"], "train": train, "stages": stages,
            "training_segments": ri.get("segments"), "config": ri.get("config"), "calibration": cal,
            "seeds": {"seed": 0, "window.order_seed": 3407, "row_order_hash": "6559fefb9a872952"}}
    json.dump(info, open(os.path.join(args.out, "run_info.json"), "w"), indent=2, default=str)
    print("\n".join(L))
    print(f"\n==> SUMMARY {args.run_name}: trained to step {train['steps_reached']} on {train['gpus']} x "
          f"{(m['gpus'] or ['?'])[0]} in {fmt(train['train_hours_wall'], '.2f')} h (wall); not scored here")


# ------------------------------------------------------------------------------- scored (4090)
def scores_of(eval_dir):
    p = os.path.join(eval_dir, "summary.json") if eval_dir else ""
    if not p or not os.path.exists(p):
        return None
    s = json.load(open(p))
    g = s.get("geneval") or {}
    return {**{k: g.get(k) for k, _ in CATS}, "geneval": g.get("overall"), "geneval_images": g.get("images"),
            "pickscore": s.get("pickscore"), "pickscore_ctx_len": s.get("pickscore_ctx_len"),
            "geneval_ctx_len": s.get("geneval_ctx_len"), "checkpoint": s.get("checkpoint")}


def report_scored(args) -> None:
    os.makedirs(args.out, exist_ok=True)
    side = args.train_side
    tri = json.load(open(os.path.join(side, "run_info.json"))) if os.path.exists(os.path.join(side, "run_info.json")) else {}
    train = tri.get("train") or train_facts(os.path.join(side, "train_log.jsonl"), tri.get("stages") or [])
    run, anchor = scores_of(args.run_eval), scores_of(args.anchor_eval)
    offset = None if anchor is None else {k: anchor[k] - PAPER["main"][k] for k in ANCHOR_TOL}
    within = None if offset is None else all(abs(offset[k]) <= ANCHOR_TOL[k] for k in ANCHOR_TOL)
    cal_p = os.path.join(side, "calibration", "calibration.json")
    cal = json.load(open(cal_p)) if os.path.exists(cal_p) else None
    step0 = sorted(glob.glob(os.path.join(side, "step0", "step0_share_step*.json")))
    m = machine()
    tm = (tri.get("machine") or {})
    L = [f"# {args.run_name} at {train.get('steps_reached')} updates (mechanism ablation)", "",
         f"Trained on {tm.get('gpu_count', '?')} x {(tm.get('gpus') or ['?'])[0]} ({tm.get('host', '?')}); scored on "
         f"{m['host']} ({(m['gpus'] or ['?'])[0]}), the paper's evaluation machine, eval root {args.eval_root}, with "
         f"{PROTOCOL}.", "",
         "| run | steps | train h (wall) | GPUs | single | two | count | colors | position | attr | GenEval | PickScore |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def row(name, s, steps, th, g):
        cats = " | ".join(fmt(None if (s or {}).get(k) is None else 100 * s[k], ".1f") for k, _ in CATS)
        return (f"| {name} | {steps} | {fmt(th, '.2f')} | {g} | {cats} | **{fmt((s or {}).get('geneval'), '.4f')}** | "
                f"{fmt((s or {}).get('pickscore'), '.3f')} |")
    L.append(row(f"**{args.run_name}**", run, train.get("steps_reached"), train.get("train_hours_wall"), train.get("gpus")))
    for key, label in (("main", "Local (main run), paper"), ("global", "Global, paper")):
        L.append(f"| {label} | 3400 | – | 2 | – | – | – | – | – | – | **{PAPER[key]['geneval']:.3f}** | "
                 f"{PAPER[key]['pickscore']:.2f} |")
    if anchor is not None:
        L.append(row("anchor: main run step 3400, re-scored here", anchor, 3400, None, 2))
        L += ["", f"Anchor offset vs the paper: GenEval {offset['geneval']:+.4f}, PickScore {offset['pickscore']:+.3f} -> "
              + ("within tolerance (0.005 / 0.02)." if within else "OUTSIDE tolerance (0.005 / 0.02).")]
    L += [""] + step0_section(step0) + calibration_section(cal) + stages_section(tri.get("stages") or [])
    open(os.path.join(args.out, "report.md"), "w").write("\n".join(L) + "\n")
    json.dump({"run": args.run_name, "step": train.get("steps_reached"), "scores": run, "table2_paper": PAPER,
               "anchor": None if anchor is None else {"scores": anchor, "paper": PAPER["main"], "offset": offset,
                                                      "tolerance": ANCHOR_TOL, "within_tolerance": within},
               "train": train, "beta_star": None if cal is None else cal.get("beta_star")},
              open(os.path.join(args.out, "scores.json"), "w"), indent=2)
    info = dict(tri)
    info["evaluation"] = {"machine": m, "eval_root": args.eval_root, "eval_repo": args.eval_repo,
                          "eval_repo_commit": _sh("git", "-C", args.eval_repo, "rev-parse", "HEAD"),
                          "eval_repo_local_changes": (_sh("git", "-C", args.eval_repo, "status", "--porcelain",
                                                          "--untracked-files=no") or "").splitlines(),
                          "protocol": PROTOCOL, "run_eval": args.run_eval, "anchor_eval": args.anchor_eval,
                          "scored": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json.dump(info, open(os.path.join(args.out, "run_info.json"), "w"), indent=2, default=str)
    print("\n".join(L))
    print(f"\n==> SUMMARY {args.run_name} @ step {train.get('steps_reached')}: GenEval "
          f"{fmt((run or {}).get('geneval'), '.4f')}, PickScore {fmt((run or {}).get('pickscore'), '.3f')} "
          f"(Table 2: Local 0.854 / 21.90, Global 0.746 / 21.81)"
          + ("" if anchor is None else f" | anchor GenEval {anchor['geneval']:.4f} ({offset['geneval']:+.4f}), "
             f"PickScore {anchor['pickscore']:.3f} ({offset['pickscore']:+.3f})"))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("train")
    t.add_argument("--run-name", required=True)
    t.add_argument("--run-dir", required=True)
    t.add_argument("--work", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--step0", default=None)
    t.add_argument("--calibration", default=None)
    s = sub.add_parser("scored")
    s.add_argument("--run-name", required=True)
    s.add_argument("--train-side", required=True)
    s.add_argument("--run-eval", required=True)
    s.add_argument("--anchor-eval", default=None)
    s.add_argument("--out", required=True)
    s.add_argument("--eval-root", default="/data/thor/jiacheng/rdm-sets-eval")
    s.add_argument("--eval-repo", default="/home/jiacheng/RDM")
    args = ap.parse_args()
    (report_train if args.mode == "train" else report_scored)(args)


if __name__ == "__main__":
    main()
