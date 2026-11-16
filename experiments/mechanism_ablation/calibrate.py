"""beta* for the global-plus-self arms: the own term's gradient share at the main run's level.

    python -m experiments.mechanism_ablation.calibrate --target step0_share_step0200.json \
        --work <dir> --out calibration.json [--gpus 2] [-- <extra run_arm.sh args>]

Procedure (fixed before any result is seen):

1. Target ``T`` = ``|g_own| / |g_total|`` of the main run at step 200 (Step 0 probe, concatenated
   over encoders, mean over its 8 windows).
2. Round 1 runs the Global arm with the extra term at the analytic ``beta_0 = (2/K)/(1-1/K)`` of J's
   weight, i.e. ``beta_0 = 1/(K-1)`` in the trainer's normalization (the own term is
   ``beta (2/B) sum_i [1 - k]`` per encoder next to the global force ``2 (repel - psi^T mu)``), for
   ``--steps`` (50) steps from the base model, on the 2000-step lr schedule. The measured share
   ``r`` is the mean of ``mech/all/share`` over those steps.
3. ``|r / T - 1| <= tol`` (0.2) accepts that beta. Otherwise the next beta solves
   ``mean_s r_s(beta) = T`` with each step's logged gradients held fixed
   (``g_total(beta) = g_global + beta h``, ``r_s(beta) = beta |h| / |g_global + beta h|``) and the
   next round re-runs 50 steps at it. At most ``--max-rounds`` (4) rounds; then it stops with an
   error and nothing is trained.

Every round (beta, the measured share, its per-step values) is kept in ``--out``; the last accepted
beta is ``beta_star``. The rounds' run directories are ``<work>/calib-r<n>``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def read_log(path: str) -> list[dict]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return [r for r in rows if "mech/all/share" in r]


def solve_beta(steps: list[dict], beta: float, target: float) -> float:
    """The beta at which the mean predicted share over ``steps`` equals ``target`` (bisection in
    log beta; the per-step gradients are taken as fixed)."""
    own2 = np.array([s["mech/all/sq_own"] for s in steps])
    tot2 = np.array([s["mech/all/sq_total"] for s in steps])
    dot = np.array([s["mech/all/dot_own_total"] for s in steps])
    a = own2 / beta ** 2                          # |h|^2
    c = np.maximum(tot2 - 2.0 * dot + own2, 0.0)  # |g_global|^2
    e = (dot - own2) / beta                       # h . g_global

    def mean_share(b: float) -> float:
        return float(np.mean(b * np.sqrt(a) / np.sqrt(np.maximum(b * b * a + 2 * b * e + c, 1e-300))))

    lo, hi = math.log(beta) - math.log(1e4), math.log(beta) + math.log(1e4)
    f_lo, f_hi = mean_share(math.exp(lo)) - target, mean_share(math.exp(hi)) - target
    if not (f_lo < 0 < f_hi):
        raise RuntimeError(f"target share {target:.4f} is outside [{f_lo + target:.4f}, "
                           f"{f_hi + target:.4f}] reachable for beta in [{math.exp(lo):.3g}, "
                           f"{math.exp(hi):.3g}]")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mean_share(math.exp(mid)) < target:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", required=True, help="Step 0 probe json of the main run at step 200")
    ap.add_argument("--work", required=True, help="directory for the calibration runs")
    ap.add_argument("--out", required=True, help="calibration.json")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--tol", type=float, default=0.2)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--window", type=int, default=128, help="K (beta_0 = 1/(K-1))")
    ap.add_argument("extra", nargs="*", help="after --: passed on to run_arm.sh")
    args = ap.parse_args()

    probe = json.load(open(args.target))
    target = float(probe["mean_over_windows"]["all"]["share"])
    K = int(args.window)
    beta = 1.0 / (K - 1)
    log = {"target": {"share": target, "from": os.path.abspath(args.target), "step": probe["step"]},
           "beta_0": beta, "beta_0_rule": "(2/K)/(1-1/K) of J's weight = 1/(K-1) in the trainer's "
                                          "normalization", "K": K, "steps": args.steps,
           "tol": args.tol, "rounds": [], "beta_star": None}
    print(f"[calibrate] target |g_own|/|g_total| = {target:.4f} (main run, step {probe['step']}); "
          f"beta_0 = 1/(K-1) = {beta:.6g}", flush=True)
    for n in range(1, args.max_rounds + 1):
        arm = f"calib-r{n}"
        run_dir = os.path.join(args.work, arm)
        if os.path.exists(run_dir):
            raise SystemExit(f"{run_dir} exists: move the old calibration away first")
        cmd = ["bash", os.path.join(HERE, "run_arm.sh"), "global_plus_self", "--gpus", str(args.gpus),
               "--set", f"exp_name={arm}", "--set", f"own_term.beta={beta!r}",
               "--set", f"steps={args.steps}",
               "--set", "lr_schedule.total_steps=2000",       # the real arm's lr over its first steps
               "--set", "save_final=false", "--set", "save_resume=false",
               "--set", f"save_freq={10 ** 9}", "--set", "ref_ablation.pre_diagnostic=false",
               *args.extra]
        t = time.time()
        print(f"[calibrate] round {n}: beta = {beta:.6g} -> {run_dir}", flush=True)
        subprocess.run(cmd, check=True, env={**os.environ, "OUTPUT_DIR": args.work})
        steps = read_log(os.path.join(run_dir, "train_log.jsonl"))
        if len(steps) != args.steps:
            raise SystemExit(f"{run_dir}: {len(steps)} logged steps, expected {args.steps}")
        shares = [s["mech/all/share"] for s in steps]
        r = float(np.mean(shares))
        ok = abs(r / target - 1.0) <= args.tol
        rec = {"round": n, "beta": beta, "measured_share": r, "relative_error": r / target - 1.0,
               "accepted": ok, "share_per_step": shares,
               "share_first10": float(np.mean(shares[:10])), "share_last10": float(np.mean(shares[-10:])),
               "minutes": (time.time() - t) / 60, "run_dir": run_dir}
        if not ok:
            try:
                rec["next_beta"] = solve_beta(steps, beta, target)
            except RuntimeError as err:          # the target is out of reach: stop, do not guess
                log["rounds"].append(rec)
                log["status"] = "failed"
                log["error"] = str(err)
                with open(args.out, "w") as f:
                    json.dump(log, f, indent=2)
                sys.exit(f"[calibrate] {err}: STOP and report (see {args.out})")
        log["rounds"].append(rec)
        print(f"[calibrate] round {n}: share {r:.4f} vs target {target:.4f} "
              f"({100 * (r / target - 1):+.1f} %) -> {'ACCEPTED' if ok else 'next beta %.6g' % rec['next_beta']}",
              flush=True)
        with open(args.out, "w") as f:
            json.dump(log, f, indent=2)
        if ok:
            log["beta_star"] = beta
            log["status"] = "accepted"
            with open(args.out, "w") as f:
                json.dump(log, f, indent=2)
            print(f"[calibrate] beta* = {beta:.6g} -> {args.out}", flush=True)
            return
        beta = rec["next_beta"]
    log["status"] = "failed"
    with open(args.out, "w") as f:
        json.dump(log, f, indent=2)
    sys.exit(f"[calibrate] no beta within {100 * args.tol:.0f} % of the target after "
             f"{args.max_rounds} rounds: STOP and report (see {args.out})")


if __name__ == "__main__":
    main()
