#!/usr/bin/env python
"""Mirror a run to Weights & Biases from the files it writes -- training is never touched.

    python -m zimage_distill.wandb_watch --run <run dir> --kind ours --gpus 8 --qib <qib>/runs \
        --pid <launcher pid> [--total 2000] [--config <yaml>]

The launchers (``_ours_run.sh``, ``_rdm_global_run.sh``, ``_dmd2_run.sh``, via ``scripts/_wandb.sh``)
start it in the background next to the run and stop it when they exit. Every ``--interval`` seconds
it reads what is new in the run dir and logs it:

* ``train/*``   every row of ``train_log.jsonl`` (the losses, per-step MMD / GAN terms, grad norm, lr,
                cache, s/step: whatever the trainer logs), at wandb step = the training step, so the
                default x axis is the trainer's own step number (ours and RDM: the optimizer step;
                DMD2: the iteration); ``train/step`` holds it too
* ``cache/*``   ours' staleness probe and cache refresh (``*drift*``, ``refresh*``: every
                ``probe_every`` / refresh steps), in a section of their own
* ``perf/*``    s/step, elapsed hours, GPU-hours spent, progress and ETA against the planned total
                (``--total``, else ``plan.json``); the plan itself goes into the run config
* ``ckpt/step`` each ``step_*.pth`` as it appears
* ``eval/<name>/*``  Qwen-Image-Bench of each name the launcher lists in ``<run>/wandb_evals.txt``:
                rendering progress, judge parts done; at the end the total, the 5 pillars and the 15
                sub-capabilities (also in the run summary), a score table and a sample of the images
                with their prompts and per-image scores

The wandb run id is kept in ``<run>/wandb.json``: a ``--resume`` or ``--eval-only`` continues the same
wandb run, and training rows already sent are not sent again. Anything that goes wrong here is
printed and skipped; the run itself does not depend on it. Exits (after a last pass) when
``<run>/.wandb_stop`` appears (the launcher's exit), when --pid is gone, or on SIGTERM.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from collections import deque
from pathlib import Path

PROJECT = "zimage-distill"
ENTITY = "sw-lmmd"                       # the project's W&B team
SAMPLE_IMAGES = 24
FACET = {0: 0.0, 1: 60.0, 2: 100.0}          # the bench's Fail / Pass / Excel


def step_of(row: dict):
    for k in ("step", "iteration"):
        if isinstance(row.get(k), (int, float)):
            return int(row[k])
    return None


def flat_numbers(row: dict, prefix: str) -> dict:
    """The numeric (and boolean) fields of a log row, under ``prefix``."""
    out = {}
    for k, v in row.items():
        if isinstance(v, bool):
            out[prefix + k] = int(v)
        elif isinstance(v, (int, float)) and math.isfinite(v):
            out[prefix + k] = v
    return out


def section(key: str) -> str:
    """W&B section of a trainer log key: the staleness probe / refresh go to ``cache/``."""
    leaf = key.rsplit("/", 1)[-1]
    return "cache/" if "drift" in leaf or leaf.startswith("refresh") else "train/"


def world_of(row: dict, gpus: int | None):
    if gpus:
        return gpus
    for a, b in (("active_rows", "active_rows_per_rank"), ("rows", "rows_per_rank")):
        if row.get(b):
            return max(1, round(row[a] / row[b]))
    return None


def image_score(judged_row: dict):
    """One image's mean facet score (0-100) over its non-N/A facets, from the judge's raw output."""
    try:
        out = json.loads(judged_row.get("judge_model_output") or "")
    except (TypeError, ValueError):
        return None
    vals = []

    def walk(x):
        if isinstance(x, dict):
            if "score" in x and not isinstance(x["score"], dict):
                if x["score"] in FACET:
                    vals.append(FACET[x["score"]])
                return
            for v in x.values():
                walk(v)
    walk(out)
    return sum(vals) / len(vals) if vals else None


class Watcher:
    def __init__(self, a):
        import wandb
        self.wandb, self.a = wandb, a
        self.run_dir, self.qib = Path(a.run), Path(a.qib) if a.qib else None
        self.state_path = self.run_dir / "wandb.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.state.setdefault("last_step", -1)
        self.state.setdefault("evals_done", [])
        self.state.setdefault("ckpts", [])
        self.state.setdefault("seconds", 0.0)
        self.state.setdefault("t0", time.time())    # eval/minutes counts from the first start
        config = {"kind": a.kind, "run_dir": str(self.run_dir), "gpus": a.gpus}
        if a.config and Path(a.config).exists():
            import yaml
            config["config"] = yaml.safe_load(Path(a.config).read_text())
        tags = [t for t in os.environ.get("WANDB_TAGS", "").split(",") if t] + [a.kind]
        self.run = wandb.init(project=os.environ.get("WANDB_PROJECT", PROJECT),
                              entity=os.environ.get("WANDB_ENTITY") or ENTITY,
                              id=self.state.get("id"), resume="allow",
                              name=self.run_dir.name, group=a.kind, tags=tags, config=config,
                              dir=str(self.run_dir))
        self.state["id"] = self.run.id
        self.state["url"] = self.run.url
        self._save()
        print(f"[wandb] {self.run.url or 'offline: ' + str(self.run_dir / 'wandb')}", flush=True)
        self.run.define_metric("train/step")
        for p in ("train/*", "cache/*", "perf/*", "ckpt/*"):
            self.run.define_metric(p, step_metric="train/step")
        self.run.define_metric("eval/minutes")
        self.run.define_metric("eval/*", step_metric="eval/minutes")
        self.offset, self.plan, self.t0 = 0, None, self.state["t0"]
        self.recent = deque(maxlen=20)           # s/step of the last rows, for the ETA

    def _save(self):
        tmp = self.state_path.with_suffix(".partial")
        tmp.write_text(json.dumps(self.state, indent=2))
        os.replace(tmp, self.state_path)

    # -------------------------------------------------------------------------- training
    def total(self):
        if self.a.total:
            return self.a.total
        plan = self.plan or {}
        return plan.get("steps") or plan.get("iterations")

    def sync_plan(self):
        p = self.run_dir / "plan.json"
        if p.exists():
            plan = json.loads(p.read_text())
            if plan != self.plan:
                self.plan = plan
                self.run.config.update({"plan": plan}, allow_val_change=True)

    def sync_train(self):
        log = self.run_dir / "train_log.jsonl"
        if not log.exists():
            return
        with open(log) as f:
            f.seek(self.offset)
            lines = f.readlines()
        if lines and not lines[-1].endswith("\n"):     # a row being written: next time
            lines = lines[:-1]
        self.offset += sum(len(line.encode()) for line in lines)
        total = self.total()
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            step = step_of(row)
            if step is None or step <= self.state["last_step"]:
                continue
            m = {section(k) + k: v for k, v in flat_numbers(row, "").items()}
            m["train/step"] = step
            sec = row.get("seconds")
            if isinstance(sec, (int, float)):
                self.state["seconds"] += sec
                self.recent.append(sec)
                world = world_of(row, self.a.gpus)
                m["perf/s_per_step"] = sec
                m["perf/train_hours"] = self.state["seconds"] / 3600
                if world:
                    m["perf/gpu_hours"] = self.state["seconds"] * world / 3600
                if total:
                    m["perf/progress"] = step / total
                    m["perf/eta_hours"] = max(0, total - step) * (sum(self.recent) / len(self.recent)) / 3600
            # wandb's step = the trainer's, unless the run's counter is already past it (a
            # restart after evaluations were logged): then the next one, so the row is kept
            if step >= self.run.step:
                self.run.log(m, step=step)
            else:
                self.run.log(m)
            self.state["last_step"] = step
        if lines:
            self._save()

    def sync_ckpts(self):
        for p in sorted(self.run_dir.glob("step_*.pth")):
            if p.name in self.state["ckpts"]:
                continue
            step = int(p.stem.split("_")[1])
            self.run.log({"ckpt/step": step}, commit=False)          # onto the current row
            self.state["ckpts"].append(p.name)
            self._save()

    # -------------------------------------------------------------------------- evaluation
    def eval_names(self):
        p = self.run_dir / "wandb_evals.txt"
        if not p.exists():
            return []
        out = []
        for line in p.read_text().splitlines():
            parts = line.split()
            if parts:
                out.append((parts[0], int(parts[1]) if len(parts) > 1 and parts[1] != "0" else 1000))
        return out

    def sync_evals(self):
        if self.qib is None:
            return
        minutes = (time.time() - self.t0) / 60
        for name, n in self.eval_names():
            if name in self.state["evals_done"]:
                continue
            d = self.qib / name
            if not d.exists():
                continue
            scores = d / "input_bench_scores.json"
            if scores.exists():
                self.log_scores(name, d, json.loads(scores.read_text()), minutes)
                self.state["evals_done"].append(name)
                self._save()
                continue
            rendered = len(list((d / "images").glob("[0-9]*.png"))) if (d / "images").exists() else 0
            m = {"eval/minutes": minutes, f"eval/{name}/rendered": min(1.0, rendered / n)}
            parts = sorted((d / "shards").glob("part_[0-9]*.jsonl")) if (d / "shards").exists() else []
            parts = [p for p in parts if not p.stem.endswith("_judged")]
            if parts:
                m[f"eval/{name}/judged_parts"] = \
                    sum((p.parent / f"{p.stem}_judged.jsonl").exists() for p in parts) / len(parts)
            self.run.log(m)

    def log_scores(self, name, d, s, minutes):
        wandb = self.wandb
        m = {"eval/minutes": minutes, f"eval/{name}/total": s["total"]}
        rows = [["total", "", s["total"]]]
        for pillar, v in s["level1"].items():
            m[f"eval/{name}/L1/{pillar}"] = v
            rows.append(["L1", pillar, v])
        for pillar, subs in s["level2"].items():
            for sub, v in (subs or {}).items():
                if v is not None:
                    m[f"eval/{name}/L2/{pillar}/{sub}"] = v
                    rows.append(["L2", f"{pillar} / {sub}", v])
        m[f"eval/{name}/scores"] = wandb.Table(columns=["level", "name", "score"], data=rows)
        judged = d / "input_judged.jsonl"
        if judged.exists():
            items = [json.loads(line) for line in open(judged, encoding="utf-8") if line.strip()]
            items.sort(key=lambda r: r.get("ID", 0))
            m[f"eval/{name}/unparsed"] = sum(1 for r in items if not r.get("judge_model_output"))
            table = wandb.Table(columns=["ID", "image", "prompt", "score"])
            for r in items[:SAMPLE_IMAGES]:
                img = d / "images" / f"{int(r['ID']):04d}.png"
                if img.exists():
                    table.add_data(r["ID"], wandb.Image(str(img)), r.get("prompt", ""), image_score(r))
            m[f"eval/{name}/samples"] = table
        self.run.log(m)
        for k, v in m.items():
            if isinstance(v, (int, float)) and k != "eval/minutes":
                self.run.summary[k.replace("eval/", "qib/", 1)] = v
        print(f"[wandb] {name}: Qwen-Image-Bench total {s['total']:.2f}", flush=True)

    # -------------------------------------------------------------------------- loop
    def sync(self):
        for f in (self.sync_plan, self.sync_train, self.sync_ckpts, self.sync_evals):
            try:
                f()
            except Exception as e:                 # never let a bad read stop the watcher
                print(f"[wandb] {f.__name__}: {type(e).__name__}: {e}", flush=True)

    def finish(self):
        self.sync()
        self.run.summary["train/last_step"] = self.state["last_step"]
        self.run.finish()


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", required=True, help="the run dir (train_log.jsonl, step_*.pth, ...)")
    ap.add_argument("--kind", required=True, help="ours | rdm-global | dmd2 (the wandb group)")
    ap.add_argument("--qib", default=None, help="<qib root>/runs: where the evaluations go")
    ap.add_argument("--gpus", type=int, default=None, help="GPUs of the training (GPU-hours)")
    ap.add_argument("--total", type=int, default=None, help="planned steps (else plan.json)")
    ap.add_argument("--config", default=None, help="the training config, stored in the run config")
    ap.add_argument("--pid", type=int, default=None, help="stop once this process has exited")
    ap.add_argument("--interval", type=float, default=30)
    a = ap.parse_args()
    Path(a.run).mkdir(parents=True, exist_ok=True)
    stop_file = Path(a.run) / ".wandb_stop"
    stop_file.unlink(missing_ok=True)
    try:
        w = Watcher(a)
    except Exception as e:
        print(f"[wandb] not logging: {type(e).__name__}: {e}", flush=True)
        return 0
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    def done():
        return stop or stop_file.exists() or (a.pid is not None and not alive(a.pid))
    while not done():
        w.sync()
        for _ in range(int(a.interval * 10)):
            if done():
                break
            time.sleep(0.1)
    w.finish()
    stop_file.unlink(missing_ok=True)
    print("[wandb] finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
