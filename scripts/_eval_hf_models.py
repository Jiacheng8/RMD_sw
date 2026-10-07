#!/usr/bin/env python
"""Driver of scripts/eval_hf_models.sh: evaluate every step_*.pth of several Hub model repos, plus the
4-step klein-4B teacher and the released iRDM s180, one checkpoint at a time per GPU.

Each checkpoint is downloaded (one or two ahead of the GPUs, so they never wait on the network),
evaluated by scripts/eval_checkpoint.sh -- GenEval, PickScore, seed diversity -- and deleted as
soon as its evaluation is complete. A failed evaluation keeps its checkpoint and is retried once at
the end of the queue; if it fails again it is reported and the checkpoint stays for a re-run.
Everything is resumable: a checkpoint whose summary.json is complete is never downloaded again.

    <out>/evals/<run>/step_NNNNNNN/       eval_checkpoint.sh output (summary.json, geneval/, logs)
    <out>/evals/baselines/<name>/         teacher_4step, s180_release
    <out>/ckpt/                           checkpoints in flight (empty when all is done)
    <out>/logs/<run>/step_NNNNNNN.log     one log per evaluation
    <out>/models.json                     what was evaluated -- read by eval_hf_report.py
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEED_MIB = 22000                     # eval_checkpoint.sh's own floor (peak ~21.2 GB on a 4090)

# The seven runs, in the order the report lists them. label: short legend name; desc: the setting.
DEFAULT_RUNS = [
    ("jiachengcui888/sw-lmmd-flux-h100-2gpu", "K1024 · no GAN",
     "K=1024 / B=128 · no GAN · ungrouped · COCO reference (to 3000 steps)"),
    ("jiachengcui888/w128_s32_2000steps_wo-gan", "K128 · no GAN",
     "K=128 / B=32 · no GAN · ungrouped · COCO reference"),
    ("jiachengcui888/w128_s32_2000steps_w-gan", "K128 · GAN",
     "K=128 / B=32 · GAN · ungrouped · COCO reference"),
    ("jiachengcui888/w128_s32_2000steps_w-gan-grouped", "K128 · GAN · grouped",
     "K=128 / B=32 · GAN · prompt-grouped · COCO reference"),
    ("jiachengcui888/sw-lmmd-flux-h100-2gpu-gan-grouped-geneval", "K128 · GAN · grouped · GE",
     "K=128 / B=32 · GAN · prompt-grouped · COCO + GenEval reference"),
    ("jiachengcui888/sw-lmmd-flux-h100-2gpu-gan-grouped-geneval-mixGU", "K128 · GAN · GU · GE",
     "K=128 / B=32 · GAN · grouped/ungrouped alternating (GU) · COCO + GenEval reference"),
    ("jiachengcui888/sw-lmmd-flux-h100-gan-grouped-geneval-k1024b128", "K1024 · GAN · grouped · GE",
     "K=1024 / B=128 · GAN · prompt-grouped · COCO + GenEval reference"),
]
S180 = ("epfl-vita/flux2-klein-1step-rdm", "flux2_klein_1step_rdm_geallcoco_s180.pth")
BASELINES = [
    {"key": "teacher_4step", "label": "klein-4B teacher (4 steps)",
     "desc": "FLUX.2 klein-4B base, 4 sampling steps (the teacher / initialisation)"},
    {"key": "s180_release", "label": "iRDM s180 (released)",
     "desc": "released iRDM one-step student, epfl-vita/flux2-klein-1step-rdm (geALLcoco s180)"},
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


@dataclass
class Job:
    key: str                      # "<run>/step_0000200" or "baselines/teacher_4step"
    run: str                      # run key (repo name) or baseline key
    step: int | None
    repo: str | None              # None: the teacher needs no download
    filename: str | None
    size: int | None
    out: str
    ckpt: str                     # local checkpoint path, or "base" for the teacher
    steps: int = 1                # sampling steps
    attempts: int = 0
    counted: bool = False         # holds one of the in-flight download slots
    download_error: str | None = None
    ready: threading.Event = field(default_factory=threading.Event)


def summary_complete(out: str, need_geneval: bool = True) -> dict | None:
    """The summary.json of a finished evaluation (GenEval + PickScore + diversity), else None."""
    path = os.path.join(out, "summary.json")
    if not os.path.isfile(path):
        return None
    try:
        s = json.load(open(path))
    except (OSError, ValueError):
        return None
    if s.get("pickscore") is None or not s.get("seed_diversity"):
        return None
    if need_geneval and (s.get("geneval") or {}).get("overall") is None:
        return None
    return s


def gpu_free_mib() -> dict[str, int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    free = {}
    for line in out.strip().splitlines():
        idx, mib = (x.strip() for x in line.split(","))
        free[idx] = int(mib)
    return free


def list_checkpoints(api, repo: str) -> list[tuple[int, str, int | None]]:
    """``[(step, filename, size), ...]`` of the repo's top-level step_*.pth, by step."""
    out = []
    for f in api.list_repo_tree(repo, recursive=False):
        m = re.fullmatch(r"step_(\d+)\.pth", f.path)
        if m:
            out.append((int(m.group(1)), f.path, getattr(f, "size", None)))
    return sorted(out)


def remove_checkpoint(job: Job) -> None:
    """Delete the checkpoint and the download bookkeeping hf_hub_download left beside it."""
    if job.ckpt == "base":
        return
    d = os.path.dirname(job.ckpt)
    for p in [job.ckpt] + [os.path.join(d, ".cache", "huggingface", "download", job.filename + ext)
                           for ext in (".metadata", ".lock", ".incomplete")]:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


class Runner:
    def __init__(self, args, jobs: list[Job], gpus: list[str]):
        self.args, self.jobs, self.gpus = args, jobs, gpus
        self.cond = threading.Condition()
        self.inflight = 0                         # downloaded or running, not finished
        self.limit = len(gpus) + args.prefetch
        self.running = 0
        self.stop = False
        self.procs: dict[int, subprocess.Popen] = {}
        self.failed: list[Job] = []
        self.durations: list[float] = []
        self.todo = queue.Queue()
        for j in jobs:
            self.todo.put(j)
        self.total = len(jobs)
        self.finished = 0

    # ------------------------------------------------------------------ downloads
    def _download(self, job: Job) -> str | None:
        from huggingface_hub import hf_hub_download
        os.makedirs(os.path.dirname(job.ckpt), exist_ok=True)
        err = None
        for attempt in range(3):
            if self.stop:
                return "stopped"
            try:
                path = hf_hub_download(job.repo, job.filename, local_dir=os.path.dirname(job.ckpt))
                if os.path.abspath(path) != os.path.abspath(job.ckpt):
                    raise OSError(f"downloaded to {path}, expected {job.ckpt}")
                got = os.path.getsize(path)
                if job.size and got != job.size:
                    os.remove(path)
                    raise OSError(f"downloaded {got} bytes, the Hub lists {job.size}")
                return None
            except Exception as e:                      # network, disk, Hub hiccups: retry
                err = f"{type(e).__name__}: {str(e)[:300]}"
                log(f"download {job.key} failed ({err}); retry {attempt + 1}/3 in {60 * (attempt + 1)} s")
                time.sleep(60 * (attempt + 1))
        return err

    def downloader(self) -> None:
        for job in self.jobs:
            if self.stop:
                return
            if job.repo is None:                        # the teacher: nothing to fetch
                job.ready.set()
                continue
            with self.cond:
                while self.inflight >= self.limit and not self.stop:
                    self.cond.wait(10)
                if self.stop:
                    return
                self.inflight += 1
                job.counted = True
            if os.path.isfile(job.ckpt) and (not job.size or os.path.getsize(job.ckpt) == job.size):
                log(f"have      {job.key} (already downloaded)")
            else:
                t0 = time.time()
                log(f"download  {job.key} ({(job.size or 0) / 1e9:.1f} GB)")
                job.download_error = self._download(job)
                if job.download_error is None:
                    log(f"got       {job.key} in {(time.time() - t0) / 60:.1f} min")
            job.ready.set()

    def _release(self, job: Job) -> None:
        with self.cond:
            if job.counted:
                job.counted = False
                self.inflight -= 1
            self.cond.notify_all()

    # ------------------------------------------------------------------ evaluation
    def _wait_for_gpu(self, gpu: str) -> bool:
        warned = 0.0
        while not self.stop:
            free = gpu_free_mib().get(gpu)
            if free is None or free >= self.args.need_mib:
                return True
            if time.time() - warned > 600:
                log(f"GPU {gpu}: {free} MiB free < {self.args.need_mib}; waiting for it to clear")
                warned = time.time()
            time.sleep(60)
        return False

    def _evaluate(self, job: Job, gpu: str) -> int:
        cmd = ["bash", os.path.join(REPO, "scripts", "eval_checkpoint.sh"), job.ckpt,
               "--root", self.args.root, "--env", self.args.env, "--out", job.out]
        if self.args.geneval_root:
            cmd += ["--geneval-root", self.args.geneval_root]
        if job.steps != 1:
            cmd += ["--steps", str(job.steps)]
        logfile = os.path.join(self.args.out, "logs", job.key + ".log")
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu,
               "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF",
                                                         "expandable_segments:True")}
        with open(logfile, "a") as f:
            f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} attempt {job.attempts + 1} on GPU "
                    f"{gpu}: {' '.join(cmd)}\n")
            f.flush()
            proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            with self.cond:
                self.procs[proc.pid] = proc
            rc = proc.wait()
            with self.cond:
                self.procs.pop(proc.pid, None)
        return rc

    def _metrics(self, s: dict) -> str:
        g = (s.get("geneval") or {}).get("overall")
        d = s.get("seed_diversity") or {}
        parts = [f"GenEval {g:.4f}" if g is not None else "GenEval -",
                 f"PickScore {s['pickscore']:.3f}"]
        parts += [f"{k} {d[k]:.4f}" for k in ("pixel", "dreamsim", "dinov3_l") if k in d]
        return "  ".join(parts)

    def worker(self, gpu: str) -> None:
        while not self.stop:
            try:
                job = self.todo.get_nowait()
            except queue.Empty:
                with self.cond:                         # a running job may still be re-queued
                    if self.running == 0:
                        return
                time.sleep(15)
                continue
            while not job.ready.wait(10):
                if self.stop:
                    return
            if job.download_error:
                self._release(job)
                self.failed.append(job)
                log(f"FAILED    {job.key}: download -- {job.download_error}")
                continue
            if not self._wait_for_gpu(gpu):
                return
            with self.cond:
                self.running += 1
            t0 = time.time()
            log(f"start     {job.key} on GPU {gpu}")
            rc = self._evaluate(job, gpu)
            minutes = (time.time() - t0) / 60
            done = summary_complete(job.out, need_geneval=not self.args.no_geneval)
            self._release(job)
            try:
                self._finish(job, rc, done, minutes)
            finally:
                with self.cond:
                    self.running -= 1

    def _finish(self, job: Job, rc: int, done: dict | None, minutes: float) -> None:
        if self.stop:
            return
        if rc == 0 and done:
            remove_checkpoint(job)
            if self.args.drop_images:
                shutil.rmtree(os.path.join(job.out, "geneval"), ignore_errors=True)
            self.durations.append(minutes)
            self.finished += 1
            left = self.todo.qsize() + max(self.running - 1, 0)
            eta = sum(self.durations) / len(self.durations) * left / len(self.gpus) / 60
            log(f"done      {job.key} in {minutes:.0f} min: {self._metrics(done)}"
                f"  [{self.finished}/{self.total}, ~{eta:.1f} h left]")
            return
        job.attempts += 1
        if job.attempts < self.args.attempts:
            log(f"retry     {job.key} later (exit {rc} after {minutes:.0f} min; "
                f"log {os.path.join(self.args.out, 'logs', job.key + '.log')})")
            self.todo.put(job)
        else:
            self.failed.append(job)
            log(f"FAILED    {job.key} (exit {rc}); checkpoint kept for a re-run: {job.ckpt}")

    # ------------------------------------------------------------------ control
    def shutdown(self, *_):
        if self.stop:
            return
        self.stop = True
        log("interrupted: stopping the running evaluations (checkpoints in flight are kept)")
        with self.cond:
            for proc in list(self.procs.values()):
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            self.cond.notify_all()

    def run(self) -> int:
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        threads = [threading.Thread(target=self.downloader, daemon=True)]
        threads += [threading.Thread(target=self.worker, args=(g,), daemon=True) for g in self.gpus]
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads[1:]):
            time.sleep(2)
        if self.stop:
            return 130
        if self.failed:
            log(f"{len(self.failed)} evaluation(s) failed: {', '.join(j.key for j in self.failed)}")
            log("re-run the same command to retry them (finished ones are skipped)")
            return 1
        log("all evaluations finished")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", required=True, help="data root holding env.sh (models, caches)")
    ap.add_argument("--geneval-root", default="")
    ap.add_argument("--env", default="rdm")
    ap.add_argument("--gpus", default="", help="comma list; default: every GPU with enough free memory")
    ap.add_argument("--runs", default="", help="comma list of Hub repos (default: the seven runs)")
    ap.add_argument("--only-steps", default="", help="comma list of steps to evaluate (default: all)")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--no-geneval", action="store_true")
    ap.add_argument("--drop-images", action="store_true", help="delete the GenEval renders once scored")
    ap.add_argument("--prefetch", type=int, default=1, help="checkpoints downloaded ahead of the GPUs")
    ap.add_argument("--attempts", type=int, default=2, help="tries per checkpoint")
    ap.add_argument("--need-mib", type=int, default=NEED_MIB)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)

    from huggingface_hub import HfApi
    api = HfApi()
    runs = DEFAULT_RUNS
    if args.runs:
        known = {r[0]: r for r in DEFAULT_RUNS}
        runs = [known.get(r, (r, r.split("/")[-1], r)) for r in args.runs.split(",") if r]
    only = {int(s) for s in args.only_steps.split(",") if s.strip()}

    jobs: list[Job] = []
    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": [], "baselines": []}
    for repo, label, desc in runs:
        key = repo.split("/")[-1]
        ckpts = list_checkpoints(api, repo)
        meta["runs"].append({"key": key, "repo": repo, "label": label, "desc": desc,
                             "steps": [s for s, _, _ in ckpts]})
        for step, fname, size in ckpts:
            if only and step not in only:
                continue
            name = f"step_{step:07d}"
            jobs.append(Job(key=f"{key}/{name}", run=key, step=step, repo=repo, filename=fname,
                            size=size, out=os.path.join(args.out, "evals", key, name),
                            ckpt=os.path.join(args.out, "ckpt", key, fname)))
    if not args.no_baselines:
        meta["baselines"] = BASELINES
        jobs.append(Job(key="baselines/teacher_4step", run="teacher_4step", step=None, repo=None,
                        filename=None, size=None,
                        out=os.path.join(args.out, "evals", "baselines", "teacher_4step"),
                        ckpt="base", steps=4))
        s180_size = next((f.size for f in api.list_repo_tree(S180[0]) if f.path == S180[1]), None)
        jobs.append(Job(key="baselines/s180_release", run="s180_release", step=None, repo=S180[0],
                        filename=S180[1], size=s180_size,
                        out=os.path.join(args.out, "evals", "baselines", "s180_release"),
                        ckpt=os.path.join(args.out, "ckpt", "baselines", S180[1])))

    # finished ones: skip them, and drop a checkpoint a crash may have left behind
    todo = []
    for j in jobs:
        if summary_complete(j.out, need_geneval=not args.no_geneval):
            remove_checkpoint(j)
        else:
            todo.append(j)
    os.makedirs(args.out, exist_ok=True)
    meta_path = os.path.join(args.out, "models.json")
    old = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
    for kind in ("runs", "baselines"):           # keep entries of earlier invocations
        keys = {m["key"] for m in meta[kind]}
        meta[kind] += [m for m in old.get(kind, []) if m["key"] not in keys]
    if not args.dry_run:
        json.dump(meta, open(meta_path, "w"), indent=2)

    free = gpu_free_mib()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()] or \
        [g for g, mib in sorted(free.items(), key=lambda x: int(x[0])) if mib >= args.need_mib]
    gb = sum((j.size or 0) for j in todo) / 1e9
    log(f"{len(jobs)} evaluation(s): {len(jobs) - len(todo)} already done, {len(todo)} to run "
        f"({gb:.0f} GB of checkpoints to download, at most {len(gpus) + args.prefetch} on disk at once)")
    for m in meta["runs"]:
        n_todo = sum(1 for j in todo if j.run == m["key"])
        log(f"  {m['key']:<52} {len(m['steps']):>2} checkpoints, {n_todo:>2} to evaluate")
    for b in meta["baselines"]:
        log(f"  baseline {b['key']:<43} {'to evaluate' if any(j.run == b['key'] for j in todo) else 'done'}")
    log(f"GPUs: {', '.join(gpus) or 'none'}  (free MiB now: {free})")
    if args.dry_run:
        log("dry run: nothing downloaded or evaluated")
        return 0
    if not todo:
        log("nothing to do")
        return 0
    if not gpus:
        log(f"ERROR: no GPU has {args.need_mib} MiB free -- free one, or pass --gpus to wait for it")
        return 2
    return Runner(args, todo, gpus).run()


if __name__ == "__main__":
    sys.exit(main())
