"""Supervise local GPU workers, with per-worker logs and failure/interrupt cleanup."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from common import load_prompts, lock

HERE = Path(__file__).resolve().parent


def supervise(commands, root, stage, dry):
    import shlex
    for gpu, command in commands:
        print(f"CUDA_VISIBLE_DEVICES={gpu} {shlex.join(command)}", flush=True)
    if dry:
        return
    logdir = root / "logs"
    logdir.mkdir(exist_ok=True)
    processes, files = [], []
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    old = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        for i, (gpu, command) in enumerate(commands):
            f = (logdir / f"{stage}_{i}.log").open("a")
            files.append(f)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONNOUSERSITE="1",
                       TOKENIZERS_PARALLELISM="false")
            processes.append(subprocess.Popen(command, env=env, stdout=f, stderr=subprocess.STDOUT,
                                              start_new_session=True))
        while any(p.poll() is None for p in processes):
            failed = [i for i, p in enumerate(processes) if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"{stage} workers {failed} failed; see {logdir}/{stage}_*.log")
            time.sleep(1)
        if any(p.returncode for p in processes):
            raise RuntimeError(f"{stage} failed; inspect {logdir}/{stage}_*.log")
    finally:
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        for p in processes:
            try:
                p.wait(timeout=max(0.01, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        for f in files:
            f.close()
        for sig, handler in old.items():
            signal.signal(sig, handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["render", "features", "all"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--gpu-ids", default=os.environ.get("GPU_IDS", os.environ.get("CUDA_VISIBLE_DEVICES", "0")),
                    help="Physical GPU IDs/UUIDs; honors GPU_IDS then CUDA_VISIBLE_DEVICES, otherwise 0")
    ap.add_argument("--rdm-repo", default=str(HERE.parents[1]))  # this checkout
    ap.add_argument("--encoders", default="dinov3_l,siglip2,aimv2_huge")
    ap.add_argument("--feature-batch", type=int, default=32)
    ap.add_argument("--feature-workers", type=int, default=4)
    ap.add_argument("--feature-img-size", type=int, default=512)
    ap.add_argument("--dry-run", action="store_true")
    argv = sys.argv[1:]
    boundary = argv.index("--") if "--" in argv else len(argv)
    a = ap.parse_args(argv[:boundary])
    extra = argv[boundary+1:]
    for arg in extra:
        if arg.split("=", 1)[0] in ("--root", "--rank", "--world", "--dry-run"):
            ap.error("Root/rank/world/dry-run are controlled by launch.py")
    gpus = [g.strip() for g in a.gpu_ids.split(",")]
    if any(not g for g in gpus) or len(set(gpus)) != len(gpus):
        ap.error("GPU IDs must be nonempty and distinct")
    if a.stage == "features" and extra:
        ap.error("Extra render options apply only to render/all")
    load_prompts(a.root)
    def run():
        if a.stage in ("render", "all"):
            commands = [(gpu, [sys.executable, "-u", str(HERE / "render.py"), "--root", str(a.root),
                         "--rank", str(i), "--world", str(len(gpus)), *extra])
                        for i, gpu in enumerate(gpus)]
            supervise(commands, a.root, "render", a.dry_run)
        if a.stage in ("features", "all"):
            names = a.encoders.split(",")
            if any(not n for n in names) or len(set(names)) != len(names):
                ap.error("Encoder names must be nonempty and distinct")
            base = ["--root", str(a.root), "--rdm-repo", a.rdm_repo,
                    "--batch", str(a.feature_batch), "--workers", str(a.feature_workers),
                    "--img-size", str(a.feature_img_size)]
            commands = [(gpu, [sys.executable, "-u", str(HERE / "features.py"), "extract", *base,
                              "--encoders", ",".join(names[i::len(gpus)])])
                        for i, gpu in enumerate(gpus) if names[i::len(gpus)]]
            supervise(commands, a.root, "features", a.dry_run)
            supervise([(gpus[0], [sys.executable, "-u", str(HERE / "features.py"), "finalize",
                                  *base, "--encoders", a.encoders])], a.root, "finalize", a.dry_run)
    if a.dry_run:
        run()
    else:
        with lock(a.root / ".launch.lock"):
            run()


if __name__ == "__main__":
    main()
