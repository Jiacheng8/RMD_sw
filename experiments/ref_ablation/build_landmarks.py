"""Offline: the Nystrom bundles of the ``global`` attraction, from OUR store with OUR kernel.

    python -m experiments.ref_ablation.build_landmarks experiments/ref_ablation/configs/ref_global.yaml \
        [--encoders dinov3_l,siglip2] [--m 4096] [--iters 20] [--seed 0] [--out DIR] [--force]

Why not the official iRDM bundles (``irdm_geall_assets/bundles/nystrom_joint/*_joint_M8192.pt``):
they use m=8192 landmarks, a different reference (302,149 rows: COCO top-3 + a 53,800-image GenEval
block) and therefore different bandwidths (sigma 5.3147 / 4.2557 / 10.987 for dinov3_l / siglip2 /
aimv2_huge vs 5.3005 / 4.2553 / 10.853 here; beta = sigma in both), and they store
``alpha = K^-1 mu`` without ``K_mm^{-1/2}``. Same feature normalization and [phi | beta*tau] order,
but any mismatch disqualifies them.

Per encoder, on exactly the joint feature the trainer reads
(``store.reference_joint_features``: ``[phi | beta * tau(c)]``, the store's sigma and beta, fp16 rows
read as fp32):

1. all N rows -> k-means, ``m`` centroids, ``iters`` Lloyd iterations, random init from the data,
   seeded (:func:`rdm.compare.reference_math.kmeans_landmarks`) -> landmarks ``L``;
2. ``K_mm = k(L, L) + jitter I`` in fp64 -> ``eigh`` -> ``K_mm^{-1/2} = U diag(max(lam, jitter))^{-1/2} U^T``
   (the eigh path of :func:`rdm.compare.nystrom.nystrom_feature_map`, made symmetric, kept in fp64
   and saved as a matrix -- the original returns only a closure);
3. ``mu_Lr = (1/N) sum_n k(L, y_n)`` streamed over all N rows in fp64
   (:func:`rdm.compare.reference_math.streaming_reference_mean`);
4. ``mu_bar = K_mm^{-1/2} mu_Lr = (1/N) sum_n psi(y_n)`` and ``v = K_mm^{-1/2} mu_bar``;
5. a quality report on real rows (test C on the real store): every row of ``--report-prompts``
   random prompts; ``psi(x)^T psi(y)`` vs ``k(x, y)`` split into the diagonal, same-prompt pairs and
   different-prompt pairs; ``psi(x)^T mu_bar`` vs the exact ``(1/N) sum_n k(x, y_n)`` over all N.

Writes ``<out>/<encoder>.pt`` atomically plus ``<out>/<encoder>_report.json``. An existing bundle that
still matches the store is kept unless ``--force``, so an interrupted run resumes per encoder.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time

import numpy as np
import torch

from rdm.compare.kernels import gamma_from_sigma, squared_distance
from rdm.compare.reference_math import kmeans_landmarks, streaming_reference_mean
from rdm.sw_lmmd.launch import apply_override, resolve_reference_extension
from rdm.sw_lmmd.reference_store import ReferenceFeatureStore

from .config import ref_ablation_from_config
from .nystrom_global import BUNDLE_VERSION, bundle_path, check_against_store, store_fingerprint


def _log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def load_joint_pool(store, name: str, joint: bool = True, device="cpu",
                    chunk: int = 65536) -> torch.Tensor:
    """All N joint reference features ``(N, d)`` fp32 on ``device``, read in row chunks."""
    n = int(store.num_rows)
    d = store.dim(name) + (int(store.text.shape[1]) if joint else 0)
    pool = torch.empty((n, d), dtype=torch.float32, device=device)
    for lo in range(0, n, int(chunk)):
        ids = np.arange(lo, min(lo + int(chunk), n), dtype=np.int64)
        pool[lo:lo + ids.size] = store.reference_joint_features(
            name, ids, dtype=torch.float32, joint=joint).to(device)
    return pool


@torch.no_grad()
def nystrom_from_pool(pool: torch.Tensor, sigma: float, *, m: int = 4096, iters: int = 20,
                      seed: int = 0, jitter: float = 1e-6, mean_chunk: int = 16384) -> dict:
    """Steps 1-4 of the module docstring on a ``(N, d)`` pool: the math alone, no store."""
    device = pool.device
    n = pool.shape[0]
    if not 0 < m <= n:
        raise ValueError(f"cannot place {m} landmarks on {n} rows")
    sigma = float(sigma)
    gamma = gamma_from_sigma(sigma)
    timings = {}

    t = time.time()
    L = kmeans_landmarks(pool, m, iters=iters, seed=seed)                   # fp32 (m, d)
    _sync(device)
    timings["kmeans"] = time.time() - t

    t = time.time()
    Ld = L.double()
    eye = torch.eye(m, dtype=torch.float64, device=device)
    K_mm = torch.exp(-gamma * squared_distance(Ld, Ld))
    K_mm = 0.5 * (K_mm + K_mm.T) + jitter * eye
    evals, evecs = torch.linalg.eigh(K_mm)
    inv_sqrt = (evecs * evals.clamp_min(jitter).rsqrt()[None, :]) @ evecs.T  # symmetric K^{-1/2}
    _sync(device)
    timings["eigh"] = time.time() - t

    t = time.time()
    mu_Lr = streaming_reference_mean(Ld, (Ld * Ld).sum(1), pool, gamma, chunk=mean_chunk)
    mu_bar = inv_sqrt @ mu_Lr                                                # (1/N) sum_n psi(y_n)
    v = inv_sqrt @ mu_bar                                                    # psi(x).mu_bar = k(x,L).v
    _sync(device)
    timings["mean_embedding"] = time.time() - t

    # how far the regularized whitening is from the exact inverse square root
    whiten_err = float((inv_sqrt @ K_mm @ inv_sqrt - eye).abs().max())
    return {"version": BUNDLE_VERSION,
            "landmarks": L.cpu(), "Kmm_inv_sqrt": inv_sqrt.cpu(), "mu_Lr": mu_Lr.cpu(),
            "mu_bar": mu_bar.cpu(), "v": v.cpu(), "sigma": sigma, "gamma": gamma,
            "m": int(m), "num_rows": int(n), "jitter": float(jitter),
            "eig_min": float(evals.min()), "eig_max": float(evals.max()),
            "eig_clamped": int((evals < jitter).sum()), "whitening_max_err": whiten_err,
            "kmeans": {"iters": int(iters), "seed": int(seed), "fit_rows": int(n),
                       "init": "random data rows (rdm.compare.reference_math.kmeans_landmarks)"},
            "timings_s": timings}


def build_bundle(store, name: str, pool: torch.Tensor, *, m: int = 4096, iters: int = 20,
                 seed: int = 0, jitter: float = 1e-6, joint: bool = True,
                 mean_chunk: int = 16384) -> dict:
    """One encoder's bundle: :func:`nystrom_from_pool` on ``pool`` (:func:`load_joint_pool`) with
    the store's sigma, plus what the loader checks it against."""
    bundle = nystrom_from_pool(pool, store.sigma(name), m=m, iters=iters, seed=seed,
                               jitter=jitter, mean_chunk=mean_chunk)
    bundle.update({"encoder": name, "beta": float(store.beta(name)) if joint else 0.0,
                   "joint": bool(joint), "store": store_fingerprint(store, name, joint=joint),
                   "store_root": str(store.root)})
    return bundle


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-300))


@torch.no_grad()
def quality_report(store, name: str, bundle: dict, pool: torch.Tensor, *, prompts: int = 500,
                   seed: int = 0, chunk: int = 32768) -> dict:
    """Step 5: the Nystrom approximation on real rows, against exact kernels."""
    device = pool.device
    n = pool.shape[0]
    rng = np.random.default_rng(seed)
    pid_all = store.prompt_rows(np.arange(n))
    chosen = rng.choice(np.unique(pid_all), size=min(int(prompts), np.unique(pid_all).size),
                        replace=False)
    rows = np.flatnonzero(np.isin(pid_all, chosen))                          # every row of each
    X = pool[torch.as_tensor(rows, device=device)].double()
    gamma = float(bundle["gamma"])
    L = bundle["landmarks"].to(device=device, dtype=torch.float64)
    W = bundle["Kmm_inv_sqrt"].to(device=device, dtype=torch.float64)
    mu_bar = bundle["mu_bar"].to(device=device, dtype=torch.float64)
    v = bundle["v"].to(device=device, dtype=torch.float64)

    k_exact = torch.exp(-gamma * squared_distance(X, X))
    k_xl = torch.exp(-gamma * squared_distance(X, L))
    psi = k_xl @ W
    k_nys = psi @ psi.T
    pid = torch.as_tensor(pid_all[rows], device=device)
    same = pid[:, None] == pid[None, :]
    diag = torch.eye(rows.size, dtype=torch.bool, device=device)
    pairs = {}
    for label, mask in (("self", diag), ("same_prompt", same & ~diag), ("different_prompt", ~same)):
        e, a = k_exact[mask], k_nys[mask]
        err = (a - e).abs()
        pairs[label] = {"pairs": int(mask.sum()), "exact_mean": float(e.mean()),
                        "nystrom_mean": float(a.mean()), "abs_err_mean": float(err.mean()),
                        "abs_err_max": float(err.max()),
                        "rel_err_of_mean": abs(float(a.mean()) - float(e.mean())) /
                        max(float(e.mean()), 1e-300),
                        "corr": _corr(a, e) if mask.sum() > 1 else None}

    exact = torch.zeros(rows.size, dtype=torch.float64, device=device)
    for lo in range(0, n, int(chunk)):
        exact += torch.exp(-gamma * squared_distance(X, pool[lo:lo + int(chunk)].double())).sum(1)
    exact /= n
    nys = psi @ mu_bar
    rel = (nys - exact).abs() / exact.clamp_min(1e-300)
    mean_embedding = {"rows": int(rows.size), "exact_mean": float(exact.mean()),
                      "nystrom_mean": float(nys.mean()),
                      "rel_err_of_mean": abs(float(nys.mean()) - float(exact.mean())) /
                      float(exact.mean()),
                      "rel_err_rows_median": float(rel.median()),
                      "rel_err_rows_p90": float(rel.quantile(0.9)),
                      "rel_err_rows_max": float(rel.max()), "corr": _corr(nys, exact),
                      "v_form_max_abs_diff": float((k_xl @ v - nys).abs().max())}
    return {"encoder": name, "prompts": int(chosen.size), "rows": int(rows.size),
            "kernel_pairs": pairs, "mean_embedding": mean_embedding}


def _git_commit() -> str | None:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, text=True,
                              capture_output=True, check=True).stdout.strip()
    except Exception:
        return None


def _save(obj, path: str) -> None:
    tmp = path + ".partial"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def summarize(report: dict) -> str:
    p, me = report["kernel_pairs"], report["mean_embedding"]
    return (f"psi.psi vs k on {report['rows']} rows ({report['prompts']} prompts): "
            f"self {p['self']['nystrom_mean']:.3f} (exact 1) | same-prompt mean "
            f"{p['same_prompt']['nystrom_mean']:.3e} vs {p['same_prompt']['exact_mean']:.3e} "
            f"(|err| mean {p['same_prompt']['abs_err_mean']:.2e}) | other prompts "
            f"{p['different_prompt']['nystrom_mean']:.3e} vs {p['different_prompt']['exact_mean']:.3e}"
            f"\n    psi.mu_bar vs exact mean over all N: {me['nystrom_mean']:.4e} vs "
            f"{me['exact_mean']:.4e} (rel. err of mean {me['rel_err_of_mean']:.2%}, per row median "
            f"{me['rel_err_rows_median']:.2%} / p90 {me['rel_err_rows_p90']:.2%}, corr "
            f"{me['corr']:.4f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="an ablation config (reference_root, encoders, joint, nystrom_dir)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--encoders", default=None, help="comma list (default: the config's encoders)")
    ap.add_argument("--m", type=int, default=4096, help="landmarks per encoder")
    ap.add_argument("--iters", type=int, default=20, help="k-means iterations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=1e-6)
    ap.add_argument("--out", default=None, help="default: the config's ref_ablation.nystrom_dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--report-prompts", type=int, default=500)
    ap.add_argument("--force", action="store_true", help="rebuild bundles that already exist")
    args = ap.parse_args()

    from rdm.train.launch import load_config
    cfg = load_config(args.config)
    for override in args.set:
        apply_override(cfg, override)
    out = args.out or ref_ablation_from_config(cfg).nystrom_dir
    if not out:
        raise SystemExit("no output directory: pass --out or set ref_ablation.nystrom_dir")
    os.makedirs(out, exist_ok=True)
    names = args.encoders.split(",") if args.encoders else list(cfg.encoders)
    joint = bool(getattr(cfg, "joint", True))
    store = ReferenceFeatureStore(cfg.reference_root, names, require_context=False,
                                  extension=resolve_reference_extension(cfg))
    device = torch.device(args.device)
    _log(f"store {cfg.reference_root}: {store.num_rows} rows / {store.num_prompts} prompts | "
         f"encoders {names} | m={args.m} iters={args.iters} seed={args.seed} | out {out} | "
         f"device {device}" + (f" ({torch.cuda.get_device_name(device)})"
                               if device.type == "cuda" else ""))

    for name in names:
        path = str(bundle_path(out, name))
        if os.path.exists(path) and not args.force:
            check_against_store(torch.load(path, map_location="cpu", weights_only=False), store,
                                name, joint=joint)
            _log(f"{name}: {path} exists and matches the store -- kept (--force rebuilds)")
            continue
        t0 = time.time()
        _log(f"{name}: sigma {store.sigma(name):.6g} beta {store.beta(name):.6g} | reading "
             f"{store.num_rows} joint rows ...")
        pool = load_joint_pool(store, name, joint=joint, device=device)
        t_load = time.time() - t0
        _log(f"{name}: pool {tuple(pool.shape)} fp32 ({pool.numel() * 4 / 2 ** 30:.2f} GB) in "
             f"{t_load:.0f} s; k-means {args.m} x {args.iters} it ...")
        bundle = build_bundle(store, name, pool, m=args.m, iters=args.iters, seed=args.seed,
                              jitter=args.jitter, joint=joint)
        bundle["timings_s"]["load"] = t_load
        tm = bundle["timings_s"]
        _log(f"{name}: k-means {tm['kmeans']:.0f} s | eigh {tm['eigh']:.0f} s (eig "
             f"{bundle['eig_min']:.3e} .. {bundle['eig_max']:.3e}, {bundle['eig_clamped']} clamped, "
             f"whitening err {bundle['whitening_max_err']:.1e}) | mean embedding "
             f"{tm['mean_embedding']:.0f} s; quality report ...")
        t = time.time()
        report = quality_report(store, name, bundle, pool, prompts=args.report_prompts,
                                seed=args.seed)
        tm["report"] = time.time() - t
        tm["total"] = time.time() - t0
        bundle["report"] = report
        bundle.update({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": _git_commit(),
                       "host": socket.gethostname(),
                       "device": torch.cuda.get_device_name(device) if device.type == "cuda"
                       else "cpu"})
        _save(bundle, path)
        with open(os.path.join(out, f"{name}_report.json"), "w") as f:
            json.dump({k: bundle[k] for k in ("encoder", "m", "num_rows", "sigma", "beta", "gamma",
                                              "jitter", "eig_min", "eig_max", "eig_clamped",
                                              "whitening_max_err", "kmeans", "store",
                                              "timings_s", "report", "created", "git_commit",
                                              "host", "device")}, f, indent=2)
        _log(f"{name}: {summarize(report)}")
        _log(f"{name}: -> {path} ({os.path.getsize(path) / 2 ** 20:.0f} MB) in {tm['total']:.0f} s")
        del pool
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
