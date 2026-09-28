"""World-size parity: the property that lets one config move from a 4090 to an H200 job.

SW-LMMD must differentiate the *same* objective regardless of how the active rows are spread
over ranks and micro-batches. That is not free -- it holds only because

* the latent of each sample is derived from its **row id** (not from a rank-ordered RNG),
* active ids are partitioned **contiguously** and gathered back in rank order,
* the generated context is gathered **detached**, so gradient flows only through a rank's own
  live rows, and
* each rank normalizes by ``1/(B_local * K)`` and the parameter gradients are combined with a
  **mean**, which telescopes to the global ``2/(BK)``:

      (1/R) sum_r  2/((B/R) K) sum_{i in A_r} (.)  ==  2/(BK) sum_{i in A} (.)

Exactly, not approximately -- so the reported force scalar is invariant too, and the test
asserts that alongside the gradients.

Real processes over gloo, not a simulation: each rank is launched with the same
RANK/WORLD_SIZE/MASTER_* environment that :func:`rdm.utils.distributed.setup_distributed` reads.
"""
import os
import socket
import subprocess
import sys

import numpy as np
import pytest
import torch

from sw_lmmd_fixtures import build_toy_store

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sw_lmmd_parity_worker.py")
pytestmark = pytest.mark.skipif(not torch.distributed.is_available(),
                                reason="torch.distributed unavailable")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(world: int, store: str, out_dir: str, micro_batch: int = 1, steps: int = 2) -> dict:
    """Launch ``world`` worker processes and return rank 0's recorded state."""
    out = os.path.join(out_dir, f"w{world}_mb{micro_batch}.pt")
    env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(_free_port()),
               WORLD_SIZE=str(world), OMP_NUM_THREADS="1")
    cmd = [sys.executable, WORKER, "--store", store, "--out", out,
           "--micro-batch", str(micro_batch), "--steps", str(steps)]
    procs = [subprocess.Popen(cmd, env=dict(env, RANK=str(r)), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) for r in range(world)]
    logs = []
    for rank, proc in enumerate(procs):
        try:
            output, _ = proc.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail(f"world={world} rank={rank} timed out")
        logs.append(f"--- world={world} rank={rank} rc={proc.returncode} ---\n{output}")
        if proc.returncode != 0:
            pytest.fail("".join(logs))
    return torch.load(out, weights_only=False)


@pytest.fixture(scope="module")
def store_root(tmp_path_factory):
    return build_toy_store(str(tmp_path_factory.mktemp("sw_parity_store")), num_rows=64)


@pytest.fixture(scope="module")
def single(store_root, tmp_path_factory):
    """The reference run: one process, B=8 active rows in one micro-batch."""
    return _run(1, store_root, str(tmp_path_factory.mktemp("sw_parity_out")))


@pytest.mark.parametrize("world", [2, 4])
def test_parameter_gradient_is_world_size_invariant(world, single, store_root, tmp_path):
    sharded = _run(world, store_root, str(tmp_path))
    assert len(sharded["grads"]) == len(single["grads"])
    for got, want in zip(sharded["grads"], single["grads"]):
        torch.testing.assert_close(got, want, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("world", [2, 4])
def test_force_scalar_is_world_size_invariant(world, single, store_root, tmp_path):
    """The mean over ranks of the per-rank 1/(B_local K) force is the global 1/(BK) force."""
    sharded = _run(world, store_root, str(tmp_path))
    assert sharded["force"] == pytest.approx(single["force"], rel=1e-9, abs=1e-12)
    assert sharded["raw_mmd2"] == pytest.approx(single["raw_mmd2"], rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("world", [2, 4])
def test_cache_is_replicated_and_row_aligned(world, single, store_root, tmp_path):
    """Every rank must hold the identical window, in the same global row order."""
    sharded = _run(world, store_root, str(tmp_path))
    for name, rows in single["cache_rows"].items():
        assert np.array_equal(sharded["cache_rows"][name], rows)
        torch.testing.assert_close(sharded["cache"][name], single["cache"][name],
                                   rtol=1e-9, atol=1e-11)


def test_micro_batching_is_orthogonal_to_sharding(single, store_root, tmp_path):
    """4 ranks x 2 rows and 1 rank x 8 rows in micro-batches of 2 must agree."""
    accumulated = _run(1, store_root, str(tmp_path), micro_batch=2)
    sharded = _run(4, store_root, str(tmp_path), micro_batch=2)
    for got, want in zip(accumulated["grads"], single["grads"]):
        torch.testing.assert_close(got, want, rtol=1e-8, atol=1e-10)
    for got, want in zip(sharded["grads"], single["grads"]):
        torch.testing.assert_close(got, want, rtol=1e-8, atol=1e-10)
