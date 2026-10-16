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


def _run(world: int, store: str, out_dir: str, micro_batch: int = 1, steps: int = 2,
         gan: bool = False, critic: str = "row", window: int = 16, stride: int = 8,
         pattern: str | None = None, critic_rows: str = "window") -> dict:
    """Launch ``world`` worker processes and return rank 0's recorded state."""
    tag = (f"_gan_{critic}_{critic_rows}" if gan else "") + f"_k{window}b{stride}" + \
        (f"_{pattern}" if pattern else "")
    out = os.path.join(out_dir, f"w{world}_mb{micro_batch}{tag}.pt")
    env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(_free_port()),
               WORLD_SIZE=str(world), OMP_NUM_THREADS="1")
    cmd = [sys.executable, WORKER, "--store", store, "--out", out,
           "--micro-batch", str(micro_batch), "--steps", str(steps),
           "--window", str(window), "--stride", str(stride)] + \
        (["--gan", "--gan-critic", critic, "--gan-critic-rows", critic_rows] if gan else []) + \
        (["--pattern", pattern] if pattern else [])
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


# --------------------------------------------------------------------------------------
# the adversarial term (rdm/sw_lmmd/adversarial.py) obeys the same contract
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def single_gan(store_root, tmp_path_factory):
    return _run(1, store_root, str(tmp_path_factory.mktemp("sw_parity_gan")), gan=True)


@pytest.mark.parametrize("world", [2, 4])
def test_adversarial_term_is_world_size_invariant(world, single_gan, store_root, tmp_path):
    """Critic step on a K/R share + mean-reduced critic grads == the critic step on all K;
    the generator side telescopes like the force; lambda comes from globally summed norms."""
    sharded = _run(world, store_root, str(tmp_path), gan=True)
    for got, want in zip(sharded["grads"], single_gan["grads"]):
        torch.testing.assert_close(got, want, rtol=1e-8, atol=1e-10)
    assert sharded["critic"].keys() == single_gan["critic"].keys()
    for key, want in single_gan["critic"].items():            # weights after two critic steps
        torch.testing.assert_close(sharded["critic"][key], want, rtol=1e-8, atol=1e-10, msg=key)
    for key in ("gan/lambda", "gan/d_loss", "gan/d_acc", "gan/grad_ratio"):
        assert sharded["gan_logs"][key] == pytest.approx(single_gan["gan_logs"][key],
                                                         rel=1e-8, abs=1e-12), key
    assert sharded["force"] == pytest.approx(single_gan["force"], rel=1e-9, abs=1e-12)


@pytest.fixture(scope="module")
def single_block_gan(store_root, tmp_path_factory):
    """K=16 / B=4: four blocks per window, so 1, 2 and 4 ranks all hold whole blocks."""
    return _run(1, store_root, str(tmp_path_factory.mktemp("sw_parity_block")), gan=True,
                critic="block", stride=4)


@pytest.mark.parametrize("world", [2, 4])
def test_block_critic_is_world_size_invariant(world, single_block_gan, store_root, tmp_path):
    """Block critic: each rank's K/R share is whole blocks, its critic grads are mean-reduced;
    the generator term scores the global block with only the rank's rows live, times R, so the
    cross-rank mean of the parameter gradients is the single-process gradient."""
    sharded = _run(world, store_root, str(tmp_path), gan=True, critic="block", stride=4)
    for got, want in zip(sharded["grads"], single_block_gan["grads"]):
        torch.testing.assert_close(got, want, rtol=1e-8, atol=1e-10)
    for key, want in single_block_gan["critic"].items():
        torch.testing.assert_close(sharded["critic"][key], want, rtol=1e-8, atol=1e-10, msg=key)
    for key in ("gan/lambda", "gan/d_loss", "gan/d_acc", "gan/grad_ratio", "gan/g_loss"):
        assert sharded["gan_logs"][key] == pytest.approx(single_block_gan["gan_logs"][key],
                                                         rel=1e-8, abs=1e-12), key


# --------------------------------------------------------------------------------------
# a mixed G/U schedule with one block critic per kind (PerKindCritic)
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def grouped_store_root(tmp_path_factory):
    """48 prompts x 4 rows with prompt_ids, so a mixed G/U schedule can be built."""
    from sw_lmmd_fixtures import CTX_DIM, D_TXT, DIMS, NAMES
    from rdm.sw_lmmd.reference_store import write_reference_store

    rng = np.random.default_rng(5)
    pid = np.repeat(np.arange(48), 4)[rng.permutation(192)]
    text = rng.standard_normal((48, D_TXT)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    return write_reference_store(
        str(tmp_path_factory.mktemp("sw_parity_grouped")), prompt_ids=pid,
        row_order=np.arange(pid.size), text_features=text,
        context=rng.standard_normal((48, CTX_DIM)).astype(np.float32),
        encoder_features={n: rng.standard_normal((pid.size, d)).astype(np.float32)
                          for n, d in zip(NAMES, DIMS)},
        bandwidths={n: {"sigma": 2.0, "beta": 1.0} for n in NAMES})


@pytest.fixture(scope="module")
def single_gu_block(grouped_store_root, tmp_path_factory):
    return _run(1, grouped_store_root, str(tmp_path_factory.mktemp("sw_parity_gu")), gan=True,
                critic="block", stride=4, pattern="GU")


@pytest.mark.parametrize("world", [2, 4])
def test_per_kind_block_critics_are_world_size_invariant(world, single_gu_block,
                                                         grouped_store_root, tmp_path):
    """Two steps (one grouped, one ungrouped): every rank steps both critics on the whole window,
    and the generator term of each step is scored by its kind's critic over the global block."""
    sharded = _run(world, grouped_store_root, str(tmp_path), gan=True, critic="block", stride=4,
                   pattern="GU")
    for got, want in zip(sharded["grads"], single_gu_block["grads"]):
        torch.testing.assert_close(got, want, rtol=1e-8, atol=1e-10)
    assert sharded["critic"].keys() == single_gu_block["critic"].keys()
    for key, want in single_gu_block["critic"].items():            # both critics' weights
        torch.testing.assert_close(sharded["critic"][key], want, rtol=1e-8, atol=1e-10, msg=key)
    for key in ("gan/lambda", "gan/grouped/d_loss", "gan/ungrouped/d_loss", "gan/g_loss"):
        assert sharded["gan_logs"][key] == pytest.approx(single_gu_block["gan_logs"][key],
                                                         rel=1e-8, abs=1e-12), key



# --------------------------------------------------------------------------------------
# gan.critic_rows: active -- every rank steps the critic on the whole fresh block
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("pattern", [None, "GU"])
def test_active_rows_block_critics_are_world_size_invariant(world, pattern, store_root,
                                                            grouped_store_root, tmp_path):
    store = grouped_store_root if pattern else store_root
    kw = dict(gan=True, critic="block", stride=4, pattern=pattern, critic_rows="active")
    single = _run(1, store, str(tmp_path), **kw)             # file names carry the world size
    sharded = _run(world, store, str(tmp_path), **kw)
    for got, want in zip(sharded["grads"], single["grads"]):
        torch.testing.assert_close(got, want, rtol=1e-8, atol=1e-10)
    for key, want in single["critic"].items():
        torch.testing.assert_close(sharded["critic"][key], want, rtol=1e-8, atol=1e-9, msg=key)
    for key in ("gan/lambda", "gan/d_loss", "gan/g_loss"):
        assert sharded["gan_logs"][key] == pytest.approx(single["gan_logs"][key],
                                                         rel=1e-8, abs=1e-12), key
