"""FSDP: the sharded path must train the same model as the unsharded one.

klein-4B's training state does not fit a 24 GB card, so the 4090 config shards it with FSDP --
and FSDP is easy to wire so that it runs *something else*. Three failures this file pins down,
none of which the unsharded tests can see:

* **Bypassed wrapper.** FSDP gathers a unit's parameters only inside that unit's ``forward``.
  ``_run_dit`` calls the blocks and embedders itself, so unless every block is called as a
  module and the rollout enters through the wrapper, the model computes on bare 1-D shards.
* **One-rank checkpoint.** The full state dict is a collective; saving on rank 0 alone hangs.
* **Mismatched reduction.** FSDP's reduce-scatter must replace the trainer's mean, not add to it.

The oracle is the unsharded single-process run. With SGD the final weights are
``init - lr * sum(grads)``, so matching them after several steps -- through bootstrap, the
drift probe, a full cache refresh and two checkpoints -- matches every gradient on the way.
Real processes over gloo on the CPU, the same RANK/WORLD_SIZE contract ``torchrun`` sets.
"""
import json
import os
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from sw_lmmd_fixtures import build_flux_toy_store, tiny_flux_adapter

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sw_lmmd_fsdp_worker.py")
CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def _flux2_importable() -> bool:
    src = os.environ.get("FLUX2_SRC")
    if src and src not in sys.path:
        sys.path.insert(0, src)
    try:
        import flux2.model  # noqa: F401
    except ImportError:
        return False
    return True


requires_flux2 = pytest.mark.skipif(
    not (torch.distributed.is_available() and _flux2_importable()),
    reason="needs torch.distributed and the flux2 package (source env.sh / set FLUX2_SRC)")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(world: int, store: str, out_dir: str, *extra: str) -> str:
    """Launch ``world`` worker ranks; return the run directory they wrote."""
    env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(_free_port()),
               WORLD_SIZE=str(world), OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, WORKER, "--store", store, "--out-dir", out_dir, *extra]
    procs = [subprocess.Popen(cmd, env=dict(env, RANK=str(r)), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) for r in range(world)]
    logs = []
    for rank, proc in enumerate(procs):
        try:
            output, _ = proc.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            for p in procs:
                p.kill()
            pytest.fail(f"world={world} rank={rank} timed out (a collective only some ranks "
                        f"entered?)")
        logs.append(f"--- world={world} rank={rank} rc={proc.returncode} ---\n{output}")
        if proc.returncode != 0:
            pytest.fail("".join(logs))
    return os.path.join(out_dir, "run")


def _log(run_dir: str) -> list:
    with open(os.path.join(run_dir, "train_log.jsonl")) as f:
        return [json.loads(line) for line in f]


def _ckpt(run_dir: str, step: int) -> dict:
    return torch.load(os.path.join(run_dir, f"step_{step:07d}.pth"), weights_only=False)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return build_flux_toy_store(str(tmp_path_factory.mktemp("sw_fsdp_store")))


@pytest.fixture(scope="module")
def init_state():
    return {k: v.clone() for k, v in tiny_flux_adapter().state_dict().items()}


@pytest.fixture(scope="module")
def unsharded(store, tmp_path_factory):
    """The oracle: one process, no FSDP."""
    return _run(1, store, str(tmp_path_factory.mktemp("sw_fsdp_plain")))


@pytest.fixture(scope="module")
def sharded(store, tmp_path_factory):
    """Two ranks, FSDP with fp32 gathers so the comparison is limited by summation order."""
    return _run(2, store, str(tmp_path_factory.mktemp("sw_fsdp_sharded")), "--fsdp")


@pytest.fixture(scope="module")
def unsharded_gan(store, tmp_path_factory):
    return _run(1, store, str(tmp_path_factory.mktemp("sw_fsdp_plain_gan")), "--gan")


@pytest.fixture(scope="module")
def sharded_gan(store, tmp_path_factory):
    return _run(2, store, str(tmp_path_factory.mktemp("sw_fsdp_sharded_gan")), "--fsdp", "--gan")


@requires_flux2
@pytest.mark.parametrize("step", [2, 3])
def test_sharded_weights_match_unsharded(unsharded, sharded, init_state, step):
    want, got = _ckpt(unsharded, step)["model"], _ckpt(sharded, step)["model"]
    assert got.keys() == want.keys()
    for name, value in want.items():
        torch.testing.assert_close(got[name], value, rtol=1e-4, atol=1e-6, msg=name)
    # ...and training actually moved every part of the model, including each sharded block and
    # the root unit (embedders / final layer): a unit FSDP never gathered would get no gradient.
    for part in ("img_in", "txt_in", "time_in", "double_blocks.0", "single_blocks.0",
                 "single_blocks.1", "final_layer"):
        delta = max(float((got[k] - init_state[k]).abs().max())
                    for k in got if f"model.{part}" in k)
        assert delta > 1e-6, f"{part} did not train under FSDP"


@requires_flux2
@pytest.mark.parametrize("key", ["force", "grad_norm", "raw_mmd2"])
def test_sharded_step_logs_match_unsharded(unsharded, sharded, key):
    want, got = _log(unsharded), _log(sharded)
    assert [r["step"] for r in got] == [r["step"] for r in want] == [1, 2, 3]
    for w, g in zip(want, got):
        assert g[key] == pytest.approx(w[key], rel=1e-4, abs=1e-9), (key, w["step"])
    # the probe and the periodic full refresh ran inside the sharded loop too
    assert all("drift_mean" in r for r in got)
    assert any("refreshed_rows" in r for r in got)


@requires_flux2
def test_checkpoints_are_whole_weights_only(unsharded, sharded, init_state):
    for run in (unsharded, sharded):
        saved = sorted(f for f in os.listdir(run) if f.endswith(".pth"))
        assert saved == ["step_0000002.pth", "step_0000003.pth"]    # save_freq 2, no duplicate
        state = _ckpt(run, 3)
        assert "optimizer" not in state and "cache" not in state
        assert {k: v.shape for k, v in state["model"].items()} == \
               {k: v.shape for k, v in init_state.items()}


@requires_flux2
def test_optimizer_holds_the_original_parameters(sharded, init_state):
    with open(os.path.join(sharded, "worker.json")) as f:
        info = json.load(f)
    assert info["flat_params_in_optimizer"] == 0
    assert info["optimizer_params"] == len(init_state)


@requires_flux2
def test_bf16_gather_keeps_fp32_master_weights(store, init_state, tmp_path):
    """The 4090 setting: bf16 all-gathers, 8-bit AdamW, fp32 master shards."""
    run = _run(2, store, str(tmp_path), "--fsdp", "--compute-dtype", "bf16", "--amp",
               "--optimizer", "adamw8bit", "--lr", "1e-2", "--steps", "2", "--save-freq", "2")
    model = _ckpt(run, 2)["model"]
    assert {k: v.shape for k, v in model.items()} == {k: v.shape for k, v in init_state.items()}
    for name, value in model.items():
        assert value.dtype == torch.float32, f"{name} left fp32 master storage"
        assert torch.isfinite(value).all(), name
    assert max(float((model[k] - init_state[k]).abs().max()) for k in model) > 1e-4
    assert all(r["force"] == r["force"] for r in _log(run))                  # no NaN force


# ---------------------------------------------------------------- memory policy and configs
def test_compute_dtype_parses_and_requires_fsdp():
    from rdm.sw_lmmd.launch import memory_from_config

    policy = memory_from_config(SimpleNamespace(memory={
        "shard": "fsdp", "grad_reduce": "none", "param_dtype": "fp32", "compute_dtype": "bf16"}))
    assert policy.param_dtype == torch.float32 and policy.compute_dtype == torch.bfloat16
    with pytest.raises(ValueError, match="compute_dtype"):
        memory_from_config(SimpleNamespace(memory={"shard": "none", "compute_dtype": "bf16"}))


def test_bf16_master_weights_warn(caplog):
    from rdm.sw_lmmd.launch import memory_from_config

    with caplog.at_level("WARNING", logger="rdm"):
        memory_from_config(SimpleNamespace(memory={"param_dtype": "bf16"}))
    assert "rounds to nothing" in caplog.text


@pytest.mark.parametrize("name,world,grad_accum,window", [
    ("sw_lmmd_train_h100_2gpu.yaml", 2, 2, (128, 32)),
    ("sw_lmmd_train_h100_2gpu_gan.yaml", 2, 2, (128, 32)),
    ("sw_lmmd_train_4x4090.yaml", 4, 32, (1024, 128)),
])
def test_training_configs_resolve(name, world, grad_accum, window):
    from rdm.sw_lmmd.launch import memory_from_config, resolve_batching, window_from_config
    from rdm.train.launch import load_config

    cfg = load_config(os.path.join(CONFIGS, name))
    policy = memory_from_config(cfg)
    assert cfg.method == "sw_lmmd" and policy.param_dtype == torch.float32
    w = window_from_config(cfg)
    assert (w.size, w.stride) == window and w.size % world == 0       # the critic shards K
    batching = resolve_batching(cfg, policy, w, world_size=world)
    assert batching["grad_accum"] == grad_accum
    assert os.path.isabs(cfg.reference_root) and os.path.isabs(cfg.output_dir)


@requires_flux2
def test_sharded_run_with_the_critic_matches_unsharded(unsharded_gan, sharded_gan):
    """The replicated critic beside the FSDP generator: same weights, same critic trajectory."""
    want, got = _ckpt(unsharded_gan, 3)["model"], _ckpt(sharded_gan, 3)["model"]
    for name, value in want.items():
        torch.testing.assert_close(got[name], value, rtol=1e-4, atol=1e-6, msg=name)
    w_log, g_log = _log(unsharded_gan), _log(sharded_gan)
    assert ["gan/lambda" in r for r in g_log] == [False, True, True]     # g_start_step=1
    for w, g in zip(w_log, g_log):
        for key in ("force", "loss_total", "gan/d_loss", "gan/d_acc"):
            assert g[key] == pytest.approx(w[key], rel=1e-4, abs=1e-8), (key, w["step"])
    assert _log(sharded_gan)[-1]["loss_total"] != _log(sharded_gan)[-1]["force"]
