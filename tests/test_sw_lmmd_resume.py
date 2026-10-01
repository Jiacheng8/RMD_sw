"""Resume: a run stopped at step k and continued from its resume.pth ends bit-identical to the
uninterrupted run -- same weights, optimizer moments, cache and window position. CPU, float64
toy fixtures, through the real training loop (rdm.sw_lmmd.launch.train) and real files.
"""
import json
import os
from types import SimpleNamespace

import pytest
import torch

from rdm.sw_lmmd.launch import train
from sw_lmmd_fixtures import NAMES, build_toy_store, build_trainer

LR = 1e-2          # large enough that every step visibly moves the weights


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return build_toy_store(str(tmp_path_factory.mktemp("resume_store")))


def _cfg(out_dir, steps, **kw):
    # probe every 2 and refresh every 3 windows, so the continued segment crosses both
    return SimpleNamespace(steps=steps, save_freq=4, output_dir=out_dir, exp_name="run",
                           print_freq=1, probe_every=2,
                           cache={"refresh_every_windows": 3, "refresh_probe_rows": 4,
                                  "drift_threshold": 0.0},
                           save_resume=True, resume_every=2, **kw)


def _params(trainer):
    return [p.detach().clone() for p in trainer.generator.model.parameters()]


def _assert_same_run(a, b):
    assert a.step_idx == b.step_idx
    for pa, pb in zip(_params(a), _params(b)):
        assert torch.equal(pa, pb)
    assert a.schedule.state_dict() == b.schedule.state_dict()
    for name in NAMES:
        ea, eb = a.cache.entries[name], b.cache.entries[name]
        assert (ea.row_ids == eb.row_ids).all()
        assert torch.equal(ea.features, eb.features)
        assert torch.equal(ea.generated_at_steps, eb.generated_at_steps)
    sa, sb = a.optimizer.state_dict()["state"], b.optimizer.state_dict()["state"]
    for k in sa:
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(sa[k][key], sb[k][key])


def test_trainer_state_round_trip_continues_exactly(store, tmp_path):
    full = build_trainer(store, lr=LR)
    full.bootstrap()
    for _ in range(6):
        full.step()

    part = build_trainer(store, lr=LR)
    part.bootstrap()
    for _ in range(3):
        part.step()
    path = tmp_path / "state.pth"
    torch.save(part.state_dict(with_cache=True, with_optimizer=True), path)

    resumed = build_trainer(store, lr=LR)                  # fresh weights: the file must restore them
    restored = resumed.load_state_dict(torch.load(path, weights_only=False))
    assert restored == {"step": 3, "model": True, "optimizer": True, "cache": True}
    assert resumed.cache.initialized
    for _ in range(3):
        resumed.step()
    _assert_same_run(full, resumed)


def test_without_cache_resume_rebootstraps(store, tmp_path):
    part = build_trainer(store, lr=LR)
    part.bootstrap()
    for _ in range(3):
        part.step()
    resumed = build_trainer(store, lr=LR)
    restored = resumed.load_state_dict(part.state_dict(with_cache=False, with_optimizer=True))
    assert not restored["cache"] and not resumed.cache.initialized
    resumed.bootstrap()                                    # the next window, current weights
    log = resumed.step()
    assert log["step"] == 4 and log["window_step"] == 3    # it continues at the right window


def test_loop_resume_matches_uninterrupted_run(store, tmp_path):
    full = train(_cfg(str(tmp_path / "full"), 8), device="cpu", trainer=build_trainer(store, lr=LR))

    run = str(tmp_path / "part")
    train(_cfg(run, 4), device="cpu", trainer=build_trainer(store, lr=LR))   # "crashes" after step 4
    resume = os.path.join(run, "run", "resume.pth")
    assert torch.load(resume, weights_only=False, mmap=True)["step"] == 4
    resumed = train(_cfg(run, 8, resume_from=resume), device="cpu",
                    trainer=build_trainer(store, lr=LR))
    _assert_same_run(full, resumed)

    files = sorted(os.listdir(os.path.join(run, "run")))
    assert files == ["resume.pth", "step_0000004.pth", "step_0000008.pth", "train_log.jsonl"]
    assert torch.load(resume, weights_only=False, mmap=True)["step"] == 8      # kept current
    steps = [json.loads(line)["step"] for line in open(os.path.join(run, "run", "train_log.jsonl"))]
    assert steps == list(range(1, 9))                       # one log, continued, no step twice
    ckpt = torch.load(os.path.join(run, "run", "step_0000008.pth"), weights_only=False)
    assert set(ckpt) == {"model", "step", "schedule", "world_size"}         # still weights-only


def test_resume_of_a_finished_run_trains_nothing(store, tmp_path):
    run = str(tmp_path / "done")
    first = train(_cfg(run, 4), device="cpu", trainer=build_trainer(store, lr=LR))
    again = train(_cfg(run, 4, resume_from=os.path.join(run, "run", "resume.pth")), device="cpu",
                  trainer=build_trainer(store, lr=LR))
    assert again.step_idx == 4
    for pa, pb in zip(_params(first), _params(again)):
        assert torch.equal(pa, pb)


def test_resume_state_is_off_by_default(store, tmp_path):
    cfg = _cfg(str(tmp_path / "plain"), 4)
    cfg.save_resume = False
    train(cfg, device="cpu", trainer=build_trainer(store, lr=LR))
    assert sorted(os.listdir(tmp_path / "plain" / "run")) == ["step_0000004.pth", "train_log.jsonl"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes 8-bit AdamW needs CUDA")
def test_adamw8bit_state_round_trip(tmp_path):
    """The H100 config's optimizer: its quantized moments survive save -> mmap load -> resume."""
    bnb = pytest.importorskip("bitsandbytes")
    from rdm.sw_lmmd.launch import build_optimizer

    def model(seed):
        torch.manual_seed(seed)                # 512x512 > bnb's 4096-element 8-bit cutoff
        return torch.nn.Sequential(torch.nn.Linear(512, 512), torch.nn.Tanh(),
                                   torch.nn.Linear(512, 512)).cuda()

    def steps(m, opt, first, n):
        for i in range(first, first + n):
            g = torch.Generator().manual_seed(i)
            x = torch.randn(32, 512, generator=g).cuda()
            opt.zero_grad(set_to_none=True)
            (m(x) ** 2).mean().backward()
            opt.step()

    cfg, policy = SimpleNamespace(lr=1e-3), SimpleNamespace(optimizer="adamw8bit")
    full = model(0)
    opt = build_optimizer(full.parameters(), cfg, policy)
    assert isinstance(opt, bnb.optim.AdamW8bit)
    steps(full, opt, 0, 6)

    part = model(0)
    opt = build_optimizer(part.parameters(), cfg, policy)
    steps(part, opt, 0, 3)
    torch.save({"model": part.state_dict(), "optimizer": opt.state_dict()}, tmp_path / "r.pth")
    assert any(v.dtype == torch.uint8 for v in opt.state[next(part.parameters())].values()
               if torch.is_tensor(v))                               # really 8-bit moments

    state = torch.load(tmp_path / "r.pth", map_location="cpu", weights_only=False, mmap=True)
    resumed = model(1)                         # different init: everything must come from the file
    resumed.load_state_dict(state["model"])
    opt = build_optimizer(resumed.parameters(), cfg, policy)
    opt.load_state_dict(state["optimizer"])
    steps(resumed, opt, 3, 3)
    for a, b in zip(full.parameters(), resumed.parameters()):
        assert torch.equal(a, b)
