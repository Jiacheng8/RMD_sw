"""rdm/baselines/dmd2_flux.py: the timestep shift and the DMD loss's gradient."""
from __future__ import annotations

import math

import pytest
import torch

from rdm.baselines.dmd2_flux import checkpoint_iterations, dmd_loss, shift_timesteps


def test_shift_matches_flux2_and_pushes_toward_noise():
    u = torch.tensor([0.02, 0.25, 0.5, 0.75, 0.98])
    t = shift_timesteps(u, 2.03)
    assert torch.allclose(t, math.exp(2.03) / (math.exp(2.03) + (1 / u - 1)))
    assert bool((t > u).all()) and bool((t[1:] > t[:-1]).all())   # monotone, toward high noise
    assert float(shift_timesteps(torch.tensor([0.5]), 2.03)) == pytest.approx(0.8839, abs=1e-4)


def test_teacher_mu_is_the_4_step_schedule_at_512():
    pytest.importorskip("flux2.sampling")
    from rdm.baselines.dmd2_flux import teacher_mu
    assert teacher_mu(512, 4) == pytest.approx(2.03, abs=0.01)     # the paper's mu ~ 2.03


def test_dmd_gradient_points_from_fake_to_real():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 3, 3, requires_grad=True)
    real, fake = torch.randn(2, 4, 3, 3), torch.randn(2, 4, 3, 3)
    dmd_loss(x, real, fake).backward()
    norm = (x - real).detach().abs().mean(dim=(1, 2, 3), keepdim=True)
    expected = (fake - real) / norm / x.numel()                     # mse's mean over elements
    assert torch.allclose(x.grad, expected, atol=1e-6)


def test_checkpoints_end_at_the_last_iteration():
    assert checkpoint_iterations(1000, 250) == {250, 500, 750, 1000}
    assert checkpoint_iterations(10, 4) == {4, 8, 10}
