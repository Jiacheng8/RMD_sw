"""Optional adversarial term for SW-LMMD: a conditional critic on the frozen-encoder features.

The force compares the window's generated and reference features under one fixed RBF
bandwidth per encoder. A critic learned on the *same* features adds a per-sample signal the
fixed kernel lacks: it keeps a usable gradient for samples far from every reference row, where
``exp(-d^2 / 2 sigma^2)`` has already flattened, and it can sharpen directions the kernel weighs
equally. Frozen features plus light trainable heads is the Projected-GAN / Vision-aided-GAN
recipe. Here it is the cheapest critic available, not the strongest one (see "What this does
not do").

Where it lives
    In GradCache's *middle* phase, as part of the loss on the cached features. It is never
    called from ``encode_fn``. Pass 1 caches the active rows' joint features, the loss is
    evaluated on them, and pass 2 pushes ``dL/dF`` through the generator. The critic therefore
    adds no rollout, no encoder forward and no change to GradCache, and pass 1 and pass 2 stay
    deterministic whatever the critic does in between.

What it sees
    The pooled image feature ``phi_e(x)`` of each encoder ``e``, one head per encoder,
    conditioned on ``tau(c)`` through a projection term when the objective is joint. Fakes are
    the K-row generated window the force has just compared (the B fresh rows plus the retained
    cache). Reals are the reference rows *of the same prompts*, so the real and fake batches
    carry identical text and the critic cannot score the caption distribution instead of the
    images. The retained rows are up to ``ceil(K/B) - 1`` steps old. That makes them a free
    replay buffer, with the same staleness the force already accepts.

Order within one step (alternating, not simultaneous)
    1. pass 1 -> cached active features;
    2. one critic step on the window (real = reference rows, fake = generated context);
    3. generator loss = force + lambda * adversarial. The adversarial term is scored by the
       *updated* critic, whose parameters are frozen for this graph;
    4. pass 2, reduce, clip, generator step.
    Gauss-Seidel order is the more stable of the two for GAN games, and since the critic is a
    function of cached features it costs nothing extra here.

Balancing against the force
    ``lambda = weight * |g_mmd| / |g_adv|``. Both gradients are taken w.r.t. the cached *image*
    columns of the active features (the ``beta * tau`` block has no path to the generator) and
    summed over every rank, so the adversarial gradient is ``weight`` times the force's
    gradient in feature space. This is the VQGAN adaptive weight, moved from the generator's
    last layer to the features GradCache already holds. The force's forward scalar is not an
    MMD and is not comparable across K (:mod:`rdm.sw_lmmd.local_mmd`), so a fixed lambda has
    no stable meaning, whereas a gradient ratio does.

The distribution contract (the same as the force's)
    * Critic step: each rank takes a contiguous ``K/R`` share of the window and means over
      it, and the critic gradients are averaged across ranks. That is the global mean over K
      at any world size. The critic is replicated, never sharded (a few million parameters),
      and initialized from a forked, fixed-seed RNG, so it starts identical on every rank.
    * Generator loss: a mean over the rank's ``B/R`` live rows, combined by the same
      cross-rank gradient mean as the force. It telescopes to the mean over B.
    * lambda comes from globally summed squared norms, so every rank applies the same value,
      equal to the single-process one.
    ``tests/test_sw_lmmd_parity.py`` checks the generator gradient and the critic weights at
    world 1/2/4.

Pitfall: real and fake must come through the same feature pipeline
    The reference features were extracted offline with fp32 encoder weights under bf16
    autocast (``scripts/_preprocess_extract.py``), whereas ``memory.battery_bf16: true`` runs
    the training battery with bf16 *weights*. On 4096 teacher renders (2026-10-03), a held-out
    linear probe separated stored from re-encoded features of the *same images* at
    53.7 / 52.5 / 56.3 % (dinov3_l / siglip2 / aimv2_huge, chance +- 0.8 %) with bf16 weights,
    and at 50.2 / 50.0 / 50.1 % with fp32 weights. A critic is a much better classifier than
    that probe, and the generator cannot remove a precision signature, so GAN runs should use
    ``battery_bf16: false`` or a store re-extracted with the training path. The launcher warns
    otherwise.

What this does not do
    The critic sees one pooled vector per image, which is exactly what the MMD sees. It can
    reweight and sharpen that space, but it cannot recover detail the pooling discarded. A
    token-level critic on an encoder's patch features (ADD-style) would see more. It would also
    need real images online (the teacher renders), intermediate tokens from the backbones and
    the critic evaluated inside ``encode_fn``, which is a different integration.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..utils.distributed import is_dist
from .config import GANConfig
from .sharding import all_reduce_mean_


# ----------------------------------------------------------------------------- losses
def critic_loss(real: torch.Tensor, fake: torch.Tensor, kind: str = "hinge") -> torch.Tensor:
    """The critic's objective on real / fake logits (it minimizes this)."""
    if kind == "hinge":
        return F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
    if kind == "ns":
        return F.softplus(-real).mean() + F.softplus(fake).mean()
    raise ValueError(f"unknown adversarial loss {kind!r}")


def adversarial_loss(fake: torch.Tensor, kind: str = "hinge") -> torch.Tensor:
    """The generator's side: raise the critic's score of its own samples."""
    if kind == "hinge":
        return -fake.mean()
    if kind == "ns":
        return F.softplus(-fake).mean()
    raise ValueError(f"unknown adversarial loss {kind!r}")


# ----------------------------------------------------------------------------- the critic
class _Standardize(nn.Module):
    """Fixed ``(x - mean) / scale`` from reference statistics (buffers, never trained).

    Raw embeddings carry a large common offset (aimv2_huge: ``|f| ~ 61`` against a spread of
    ``~31``), which leaves a freshly initialized MLP badly conditioned. ``scale`` is ONE scalar
    per encoder rather than a per-dimension std, because per-dimension whitening would inflate
    the low-variance directions, which is where precision noise lives.
    """

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("scale", scale.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.scale


class _ProjectionHead(nn.Module):
    """One encoder's critic: ``out(h) + <embed(tau), h>`` with ``h = MLP(phi)``.

    The projection term (Miyato & Koyama, 2018) is how the condition enters. The head scores
    how well an image feature fits *its* caption, not merely whether it looks like a reference.
    """

    def __init__(self, d_img: int, d_txt: int, hidden: int, depth: int, spectral_norm: bool):
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm if spectral_norm else (lambda m: m)
        layers, width = [], d_img
        for _ in range(depth):
            layers += [sn(nn.Linear(width, hidden)), nn.LeakyReLU(0.2)]
            width = hidden
        self.body = nn.Sequential(*layers)
        self.out = sn(nn.Linear(hidden, 1))
        self.embed = sn(nn.Linear(d_txt, hidden, bias=False)) if d_txt else None

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        h = self.body(x)
        logit = self.out(h).squeeze(-1)
        if self.embed is not None:
            logit = logit + (self.embed(t) * h).sum(-1)
        return logit


class FeatureCritic(nn.Module):
    """Per-encoder projection heads over standardized pooled features.

    ``forward({name: (n, d_img)}, tau (n, d_txt) or None) -> {name: (n,) logits}``.
    """

    def __init__(self, image_dims: dict, image_stats: dict, *, text_dim: int = 0,
                 text_stats=None, hidden: int = 1024, depth: int = 2,
                 spectral_norm: bool = True):
        super().__init__()
        self.names = list(image_dims)
        self.image_dims = {n: int(d) for n, d in image_dims.items()}
        self.conditional = text_dim > 0
        self.image_norm = nn.ModuleDict({n: _Standardize(*image_stats[n]) for n in self.names})
        self.text_norm = _Standardize(*text_stats) if self.conditional else None
        self.heads = nn.ModuleDict({
            n: _ProjectionHead(self.image_dims[n], text_dim, hidden, depth, spectral_norm)
            for n in self.names})

    def standardize(self, phi: dict, tau: torch.Tensor | None = None):
        if self.conditional and tau is None:
            raise ValueError("this critic is conditional: pass the rows' tau(c)")
        x = {n: self.image_norm[n](phi[n]) for n in self.names}
        return x, (self.text_norm(tau) if self.conditional else None)

    def score(self, x: dict, t: torch.Tensor | None = None) -> dict:
        return {n: self.heads[n](x[n], t) for n in self.names}

    def forward(self, phi: dict, tau: torch.Tensor | None = None) -> dict:
        return self.score(*self.standardize(phi, tau))


@torch.no_grad()
def reference_stats(store, names, n_rows: int = 8192, seed: int = 0, with_text: bool = True):
    """Mean vector and one RMS scale per encoder (and for ``tau``) from a fixed row sample.

    Fixed once at build time and saved with the critic, so a resumed run standardizes with
    the same numbers even if the sample would now come out differently.
    """
    n = min(int(n_rows), store.num_rows)
    rows = np.sort(np.random.default_rng(seed).choice(store.num_rows, size=n, replace=False))

    def stats(x: torch.Tensor):
        x = x.double()
        mean = x.mean(0)
        return mean, (x - mean).pow(2).mean().sqrt().clamp_min(1e-6)

    image = {name: stats(store.reference_image_features(name, rows)) for name in names}
    text = stats(store.text_features(rows)) if with_text else None
    return image, text


@contextlib.contextmanager
def _frozen(module: nn.Module):
    """Build a graph through ``module`` that treats its parameters as constants."""
    flags = [(p, p.requires_grad) for p in module.parameters()]
    for p, _ in flags:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)


# ----------------------------------------------------------------------------- trainer-facing
class AdversarialCritic:
    """The critic and its optimizer, plus the three things the trainer asks of them.

    :meth:`update` takes one critic step on detached window features, :meth:`generator_loss`
    scores the live active features, and :meth:`balance` sets lambda against the force.
    """

    def __init__(self, critic: FeatureCritic, cfg: GANConfig, device="cpu",
                 dtype: torch.dtype = torch.float32):
        self.cfg = cfg
        self.dtype = dtype
        self.critic = critic.to(device=device, dtype=dtype).eval()
        self.names = list(critic.names)
        self.image_dims = dict(critic.image_dims)
        self.optimizer = torch.optim.AdamW(self.critic.parameters(), lr=cfg.lr,
                                           betas=tuple(cfg.betas), weight_decay=0.0)
        self.updates = 0

    @property
    def conditional(self) -> bool:
        return self.critic.conditional

    def generator_active(self, step_idx: int) -> bool:
        """Past the critic-only warm-up (the critic trains from the first step regardless)."""
        return self.cfg.weight > 0 and step_idx >= self.cfg.g_start_step

    def _image(self, feats: dict) -> dict:
        """The image block of joint ``[phi | beta*tau]`` rows; slicing keeps the autograd edge."""
        return {n: feats[n][:, :self.image_dims[n]].to(self.dtype) for n in self.names}

    def _text(self, tau):
        return tau.to(self.dtype) if self.conditional else None

    # ---------------------------------------------------------------- critic step
    def update(self, fake: dict, real: dict, tau: torch.Tensor | None = None) -> dict:
        """One critic step on this rank's share of the window, gradients averaged over ranks.

        ``fake`` and ``real`` are the SAME rows (generated / reference), so they share ``tau``.
        Real and fake go through one forward, which keeps spectral norm at one power
        iteration per step on every rank and at every world size.
        """
        cfg = self.cfg
        fake, real = self._image(fake), self._image(real)
        n_real = next(iter(real.values())).shape[0]
        if next(iter(fake.values())).shape[0] != n_real:
            raise ValueError("update() expects the generated and reference rows of the SAME "
                             "window rows")
        t = self._text(tau)
        self.critic.train()
        x, tt = self.critic.standardize(
            {n: torch.cat([real[n], fake[n]], 0).detach() for n in self.names},
            None if t is None else torch.cat([t, t], 0))
        if cfg.r1_gamma > 0:
            x = {n: v.requires_grad_(True) for n, v in x.items()}
        logits = self.critic.score(x, tt)

        loss = sum(critic_loss(lg[:n_real], lg[n_real:], cfg.loss)
                   for lg in logits.values()) / len(self.names)
        r1 = loss.new_zeros(())
        if cfg.r1_gamma > 0:      # on the standardized input, so gamma is comparable across heads
            grads = torch.autograd.grad(sum(lg[:n_real].sum() for lg in logits.values()),
                                        [x[n] for n in self.names], create_graph=True)
            r1 = sum(g[:n_real].pow(2).sum(1).mean() for g in grads) / len(self.names)
            loss = loss + 0.5 * cfg.r1_gamma * r1
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = self._reduce_grads()
        if finite:
            self.optimizer.step()
            self.updates += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.critic.eval()

        with torch.no_grad():
            accs = [((lg[:n_real] > 0).to(loss.dtype).mean() +
                     (lg[n_real:] < 0).to(loss.dtype).mean()) / 2 for lg in logits.values()]
            vec = torch.stack([loss.detach(), r1.detach(),
                               torch.stack([lg[:n_real].mean() for lg in logits.values()]).mean(),
                               torch.stack([lg[n_real:].mean() for lg in logits.values()]).mean(),
                               *accs]).double()
        all_reduce_mean_(vec)                            # one collective for every logged value
        v = vec.tolist()
        out = {"gan/d_loss": v[0], "gan/d_real": v[2], "gan/d_fake": v[3],
               "gan/d_acc": float(np.mean(v[4:])), "gan/d_skipped": not finite}
        out.update({f"gan/{n}/d_acc": a for n, a in zip(self.names, v[4:])})
        if cfg.r1_gamma > 0:
            out["gan/r1"] = v[1]
        return out

    def _reduce_grads(self) -> bool:
        """Average the critic gradients across ranks in one flat all-reduce; True if finite.

        Every rank sees the same averaged gradient, so the replicated critics stay identical.
        A missing gradient enters as zeros, which keeps the flat layout the same on every rank.
        """
        params = list(self.critic.parameters())
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
        flat = torch.cat([g.reshape(-1) for g in grads])
        all_reduce_mean_(flat)
        offset = 0
        for p in params:
            p.grad = flat[offset:offset + p.numel()].view_as(p)
            offset += p.numel()
        return bool(torch.isfinite(flat).all())

    # ---------------------------------------------------------------- generator side
    def generator_loss(self, feats: dict, tau: torch.Tensor | None = None):
        """The adversarial term on the live active features.

        Gradient reaches ``feats``, never the critic, whose parameters are frozen for this
        graph. Otherwise the generator's backward would also accumulate into the critic.
        """
        phi, t = self._image(feats), self._text(tau)
        self.critic.eval()
        with _frozen(self.critic):
            logits = self.critic(phi, t)
        loss = sum(adversarial_loss(lg, self.cfg.loss) for lg in logits.values()) / len(self.names)
        mean_logit = torch.stack([lg.detach().mean() for lg in logits.values()]).mean()
        return loss, {"gan/g_loss": float(loss.detach()), "gan/g_fake": float(mean_logit)}

    def balance(self, mmd_loss: torch.Tensor, adv_loss: torch.Tensor, leaves: dict,
                image_dims: dict) -> tuple[float, dict]:
        """``lambda`` for ``mmd_loss + lambda * adv_loss`` (a plain float, no gradient).

        Adaptive: ``weight * |g_mmd| / |g_adv|`` over the image columns of ``leaves`` (the
        cached active features, every encoder), with squared norms summed across ranks. The
        floor ``min_norm_ratio * |g_mmd|`` on the denominator caps lambda when the critic has
        gone flat; the adversarial gradient then stays below ``weight * |g_mmd|``.
        """
        cfg = self.cfg
        if not cfg.adaptive:
            return float(cfg.weight), {"gan/lambda": float(cfg.weight)}
        names = list(leaves)
        inputs = [leaves[n] for n in names]
        g_mmd = torch.autograd.grad(mmd_loss, inputs, retain_graph=True, allow_unused=True)
        g_adv = torch.autograd.grad(adv_loss, inputs, retain_graph=True, allow_unused=True)

        def squared(grads):
            total = torch.zeros((), dtype=torch.float64, device=inputs[0].device)
            for n, g in zip(names, grads):
                if g is not None:
                    total = total + g[:, :image_dims[n]].double().pow(2).sum()
            return total

        norms = torch.stack([squared(g_mmd), squared(g_adv)])
        if is_dist():
            dist.all_reduce(norms)                       # SUM: squared norms add over ranks
        n_mmd, n_adv = norms.sqrt().tolist()
        lam = cfg.weight * n_mmd / max(n_adv, cfg.min_norm_ratio * n_mmd, 1e-30)
        return lam, {"gan/lambda": lam,
                     "gan/grad_ratio": lam * n_adv / max(n_mmd, 1e-30),
                     "gan/grad_ratio_raw": n_adv / max(n_mmd, 1e-30)}

    # ---------------------------------------------------------------- checkpointing
    def state_dict(self) -> dict:
        """Weights, the standardization buffers, spectral-norm vectors and AdamW moments."""
        return {"critic": self.critic.state_dict(), "optimizer": self.optimizer.state_dict(),
                "updates": self.updates}

    def load_state_dict(self, state: dict) -> None:
        self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state.get("updates", 0))


def build_critic(cfg: GANConfig, store, encoder_names, *, joint: bool = True, device="cpu",
                 dtype: torch.dtype = torch.float32) -> AdversarialCritic:
    """Assemble the critic over ``store`` for a subset of the training encoders."""
    cfg.validate()
    names = list(cfg.encoders) if cfg.encoders else list(encoder_names)
    unknown = [n for n in names if n not in encoder_names]
    if unknown:
        raise ValueError(f"gan.encoders {unknown} are not training encoders {list(encoder_names)}: "
                         f"the critic scores the features the force already computes")
    conditional = bool(cfg.conditional and joint)
    image_stats, text_stats = reference_stats(store, names, cfg.stats_rows, cfg.seed,
                                              with_text=conditional)
    # A forked, fixed-seed RNG: identical weights (and spectral-norm vectors) on every rank,
    # and the global stream the rest of the run consumes is left untouched.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        critic = FeatureCritic({n: store.dim(n) for n in names}, image_stats,
                               text_dim=int(store.text.shape[1]) if conditional else 0,
                               text_stats=text_stats, hidden=cfg.hidden, depth=cfg.depth,
                               spectral_norm=cfg.spectral_norm)
    return AdversarialCritic(critic, cfg, device=device, dtype=dtype)
