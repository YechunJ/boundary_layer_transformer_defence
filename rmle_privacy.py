#!/usr/bin/env python3
"""
rmle_privacy.py

Per-layer privacy score r_l for defense_training.py, measured with the
regularized Maximum Likelihood Estimation (rMLE) white-box inversion attack:

    He, Zhang & Lee. "Model Inversion Attacks Against Collaborative
    Inference." ACSAC 2019.  (Algorithm 1, Section 4.1)

WHICH PAPER SCRIPT THIS PORTS
-----------------------------
The paper ships four scripts. Algorithm 1 is the `inverse()` function, and it
appears in all four; the one to copy is `inverse_whitebox_CIFAR_defense.py`,
because it is the only one that (a) evaluates a NOISE DEFENSE and (b) returns
PSNR/SSIM. `inverse_whitebox_CIFAR.py` is the same attack without the metrics.
The two MNIST files are the same code on a different dataset.

Ported changes, all necessary:
  * Python 2 -> 3 (`print "x"` is a syntax error in 3).
  * `skimage.measure.compare_ssim` was removed from skimage; and skimage is not
    in your Narval venv. SSIM is reimplemented here in pure torch (~30 lines),
    so there is no new dependency.
  * CNN feature maps -> ViT token tensors [B, T, D]. The Euclidean-distance
    term is shape-agnostic so it carries over unchanged.
  * Batched: the paper attacks one image at a time (one per class). Attacking
    a batch in parallel is the same computation and much faster on an A100.

THE OBJECTIVE (Eq. 3 in the paper)
----------------------------------
    ED(x, x0) = || f(x) - f(x0) ||^2          feature-space fidelity
    TV(x)     = sum_ij (|x_{i+1,j}-x_ij|^2 + |x_{i,j+1}-x_ij|^2)^{beta/2}
    x*        = argmin_x  ED(x, x0) + lambda_TV * TV(x)

solved by Adam on the image, starting from constant gray. In the boundary
setting `f` is whatever representation is being scored. THE DEFAULT SCORING
PATH USES z_out, the boundary's output, so that "with block" and "without
block" are the same kind of tensor at the same depth.

CAVEAT, recorded deliberately: z_out is computed by the SERVER from the
received u; only u crosses the wire. An untrusted server already holds u and
can produce z_out itself, so scoring z_out measures something strictly
downstream of the true exposure point and will read as more private than
reality. `model.forward_to_u` is still defined, and forward_collect returns
"u" alongside "raw" and "zout" -- change `kinds` in measure_scores to score the
transmitted tensor instead.

HOW r_l ENTERS THE LOSS  -- read this, the sign and the weighting both matter
----------------------------------------------------------------------------
SSIM measures similarity between the reconstruction and the true image, so
HIGH SSIM = the attack succeeded = BAD privacy. The slide-2 objective is

    R_privacy(alpha) = sum_l alpha_l * r_l ,   entering L as  - mu * R_privacy

with r_l = "privacy benefit". Setting r_l = -SSIM_l gives

    L = L_task + lambda * sum_l |alpha_l| + mu * sum_l alpha_l * SSIM_l

Two things about this that are easy to get wrong:

  1. The alpha WEIGHTING is load-bearing. A bare `+ SSIM` term (unweighted) is
     a constant with respect to alpha -- it changes the loss value and moves
     nothing. Every gate would get the identical L1 push and the privacy
     measurement would have no effect at all. It must be alpha_l * SSIM_l.

  2. SSIM_l is DETACHED on purpose. Differentiating through the attack would
     mean backpropagating through a 500-step inner optimisation (bi-level,
     DARTS-second-order style) -- enormously expensive and unnecessary. In
     `sum_l alpha_l * r_l` the gradient w.r.t. alpha_l is just r_l, so a
     measured constant is exactly what the formula wants.

Result: a layer whose block makes inversion hard (low SSIM_l) gets a small
extra downward push and survives the sparsity penalty; a leaky layer (high
SSIM_l) gets a large downward push and collapses.

COST -- why this is measured periodically, not every batch
----------------------------------------------------------
One rMLE run is `n_iters` forward+backward passes through the client portion.
The paper uses 500 (shallow) to 5000 (deep) iterations. Per training batch
that would multiply your step cost by ~500x; at ~390 batches/epoch it is
completely infeasible.

So r_l is REFRESHED every `refresh_every` epochs on a small fixed probe batch,
cached, and reused for every step in between. The cached value is stale by
construction (the weights keep moving) -- that is the standard approximation
in this kind of alternating scheme, and it is acceptable because r_l only has
to RANK layers, not be exact.

Use `--benchmark` in defense_training.py to time one refresh on your own GPU
before committing to a full run.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# SSIM  (pure torch -- skimage is not available in the Narval venv)
# -----------------------------------------------------------------------------

def _gaussian_window(win_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(win_size, dtype=dtype, device=device) - win_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    win2d = g[:, None] @ g[None, :]
    return win2d.expand(channels, 1, win_size, win_size).contiguous()


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 1.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
) -> torch.Tensor:
    """
    Structural Similarity Index, per image.

    x, y : [N, C, H, W], values in [0, data_range]
    returns: [N] tensor of SSIM values in roughly [-1, 1]; 1.0 = identical.

    Same formulation as skimage's `structural_similarity` with a Gaussian
    window (gaussian_weights=True), which is what the paper's `compare_ssim`
    call resolves to.
    """
    if x.shape != y.shape:
        raise ValueError(f"ssim: shape mismatch {tuple(x.shape)} vs {tuple(y.shape)}")

    C = x.shape[1]
    win = _gaussian_window(win_size, win_sigma, C, x.device, x.dtype)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu_x = F.conv2d(x, win, groups=C)
    mu_y = F.conv2d(y, win, groups=C)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sig_x2 = F.conv2d(x * x, win, groups=C) - mu_x2
    sig_y2 = F.conv2d(y * y, win, groups=C) - mu_y2
    sig_xy = F.conv2d(x * y, win, groups=C) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
    return (num / den).mean(dim=[1, 2, 3])


def psnr(x: torch.Tensor, y: torch.Tensor, peak: float = 1.0) -> torch.Tensor:
    """PSNR per image, Eq. 2 in the paper. Reported alongside SSIM for context."""
    mse = ((x - y) ** 2).flatten(1).mean(dim=1).clamp(min=1e-12)
    return 10.0 * torch.log10(peak ** 2 / mse)


# -----------------------------------------------------------------------------
# Total Variation prior  (Eq. 3b)
# -----------------------------------------------------------------------------

def total_variation(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    TV(x) = sum_ij ( |x_{i+1,j} - x_ij|^2 + |x_{i,j+1} - x_ij|^2 )^{beta/2}

    The paper uses beta = 1.0 and notes that is enough for good results.
    Returned as a MEAN rather than a sum so lambda_TV does not have to be
    rescaled when the image size changes (the paper's CNNs use 32x32; the
    boundary pipeline upsamples CIFAR to 224x224, a 49x difference in pixel
    count, which would otherwise shift the useful lambda_TV range by ~49x).
    """
    dh = (x[:, :, 1:, :-1] - x[:, :, :-1, :-1]) ** 2
    dw = (x[:, :, :-1, 1:] - x[:, :, :-1, :-1]) ** 2
    return ((dh + dw + 1e-12) ** (beta / 2.0)).mean()


# -----------------------------------------------------------------------------
# Algorithm 1: white-box model inversion (rMLE)
# -----------------------------------------------------------------------------

def rmle_attack(
    client_fn: Callable[[torch.Tensor], torch.Tensor],
    u_target: torch.Tensor,
    x_shape: Tuple[int, ...],
    device,
    n_iters: int = 500,
    lr: float = 1e-2,
    lambda_tv: float = 1e1,
    beta: float = 1.0,
    eps: float = 1e-3,
    amsgrad: bool = True,
    init_value: float = 0.0,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Algorithm 1 of He et al., ACSAC'19.

        L(x) = ||client_fn(x) - u_target||^2 + lambda_tv * TV(x)
        x^(0) = ConstantInit();  x^(t+1) = x^(t) - eps * dL/dx

    client_fn : x -> u.  Everything the client computes and transmits. Must be
                differentiable w.r.t. x. Model weights should already be frozen
                by the caller (requires_grad=False); only x is optimised.
    u_target  : the intercepted representation, [N, ...], detached.
    x_shape   : shape of the image tensor to recover, e.g. (N, 3, 224, 224),
                in the model's INPUT space (ImageNet-normalised here).
    init_value: 0.0 in normalised space == the ImageNet mean == constant gray,
                which is the paper's `ConstantInit()` (its code likewise uses
                torch.zeros on normalised inputs).

    Returns the reconstruction, detached, same shape as x_shape.
    """
    x_gen = torch.full(x_shape, float(init_value), device=device, requires_grad=True)
    opt = torch.optim.Adam([x_gen], lr=lr, eps=eps, amsgrad=amsgrad)
    u_target = u_target.detach()

    for t in range(n_iters):
        opt.zero_grad(set_to_none=True)
        u_gen = client_fn(x_gen)
        feature_loss = ((u_gen - u_target) ** 2).mean()
        tv = total_variation(x_gen, beta=beta)
        (feature_loss + lambda_tv * tv).backward()
        opt.step()

        if verbose and (t % max(1, n_iters // 5) == 0 or t == n_iters - 1):
            print(f"      rMLE iter {t:>5}: ED={feature_loss.item():.6f} "
                  f"TV={tv.item():.6f}")

    return x_gen.detach()


# -----------------------------------------------------------------------------
# Denormalisation, so SSIM is computed in pixel space
# -----------------------------------------------------------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Undo the ImageNet normalisation applied by build_dataloaders, then clamp
    to [0,1]. The paper's `deprocess()` does the same thing before computing
    its metrics -- SSIM is only meaningful on pixel-space images."""
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


# -----------------------------------------------------------------------------
# The measurement: one score s_l per candidate layer
# -----------------------------------------------------------------------------
#
# THREE SCORING MODES. All feed the same loss term
#
#       L += mu * sum_l alpha_l * s_l
#
# so the SIGN is baked into s_l: positive s_l pushes that gate DOWN, negative
# s_l pushes it UP. Pick with `r_mode` in the config.
#
#   'raw'           s_l = + SSIM_zout(l)
#       Leakage at layer l. Simple, one attack per layer.
#       PROBLEM: SSIM falls with depth on its own -- the paper states this
#       plainly ("the quality of recovered images decreases when the split
#       layer goes deeper"). So deep layers score well on DEPTH, not on block
#       quality, and the search inherits a positional bias.
#
#   'marginal'      s_l = - ( SSIM_raw(l) - SSIM_zout(l) )                <- clean
#       How much the block at l reduces reconstructability, measured AT THE
#       SAME DEPTH with the block present vs absent. The depth trend cancels
#       exactly because both terms sit at layer l; there is no spacing
#       artefact and no boundary case at the deepest layer. This is the
#       marginal contribution of the block itself, which is what a placement
#       search is trying to rank. Costs 2 attacks per layer instead of 1.
#
# -----------------------------------------------------------------------------


@torch.enable_grad()
def measure_scores(
    model,
    probe_images: torch.Tensor,
    device,
    r_mode: str = "raw",
    attack_kind: str = "zout",
    n_iters: int = 500,
    lr: float = 1e-2,
    lambda_tv: float = 1e1,
    beta: float = 1.0,
    verbose: bool = True,
) -> Tuple[Dict[int, float], Dict[str, Dict]]:
    """
    Walk the network ONCE, capture every exposure point, attack each, and
    return the per-layer scores.

        for each candidate layer l:
            raw_l   = z_l          (what would be exposed with NO block at l)
            zout_l  = B_l(z_l)     (what is exposed WITH the block at l)
            SSIM_raw(l), SSIM_zout(l) = rMLE against each
            store the difference

    scores[l] is s_l, sign-adjusted for  L += mu * sum_l alpha_l * s_l:

        r_mode='raw'       s_l = + SSIM_zout(l)
        r_mode='marginal'  s_l = - ( SSIM_raw(l) - SSIM_zout(l) )

    TARGET (f_theta1(x0)): captured ONCE, per attack, by the single
    forward_collect below, with noise ON. raw_l and zout_l therefore sit
    downstream of the SAME upstream noise draws, so their difference isolates
    the block at l rather than upstream randomness. This replaces the old
    forward_to_raw / forward_to_zout, which each re-ran the network and so drew
    DIFFERENT upstream noise on every call.

    ATTACK LOOP (f_theta1(x^(t))): noise stays ON. The deployed client injects
    a fresh draw on every forward call, so the attacker's own n_iters forward
    passes do too. Consequences you are accepting by choosing this: the line-12
    gradient is stochastic, rMLE converges to a noise-AVERAGED reconstruction,
    and SSIM carries run-to-run variance -- raise n_iters (or average over a
    couple of probe passes) if the scores look jittery.
    """
    if attack_kind not in ("zout", "u"):
        raise ValueError(f"attack_kind must be 'zout' | 'u', got {attack_kind!r}")
    cands = list(model.candidate_layers)
    if r_mode == "raw":
        kinds = (attack_kind,)
    elif r_mode == "marginal":
        kinds = ("raw", attack_kind)
    else:
        raise ValueError(f"r_mode must be 'raw' | 'marginal', got {r_mode!r}")

    was_training = model.training
    model.eval()
    saved_rg = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in saved_rg:
        p.requires_grad_(False)

    probe_images = probe_images.to(device)
    diag: Dict[str, Dict] = {"points": {}, "r_mode": r_mode}
    measured: Dict[Tuple[int, str], float] = {}

    try:
        # ---- ONE traversal, noise ON: every target shares this realisation --
        with torch.no_grad():
            exposures = model.forward_collect(probe_images)
        targets = {
            (l, k): exposures[l][k].detach()
            for l in cands for k in kinds
        }
        if verbose:
            print(f"    [rMLE] one traversal captured {len(targets)} exposure "
                  f"point(s) over layers {cands}; attacking each for {n_iters} iters")

        ref = denormalize(probe_images)

        # ---- attack each point, noise ON (fixed target, live noisy client) ---
        for (l, kind) in sorted(targets):
            def fn(z, _l=l, _k=kind):
                return model.forward_collect(z, stop_at=_l)[_l][_k]

            # Noise stays ON inside the attack: f_theta1 keeps injecting a
            # fresh draw on every one of the n_iters forward calls, exactly as
            # a deployed noisy client would. The gradient in line 12 of the
            # paper's Algorithm 1 is therefore stochastic, and rMLE converges
            # to a noise-averaged reconstruction. The TARGET below was captured
            # once, with noise, so f_theta1(x0) is a single fixed observation.
            x_hat = rmle_attack(
                client_fn=fn, u_target=targets[(l, kind)],
                x_shape=tuple(probe_images.shape), device=device,
                n_iters=n_iters, lr=lr, lambda_tv=lambda_tv, beta=beta,
                verbose=False,
            )

            rec = denormalize(x_hat)
            sv = ssim(rec, ref).mean().item()
            pv = psnr(rec, ref).mean().item()
            measured[(l, kind)] = sv
            diag["points"][f"L{l}_{kind}"] = {
                "ssim": sv, "psnr": pv,
                "shape": list(targets[(l, kind)].shape),
            }
            if verbose:
                tag = "block" if kind == attack_kind else " none"
                print(f"      layer {l:>2} [{tag}]  target"
                      f"{tuple(targets[(l, kind)].shape)}  "
                      f"SSIM={sv:.4f}  PSNR={pv:.2f}dB")

        # ---- assemble s_l ---------------------------------------------------
        scores: Dict[int, float] = {}
        for l in cands:
            if r_mode == "raw":
                scores[l] = +measured[(l, attack_kind)]
            else:
                d = measured[(l, "raw")] - measured[(l, attack_kind)]
                scores[l] = -d
                diag["points"].setdefault(f"L{l}_marginal", {})["value"] = d
    finally:
        for p_, rg in saved_rg:
            p_.requires_grad_(rg)
        if was_training:
            model.train()

    if verbose:
        print("    [rMLE] scores s_l  (positive -> push gate DOWN, "
              "negative -> push gate UP):")
        for l in cands:
            extra = ""
            if r_mode == "marginal":
                extra = (f"   [SSIM_raw-SSIM_zout = "
                         f"{diag['points'][f'L{l}_marginal']['value']:+.4f}]")
            print(f"      layer {l:>2}: s = {scores[l]:+.4f}{extra}")

    diag["scores"] = dict(scores)
    return scores, diag


# Back-compat alias: the original single-mode entry point.
def measure_layer_ssim(model, probe_images, device, **kw) -> Dict[int, float]:
    kw.pop("r_mode", None)
    scores, _ = measure_scores(model, probe_images, device, r_mode="raw", **kw)
    return scores


def privacy_reward_term(model, scores: Dict[int, float]) -> torch.Tensor:
    """
    sum_l alpha_l * s_l    (caller multiplies by mu)

    s_l is a python float with no grad_fn, so d/d(alpha_l) = s_l exactly --
    which is what `sum_l alpha_l * r_l` means. No differentiation through the
    attack is needed or wanted.
    """
    dev = next(model.parameters()).device
    if not scores:
        return torch.zeros((), device=dev)
    terms = [
        model.alphas[str(l)] * float(s)
        for l, s in scores.items()
        if str(l) in model.alphas
    ]
    return torch.stack(terms).sum() if terms else torch.zeros((), device=dev)