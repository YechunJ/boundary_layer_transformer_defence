#!/usr/bin/env python3
"""
dcor_privacy.py

Privacy term based on DISTANCE CORRELATION between the exposed representation
and the raw input -- NoPeek's mechanism -- replacing the rMLE/SSIM attack.

    Vepakomma, Singh, Gupta & Raskar. "NoPeek: Information leakage reduction
    to share activations in distributed deep learning." ICDMW 2020.
    Loss (their Eq. 1):  alpha1 * DCOR(X, Z) + alpha2 * CCE(Y_true, Y)

WHY dCor REPLACES rMLE HERE
---------------------------
rMLE scored leakage by reconstructing the image (500 gradient steps per point)
and measuring SSIM. On this ViT setup it failed: SSIM came back ~0.69 at EVERY
candidate layer including layer 2 -- exactly what a content-free grey image
scores against upsampled CIFAR. It returned a floor, not a measurement, so the
blocked-vs-unblocked difference was noise.

dCor measures the SAME underlying quantity -- how much the exposed tensor tells
you about the input -- but DIRECTLY, as a statistical dependence, with:
  * a closed form: no inner optimisation, no attack to train;
  * differentiability: the gradient reaches the boundary weights, so the block
    is actively trained to decorrelate from the input (NoPeek Eq. 1). rMLE's
    SSIM was a detached constant and could not do this;
  * no tuning: parameter-free, unlike rMLE's per-layer lambda_TV (the knob that
    sank it) and unlike HSIC/MMD which need kernels;
  * cost cheap enough to evaluate on EVERY batch, removing the refresh/cache
    staleness the rMLE version needed.

THE FORMULA (Szekely-Rizzo U-statistic; see WHY UNBIASED below)
---------------------------------------------------------------
    a_ij = ||x_i - x_j||        pairwise Euclidean distances within the batch
    b_ij = ||u_i - u_j||
    A~, B~ = U-centre(a), U-centre(b)          (unbiased centring)
    dCov2_U(X,U) = <A~, B~> / (n(n-3))
    dCor2_U      = dCov2_U(X,U) / sqrt( dVar2_U(X) * dVar2_U(U) )

dCor2 is ~0 under independence (may dip slightly negative -- the price of being
unbiased), and rises toward 1 with dependence. HIGH = the exposed tensor reveals
a lot about the input = BAD privacy. For scale, NoPeek Fig. 13 reports dCor
dropping 0.6 -> 0.22 with their training -- the dynamic range a working signal
has, and exactly what the flat 0.69 SSIM lacked.

WHY UNBIASED (this was a real bug in the naive port)
----------------------------------------------------
NoPeek's Appendix-B code uses the classical (V-statistic) double-centring. That
estimator is BIASED, and the bias explodes with dimension. Measured on
independent Gaussians at n=64:
    d=20    -> 0.74     d=1000 -> 0.99     d=150528 -> 0.9999
The cause is distance concentration: at d=150k every pairwise distance collapses
to one value, so any two distance matrices look identical and correlate ~1
regardless of real dependence. Our inputs are 3*224*224 = 150,528-dim and tokens
are 197*D'; the classical estimator would report ~1.0 leakage everywhere and
carry no signal. The U-statistic form (Szekely & Rizzo 2013) is unbiased and
sits at ~0 under independence at ANY dimension. It needs n > 3.

TWO NUMERICAL GUARDS, both load-bearing
---------------------------------------
1. fp32 forced. Under fp16 autocast the sum of squares over 150,528 dims is
   ~1.5e5, which overflows fp16's 65504 ceiling to inf. Everything here runs in
   float().
2. dCor^2 (squared) is used as the signal, never sqrt(dCor2). d/dx sqrt(x) is
   infinite at 0, and the diagonal of a distance matrix is exactly 0, so a
   sqrt would produce NaN gradients. The squared form is smooth, monotone in
   dependence, and has finite gradients everywhere.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


# -----------------------------------------------------------------------------
# Core distance-correlation machinery (U-statistic, unbiased)
# -----------------------------------------------------------------------------

def pairwise_dist(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Euclidean pairwise distance matrix.  A: [n, d]  ->  [n, n].

    Uses ||a-b||^2 = ||a||^2 - 2<a,b> + ||b||^2 rather than an explicit
    [n, n, d] difference tensor, which for d=150k would be enormous.
    clamp(min=0) guards float cancellation; +eps under the sqrt keeps the
    gradient finite on the zero diagonal.
    """
    r = (A * A).sum(dim=1, keepdim=True)          # [n, 1]
    d2 = r - 2.0 * (A @ A.t()) + r.t()            # [n, n]
    return torch.sqrt(d2.clamp(min=0.0) + eps)


def u_center(D: torch.Tensor) -> torch.Tensor:
    """
    U-centring (Szekely & Rizzo 2013), the unbiased alternative to classical
    double-centring:

        A~_ij = D_ij - a_i./(n-2) - a_.j/(n-2) + a_../((n-1)(n-2))   for i != j
        A~_ii = 0

    Requires n > 3. See the module docstring for why the classical form is
    unusable at our dimensionality.
    """
    n = D.shape[0]
    if n <= 3:
        raise ValueError(f"u_center needs batch size > 3, got {n}")
    row = D.sum(dim=1, keepdim=True)
    col = D.sum(dim=0, keepdim=True)
    total = D.sum()
    U = D - row / (n - 2) - col / (n - 2) + total / ((n - 1) * (n - 2))
    off_diag = 1.0 - torch.eye(n, device=D.device, dtype=D.dtype)
    return U * off_diag


def double_center(D: torch.Tensor) -> torch.Tensor:
    """
    CLASSICAL (V-statistic) double-centring -- the estimator in NoPeek's
    Appendix B and the original Szekely-Rizzo-Bakirov 2007 paper:

        A_ij = D_ij - a_i. / n - a_.j / n + a_.. / n^2

    Unlike u_center this KEEPS the diagonal and divides by n (not n-2). It is
    BIASED: on independent inputs its dCor^2 rises with dimension (measured
    ~0.74 at d=20, ~0.9999 at d=150528; see the module docstring). Provided
    here so the two estimators can be benchmarked head to head on real data --
    it may still be usable if your batch/feature dims keep the bias small.
    Selected via center_mode='classical'.
    """
    n = D.shape[0]
    row = D.mean(dim=1, keepdim=True)     # a_i. / n
    col = D.mean(dim=0, keepdim=True)     # a_.j / n
    total = D.mean()                      # a_.. / n^2
    return D - row - col + total


def dcov2_v(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    CLASSICAL squared distance covariance: <A, B> / n^2, the V-statistic
    normalisation from the original paper (contrast dcov2_u's n(n-3)).
    Pairs with double_center.
    """
    n = A.shape[0]
    return (A * B).sum() / (n * n)


def dcov2_u(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Unbiased squared distance covariance from two U-centred matrices."""
    n = A.shape[0]
    return (A * B).sum() / (n * (n - 3))


def dcor2_from_centered(
    A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12, estimator: str = "u"
) -> torch.Tensor:
    """
    Squared distance correlation from two ALREADY-CENTRED distance matrices.
    Split out so the input side (A) is computed once and shared across layers.

    estimator='u'         : unbiased, dcov2 = <A,B>/(n(n-3)). A,B must be
                            u_center'd. Default. Independent -> ~0 at any dim.
    estimator='classical' : biased V-statistic, dcov2 = <A,B>/n^2. A,B must be
                            double_center'd. Matches NoPeek's Appendix B.

    The caller is responsible for centring A and B with the MATCHING routine
    (compute_centered picks it from estimator). Returns dCor^2 (squared form).
    """
    dcov2_fn = dcov2_v if estimator == "classical" else dcov2_u
    dcov2 = dcov2_fn(A, B)
    var_a = dcov2_fn(A, A).clamp(min=0.0)
    var_b = dcov2_fn(B, B).clamp(min=0.0)
    return dcov2 / torch.sqrt(var_a * var_b + eps)


def compute_centered(X: torch.Tensor, estimator: str = "u", eps: float = 1e-12) -> torch.Tensor:
    """Pairwise-distance then centre X with the routine matching `estimator`."""
    D = pairwise_dist(X.flatten(1), eps)
    return double_center(D) if estimator == "classical" else u_center(D)


def distance_correlation(
    X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12, estimator: str = "u"
) -> torch.Tensor:
    """
    dCor^2 between two batches. Trailing dims flattened, so images [n,3,224,224]
    and tokens [n,197,D'] both work. Differentiable w.r.t. both arguments.
    estimator: 'u' (unbiased, default) or 'classical' (NoPeek Appendix B).
    """
    A = compute_centered(X, estimator, eps)
    B = compute_centered(Y, estimator, eps)
    return dcor2_from_centered(A, B, eps, estimator)


# -----------------------------------------------------------------------------
# The privacy term
# -----------------------------------------------------------------------------

def privacy_term(
    model,
    exposures: Dict[int, Dict[str, torch.Tensor]],
    images: torch.Tensor,
    kind: str = "u",
    r_mode: str = "raw",
    detach: bool = False,
    eps: float = 1e-12,
    estimator: str = "u",
) -> Tuple[torch.Tensor, Dict[int, Dict[str, float]]]:
    """
    Computes   sum_l alpha_l * s_l   (caller multiplies by mu). Returns
    (term, diagnostics).

    SIGN CONVENTION -- identical to the rMLE version so nothing downstream
    changes. Under  L += mu * sum_l alpha_l * s_l :
        s_l > 0  ->  dL/d(alpha_l) > 0  ->  gate CLOSES
        s_l < 0  ->  gate OPENS

      r_mode='raw'       s_l = + dCor2(exposed_l, x)
          Dependence still present AFTER the block; high -> close that gate.
          Carries the same depth confound rMLE had: NoPeek Fig. 11 states the
          network reduces dCor on its own, so deep layers score well on depth.

      r_mode='marginal'  s_l = - ( dCor2(z_l, x) - dCor2(exposed_l, x) )
          How much the block REDUCED dependence, measured at the same depth.
          The depth trend cancels; this is the default.

    kind: 'u' (transmitted bottleneck -- what crosses the wire) or 'zout'.
    detach: False (default) lets the gradient reach the BOUNDARY WEIGHTS as well
        as the gates, so the blocks learn to decorrelate (NoPeek Eq. 1). True
        restricts it to the gates (reproduces the rMLE detached behaviour).
    """
    if kind not in ("u", "zout"):
        raise ValueError(f"kind must be 'u' | 'zout', got {kind!r}")
    if r_mode not in ("raw", "marginal"):
        raise ValueError(f"r_mode must be 'raw' | 'marginal', got {r_mode!r}")

    # fp32 -- fp16 overflows on the sum of squares over 150,528 dims (see docstr)
    if estimator not in ("u", "classical"):
        raise ValueError(f"estimator must be 'u' | 'classical', got {estimator!r}")
    x = images.detach().float()
    A_in = compute_centered(x, estimator, eps)           # computed ONCE, reused

    terms: List[torch.Tensor] = []
    diag: Dict[int, Dict[str, float]] = {}

    for l in model.candidate_layers:
        exp = exposures[l]

        B_blk = compute_centered(exp[kind].float(), estimator, eps)
        d_blk = dcor2_from_centered(A_in, B_blk, eps, estimator)

        if r_mode == "raw":
            s = d_blk
            d_raw_val = float("nan")
        else:
            B_raw = compute_centered(exp["raw"].float(), estimator, eps)
            d_raw = dcor2_from_centered(A_in, B_raw, eps, estimator)
            s = -(d_raw - d_blk)
            d_raw_val = float(d_raw.detach())

        if detach:
            s = s.detach()

        terms.append(model.alphas[str(l)] * s)
        diag[l] = {
            "dcor_raw": d_raw_val,
            f"dcor_{kind}": float(d_blk.detach()),
            "s": float(s.detach()),
        }

    device = next(model.parameters()).device
    total = torch.stack(terms).sum() if terms else torch.zeros((), device=device)
    return total, diag


def format_dcor_table(diag: Dict[int, Dict[str, float]], kind: str = "u") -> str:
    """One-line summary: L6: raw=0.612 u=0.488 s=-0.124 | L8: ..."""
    parts = []
    for l in sorted(diag):
        d = diag[l]
        raw = d.get("dcor_raw", float("nan"))
        blk = d.get(f"dcor_{kind}", float("nan"))
        if raw == raw:  # not NaN
            parts.append(f"L{l}: raw={raw:.3f} {kind}={blk:.3f} s={d['s']:+.3f}")
        else:
            parts.append(f"L{l}: {kind}={blk:.3f} s={d['s']:+.3f}")
    return " | ".join(parts)