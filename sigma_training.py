#!/usr/bin/env python3
"""
sigma_training.py

Learns the per-layer Gaussian noise level `sigma` of the boundary defense,
using the REAL deployed mechanism (no alpha, no residual highway).

HOW THIS DIFFERS FROM defense_training.py
-----------------------------------------
  defense_training.py :  Z_out = Z + alpha_l * B_l(Z)     <- gated supernet.
                         The original Z survives, so the network can route
                         around the bottleneck. Good for ranking placements,
                         but it is NOT the network you deploy.

  this file           :  Z_out, u = B_l(Z)                 <- real mechanism.
                         The boundary REPLACES the signal, exactly as in
                         train_deit_boundary_new_boundaries.py. All information
                         is forced through the noisy bottleneck u, so the
                         data-processing argument holds and whatever this
                         script learns transfers to deployment directly.

WHAT IT SEARCHES
----------------
Not placement -- STRENGTH. Because the real block always compresses and
quantizes regardless of sigma, sigma=0 does NOT mean "no boundary here". So
this cannot select placements; every listed layer gets a real, permanent
boundary. Decide placement first (slide-3 sweep, or defense_training.py),
then run this to find how much noise those placements can tolerate.

    candidate_layers: [9]      -> "how much noise can a boundary at 9 take?"
    candidate_layers: [6, 9]   -> two REAL boundaries in series (see warning
                                  in the docstring of DeiTWithLearnableSigma).

THE OBJECTIVE (read sigma_penalty() before running -- the sign matters)
----------------------------------------------------------------------
    L(theta, sigma) = L_task + lambda_sigma * sum_l sigma_l

  lambda_sigma < 0  ->  the term REWARDS noise. Task loss pushes sigma down,
                        this pushes it up, equilibrium = the max noise each
                        layer tolerates. This is the working configuration.

  lambda_sigma > 0  ->  DEGENERATE. Both terms push sigma toward 0; the global
                        optimum is sigma=0 everywhere, i.e. no noise, no
                        privacy. The script warns loudly if you do this.

USAGE
-----
    python sigma_training.py --config configs/generated/deit_tiny_cifar100_sigma.yaml
    python sigma_training.py --config <cfg>.yaml --lambda-sigma -0.05
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from train_deit_boundary_new_boundaries import (
    AverageMeter,
    accuracy_top1,
    build_boundary_module,
    build_dataloaders,
    build_scheduler,
    ensure_dir,
    load_yaml,
    set_seed,
)


# -----------------------------------------------------------------------------
# Making sigma learnable
# -----------------------------------------------------------------------------

class LearnableSigmaBoundary(nn.Module):
    """
    Wraps any boundary module and replaces its fixed float `sigma` with a
    learnable, bounded parameter.

    WHY A LOGIT INSTEAD OF A RAW sigma PARAMETER
    --------------------------------------------
    The boundary modules gate their noise on `if self.sigma > 0`. If sigma were
    a raw learnable scalar, nothing stops gradient descent pushing it negative
    -- at which point the condition fails, the noise line is skipped, sigma
    leaves the computation graph entirely, and it receives no further gradient.
    It would be stuck at "no noise" permanently. A one-way cliff.

    So sigma is parameterised as

        sigma_l = sigma_max * sigmoid(s_l)

    which is smooth, strictly positive (never trips the cliff), and bounded
    above by sigma_max (so it cannot run away to a degenerate all-noise
    solution). The learnable parameter is the unconstrained logit s_l.

    HOW THE GRADIENT REACHES sigma
    ------------------------------
    The noise line is  u = u + sigma * randn_like(u), so  d(u)/d(sigma) = eps,
    the sampled noise itself. This is the reparameterisation trick: the
    gradient is well-defined and unbiased. It IS noisy, though -- it depends on
    which eps was drawn this step -- so sigma typically wants a larger learning
    rate and more steps than ordinary weights. Hence `sigma_lr` in the config.
    """

    def __init__(self, boundary: nn.Module, sigma_init: float, sigma_max: float) -> None:
        super().__init__()
        if not (0.0 < sigma_init < sigma_max):
            raise ValueError(
                f"need 0 < sigma_init ({sigma_init}) < sigma_max ({sigma_max})"
            )
        self.inner = boundary
        self.sigma_max = float(sigma_max)

        # Invert the sigmoid so training starts exactly at sigma_init.
        p = sigma_init / sigma_max
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        self.sigma_logit = nn.Parameter(torch.tensor(math.log(p / (1.0 - p))))

        # The inner module must not skip its own noise line.
        self.inner.noise_in_eval = True

    def effective_sigma(self) -> torch.Tensor:
        return self.sigma_max * torch.sigmoid(self.sigma_logit)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.sigma == 0:
            return z, z
        # Hand the inner module a *tensor* sigma carrying grad_fn. The inner
        # forward then does `u + self.sigma * randn_like(u)` as usual, and the
        # gradient flows back into sigma_logit. `sigma` was a plain float
        # attribute (not a registered Parameter/buffer), so assigning to it is
        # an ordinary attribute set -- no nn.Module bookkeeping is disturbed.
        self.inner.sigma = self.effective_sigma()
        return self.inner(z)


# -----------------------------------------------------------------------------
# The model: real mechanism, no alpha, no residual
# -----------------------------------------------------------------------------

class DeiTWithLearnableSigma(nn.Module):
    """
    DeiT with REAL boundaries at the given layers, each with a learnable sigma.

    WARNING -- boundaries compose in series.
    With more than one layer listed, boundary #2 compresses+noises the OUTPUT
    of boundary #1. Damage compounds; there is no residual path to soften it
    (that is the whole point of using the real mechanism). Two is already
    aggressive; eight would destroy the signal and the run would be
    meaningless. Keep this list to the placements you actually intend to ship.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        layers: List[int],
        boundary_cfg: Dict[str, Any],
        sigma_init: float,
        sigma_max: float,
        pretrained: bool = False,
    ) -> None:
        super().__init__()

        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        if not hasattr(self.vit, "blocks"):
            raise ValueError("Expected a timm VisionTransformer-like model with .blocks")

        self.num_blocks = len(self.vit.blocks)
        self.embed_dim = self.vit.embed_dim

        num_patches = self.vit.patch_embed.num_patches
        extra_tokens = 1
        if getattr(self.vit, "dist_token", None) is not None:
            extra_tokens += 1
        self.seq_len = num_patches + extra_tokens

        self.layers = sorted({int(l) for l in layers})
        if not self.layers:
            raise ValueError("sigma_search.layers must not be empty")
        for l in self.layers:
            if not (0 <= l <= self.num_blocks):
                raise ValueError(f"layer {l} out of range [0, {self.num_blocks}]")
        if len(self.layers) > 2:
            print(
                f"[WARNING] {len(self.layers)} boundaries in series with the real "
                f"(non-residual) mechanism. Damage compounds and accuracy will "
                f"likely collapse. Consider 1-2 layers."
            )

        # `sigma` in boundary_cfg is ignored -- LearnableSigmaBoundary overrides it.
        self.boundaries = nn.ModuleDict()
        for l in self.layers:
            inner = build_boundary_module(boundary_cfg, seq_len=self.seq_len, dim=self.embed_dim)
            self.boundaries[str(l)] = LearnableSigmaBoundary(inner, sigma_init, sigma_max)

        self.last_exposed = None

        print(
            f"[LearnableSigma] {model_name} | blocks={self.num_blocks} | "
            f"seq_len={self.seq_len} | boundary layers={self.layers} | "
            f"sigma_init={sigma_init} | sigma_max={sigma_max}"
        )

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        if getattr(self.vit, "dist_token", None) is not None:
            dist_token = self.vit.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_token, dist_token, x), dim=1)
        else:
            x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit.patch_embed(x)
        x = self._pos_embed(x)
        exposed = None

        # THE REAL MECHANISM: x is REPLACED by the boundary's output.
        # No `x + alpha * B(x)`. Everything downstream is a function of u only.
        if 0 in self.layers:
            x, exposed = self.boundaries["0"](x)

        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            l = i + 1
            if l in self.layers:
                x, exposed = self.boundaries[str(l)](x)

        x = self.vit.norm(x)
        if getattr(self.vit, "dist_token", None) is not None:
            feat = (x[:, 0] + x[:, 1]) / 2
        else:
            feat = x[:, 0]

        self.last_exposed = exposed
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.vit.head(self.forward_features(x))

    def sigma_dict(self) -> Dict[int, float]:
        return {
            int(l): float(b.effective_sigma().detach().cpu())
            for l, b in self.boundaries.items()
        }


# -----------------------------------------------------------------------------
# Objective
# -----------------------------------------------------------------------------

def compute_task_loss(
    logits: torch.Tensor, targets: torch.Tensor, num_classes: int, mode: str = "cross_entropy"
) -> torch.Tensor:
    """Same two options as defense_training.py: cross_entropy (default) or mse ('Y - Y'')."""
    if mode == "cross_entropy":
        return F.cross_entropy(logits, targets)
    if mode == "mse":
        y_true = F.one_hot(targets, num_classes=num_classes).to(logits.dtype)
        y_pred = logits.softmax(dim=1)
        return F.mse_loss(y_pred, y_true)
    raise ValueError(f"Unknown task_loss mode: {mode!r}")


def sigma_penalty(model: DeiTWithLearnableSigma) -> torch.Tensor:
    """
    sum_l sigma_l.

    THE SIGN IS THE WHOLE STORY. This function returns the raw sum; the caller
    multiplies by lambda_sigma. With

        L = L_task + lambda_sigma * sum_l sigma_l

    a POSITIVE lambda_sigma is degenerate: L_task already penalises noise
    (noise hurts accuracy), so both terms point the same way and the optimum is
    sigma = 0 everywhere. A NEGATIVE lambda_sigma turns the term into a reward
    for noise and creates the real accuracy-vs-privacy tension.

    Note also what this sum does NOT do with multiple layers: it constrains the
    TOTAL, not each layer. The optimizer is free to dump all the noise into
    whichever layer is cheapest accuracy-wise and leave the others near zero.
    If you want every layer to defend, penalise/reward min_l sigma_l, or add a
    per-layer floor.
    """
    return torch.stack([b.effective_sigma() for b in model.boundaries.values()]).sum()


# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

def load_baseline_checkpoint(model: nn.Module, cfg: Dict[str, Any], device) -> None:
    ckpt_path = cfg.get("checkpoint", {}).get("load", None)
    if not ckpt_path:
        print("[Info] No checkpoint.load given; backbone stays at init.")
        return
    print(f"[Info] Loading baseline checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    n_b = sum(1 for k in missing if k.startswith("boundaries."))
    print(f"[Info] Missing keys: {len(missing)} ({n_b} boundary params -- expected)")
    print(f"[Info] Unexpected keys: {len(unexpected)}")


def apply_freeze_policy(model: DeiTWithLearnableSigma, cfg: Dict[str, Any]) -> None:
    """
    Mirrors the deployed recipe: freeze the prefix, train the boundaries and
    everything downstream of the FIRST one, plus final norm and head.
    """
    train_cfg = cfg["train"]
    freeze_prefix = train_cfg.get("freeze_prefix", True)

    if not freeze_prefix:
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.boundaries.parameters():
            p.requires_grad = True
        first = min(model.layers)
        for i in range(first, model.num_blocks):
            for p in model.vit.blocks[i].parameters():
                p.requires_grad = True
        for p in model.vit.norm.parameters():
            p.requires_grad = True
        for p in model.vit.head.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Info] Trainable params: {trainable/1e6:.3f}M / {total/1e6:.3f}M")


def build_sigma_optimizer(model: DeiTWithLearnableSigma, cfg: Dict[str, Any]):
    """
    Two groups. sigma_logits get their own lr (their gradient is noisy -- it is
    multiplied by a fresh random eps every step) and weight_decay MUST be 0:
    AdamW's decay is an L2 pull toward logit=0, i.e. toward sigma = sigma_max/2,
    which is a meaningless bias that would silently contaminate lambda_sigma.
    """
    opt_cfg = cfg["optimizer"]
    name = opt_cfg.get("name", "adamw").lower()
    lr = float(opt_cfg.get("lr", 1e-4))
    wd = float(opt_cfg.get("weight_decay", 0.05))
    sigma_lr = float(opt_cfg.get("sigma_lr", 1e-2))
    sigma_wd = float(opt_cfg.get("sigma_weight_decay", 0.0))

    sigma_params = [
        b.sigma_logit for b in model.boundaries.values() if b.sigma_logit.requires_grad
    ]
    sigma_ids = {id(p) for p in sigma_params}
    weight_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in sigma_ids
    ]

    groups = [
        {"params": weight_params, "lr": lr, "weight_decay": wd},
        {"params": sigma_params, "lr": sigma_lr, "weight_decay": sigma_wd},
    ]
    print(
        f"[Info] Optimizer {name}: weights lr={lr} wd={wd} | "
        f"sigma lr={sigma_lr} wd={sigma_wd} ({len(sigma_params)} sigmas)"
    )
    if name == "adamw":
        return optim.AdamW(groups)
    if name == "sgd":
        return optim.SGD(groups, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, epoch, cfg, num_classes):
    model.train()
    tot_m, task_m, sig_m, acc_m = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    use_amp = cfg["train"].get("amp", True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    lam = float(cfg["sigma_search"]["lambda_sigma"])
    task_mode = cfg["sigma_search"].get("task_loss", "cross_entropy")

    pbar = tqdm(loader, desc=f"Sigma epoch {epoch}", ncols=110)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            t_loss = compute_task_loss(logits, targets, num_classes, task_mode)

        s_sum = sigma_penalty(model)          # fp32, outside autocast
        loss = t_loss.float() + lam * s_sum

        scaler.scale(loss).backward()
        grad_clip = cfg["train"].get("grad_clip", None)
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        tot_m.update(loss.item(), bs)
        task_m.update(t_loss.item(), bs)
        sig_m.update(s_sum.item(), bs)
        acc_m.update(accuracy_top1(logits.detach().float(), targets), bs)
        pbar.set_postfix(
            loss=f"{tot_m.avg:.4f}", task=f"{task_m.avg:.4f}",
            sig=f"{sig_m.avg:.4f}", acc=f"{acc_m.avg*100:.2f}%",
        )

    return tot_m.avg, task_m.avg, sig_m.avg, acc_m.avg


@torch.no_grad()
def evaluate(model, loader, device, cfg, num_classes):
    model.eval()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    use_amp = cfg["train"].get("amp", True)
    task_mode = cfg["sigma_search"].get("task_loss", "cross_entropy")

    for images, targets in tqdm(loader, desc="Eval", ncols=110):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = compute_task_loss(logits, targets, num_classes, task_mode)
        bs = images.size(0)
        loss_m.update(loss.item(), bs)
        acc_m.update(accuracy_top1(logits.float(), targets), bs)
    return loss_m.avg, acc_m.avg


def format_sigmas(model: DeiTWithLearnableSigma) -> str:
    d = model.sigma_dict()
    return " | ".join(f"L{l}:sigma={d[l]:.4f}" for l in sorted(d))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--lambda-sigma", type=float, default=None,
                        help="Override sigma_search.lambda_sigma. NEGATIVE rewards noise.")
    parser.add_argument("--task-loss", type=str, default=None, choices=["cross_entropy", "mse"])
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg.setdefault("sigma_search", {})
    if args.lambda_sigma is not None:
        cfg["sigma_search"]["lambda_sigma"] = args.lambda_sigma
    if args.task_loss is not None:
        cfg["sigma_search"]["task_loss"] = args.task_loss

    lam = float(cfg["sigma_search"]["lambda_sigma"])
    if lam > 0:
        print("=" * 74)
        print("[WARNING] lambda_sigma > 0.")
        print("  The task loss ALREADY pushes sigma down (noise hurts accuracy).")
        print("  A positive lambda pushes it down too, so BOTH terms agree and the")
        print("  global optimum is sigma = 0 at every layer: no noise, no privacy.")
        print("  This run will very likely just drive every sigma to ~0.")
        print("  Use a NEGATIVE lambda_sigma to reward noise and get a real tradeoff.")
        print("=" * 74)
    elif lam == 0:
        print("[WARNING] lambda_sigma == 0: no pressure on sigma except the task loss, "
              "which will drive it to ~0.")

    set_seed(cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")
    print(f"[Info] lambda_sigma={lam} | task_loss={cfg['sigma_search'].get('task_loss','cross_entropy')}")

    train_loader, test_loader, num_classes = build_dataloaders(cfg)

    ss = cfg["sigma_search"]
    model = DeiTWithLearnableSigma(
        model_name=cfg["model"].get("name", "deit_tiny_patch16_224"),
        num_classes=num_classes,
        layers=ss["layers"],
        boundary_cfg=cfg["boundary"],
        sigma_init=float(ss.get("sigma_init", 0.02)),
        sigma_max=float(ss.get("sigma_max", 0.5)),
        pretrained=cfg["model"].get("pretrained", False),
    )
    model.to(device)

    load_baseline_checkpoint(model, cfg, device)
    apply_freeze_policy(model, cfg)
    optimizer = build_sigma_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    epochs = cfg["train"].get("epochs", 20)
    best_acc = 0.0
    history = []

    print(f"[Info] sigmas @ init: {format_sigmas(model)}")

    for epoch in range(1, epochs + 1):
        tot, task, sig, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, cfg, num_classes
        )
        val_loss, val_acc = evaluate(model, test_loader, device, cfg, num_classes)
        if scheduler is not None:
            scheduler.step()

        print(
            f"[Epoch {epoch}] loss={tot:.4f} (task={task:.4f}, sum_sigma={sig:.4f}) "
            f"train_acc={tr_acc*100:.2f}% val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%"
        )
        print(f"           {format_sigmas(model)}")

        history.append({
            "epoch": epoch, "total_loss": tot, "task_loss": task,
            "sum_sigma": sig, "train_acc": tr_acc, "val_acc": val_acc,
            "sigmas": model.sigma_dict(),
        })

        if val_acc > best_acc:
            best_acc = val_acc
            if cfg.get("checkpoint", {}).get("save_best", True):
                out_dir = cfg["checkpoint"].get("out_dir", "./checkpoints")
                ensure_dir(out_dir)
                name = cfg["checkpoint"].get("name", "sigma_search")
                path = os.path.join(out_dir, f"{name}_epoch{epoch}_acc{best_acc:.4f}.pt")
                torch.save({
                    "epoch": epoch, "best_acc": best_acc,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "cfg": cfg, "sigmas": model.sigma_dict(),
                }, path)
                print(f"[Info] Saved checkpoint to {path}")

    final = model.sigma_dict()
    print("\n" + "=" * 68)
    print(f"[Result] Best val acc: {best_acc*100:.2f}%")
    print("[Result] Learned noise levels:")
    for l in sorted(final):
        print(f"           layer {l:>2}:  sigma = {final[l]:.5f}")
    print("=" * 68)
    print("[Note] These sigmas were learned with the REAL mechanism, so they can be\n"
          "       written straight into a boundary config as `sigma:` and retrained\n"
          "       with train_deit_boundary_new_boundaries.py -- unlike alpha, they\n"
          "       mean the same thing here as in deployment.")

    out_path = cfg.get("sigma_out", "./search_results/sigma_search.json")
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w") as f:
        json.dump({
            "final_sigmas": final, "best_val_acc": best_acc,
            "lambda_sigma": lam, "sigma_max": float(ss.get("sigma_max", 0.5)),
            "sigma_init": float(ss.get("sigma_init", 0.02)),
            "layers": model.layers,
            "task_loss": ss.get("task_loss", "cross_entropy"),
            "history": history,
        }, f, indent=2)
    print(f"[Info] Wrote results to {out_path}")


if __name__ == "__main__":
    main()