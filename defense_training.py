#!/usr/bin/env python3
"""
defense_training.py

Alpha-Gated Placement Search (steps 1-4 of the slide-2 algorithm).

WHAT THIS DOES
--------------
Instead of inserting ONE boundary at a hand-picked layer (what
train_deit_boundary_new_boundaries.py does), this builds a "supernet":
one candidate boundary block after EVERY candidate layer, each scaled by
its own learnable gate alpha_l:

    Z_out = Z + alpha_l * B_l(Z)

and trains the whole thing against:

    L(theta, alpha) = L_task + lambda * sum_l |alpha_l|

After training, the layers with the largest |alpha_l| are the ones the
search selected. Step 5 (prune) and step 6 (fine-tune final network) are
NOT done here -- this script's output is the ranking / selection.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
The third term of the full objective, the privacy reward
(- mu * sum_l alpha_l * r_l), is NOT implemented. It needs a cheap,
differentiable leakage proxy r_l (e.g. distance correlation between the
block output and the input image, or a small inverse decoder's error).
Until r_l exists, this script searches on task-loss + sparsity only.
See `privacy_reward()` at the bottom for the hook.

IMPORTANT CAVEAT (read before trusting results)
-----------------------------------------------
The gated form `Z + alpha*B(Z)` keeps a residual path around the block,
so the original signal Z is always still present during the search. The
DEPLOYED boundary (ResidualAdapterBoundary etc.) drops that residual --
its output is a function of the noisy bottleneck u only. So the network
trained here is NOT the network you would ship; this script's job is to
rank placements, not to produce a final defended model.

USAGE
-----
    python defense_training.py --config configs/generated/deit_tiny_cifar100_gated.yaml

    # override lambda from the CLI to sweep it without editing the YAML:
    python defense_training.py --config <cfg>.yaml --lambda-sparsity 0.05

EXAMPLE CONFIG (see also configs/generated/deit_tiny_cifar100_gated.yaml)
------------------------------------------------------------------------
    seed: 0
    mode: gated_search
    data:
      dataset: cifar100
      root: ../datasets
      image_size: 224
      download: false
    model:
      name: deit_tiny_patch16_224
      pretrained: false
    search:
      candidate_layers: [3, 4, 5, 6, 7, 8, 9, 10]
      k_max: 2
      alpha_init: 0.1
      lambda_sparsity: 0.01      # <-- the knob you asked for
      task_loss: cross_entropy   # or: mse   (the "Y - Y'" variant)
    boundary:                    # the DEFAULT MILD BLOCK B, shared by all candidates
      module: residual_adapter
      bottleneck_dim: 192
      sigma: 0.02
      quant_bits: 8
      power: 1.0
      power_norm_mode: clip
      noise_in_eval: true
      adapter_mlp_ratio: 2.0
      adapter_gamma_init: 0.1
      adapter_dropout: 0.0
    train:
      epochs: 20
      batch_size: 128
      num_workers: 4
      amp: true
      grad_clip: 1.0
      freeze_backbone: true
      unfreeze_head: true
    optimizer:
      name: adamw
      lr: 1.0e-04
      weight_decay: 0.05
      alpha_lr: 1.0e-02          # gates usually want a BIGGER lr than weights
      alpha_weight_decay: 0.0    # keep 0: see note in build_gated_optimizer
    scheduler:
      name: cosine
      eta_min: 1.0e-06
    checkpoint:
      load: checkpoints/deit_tiny_cifar100_baseline_epoch49_acc0.8394.pt
      out_dir: ./checkpoints
      name: deit_tiny_cifar100_gated_search
      save_best: true
    search_out: ./search_results/alpha_search.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# Reuse everything that already exists rather than re-implementing it.
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
# The gated supernet
# -----------------------------------------------------------------------------

class DeiTWithGatedBoundaries(nn.Module):
    """
    A DeiT with ONE candidate boundary block after EVERY candidate layer, each
    scaled by its own learnable gate alpha_l.

    Contrast with DeiTWithBoundary (the fixed-placement model):
      - that one has a single `self.boundary` and a single `after_block` int;
      - this one has `self.boundaries` (a ModuleDict, one per candidate layer)
        and `self.alphas` (a ParameterDict, one scalar per candidate layer).

    Layer indexing follows the same convention as DeiTWithBoundary:
      l = 0  -> insert before block 1
      l = 6  -> insert after the 6th transformer block
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        candidate_layers: List[int],
        boundary_cfg: Dict[str, Any],
        pretrained: bool = False,
        alpha_init: float = 0.1,
    ) -> None:
        super().__init__()

        self.vit = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )
        if not hasattr(self.vit, "blocks"):
            raise ValueError("Expected a timm VisionTransformer-like model with .blocks")

        self.num_blocks = len(self.vit.blocks)
        self.embed_dim = self.vit.embed_dim
        self.num_classes = num_classes

        # Same seq_len computation as DeiTWithBoundary: patches + CLS (+ dist).
        num_patches = self.vit.patch_embed.num_patches
        extra_tokens = 1
        if getattr(self.vit, "dist_token", None) is not None:
            extra_tokens += 1
        self.seq_len = num_patches + extra_tokens

        self.candidate_layers = sorted({int(l) for l in candidate_layers})
        for l in self.candidate_layers:
            if not (0 <= l <= self.num_blocks):
                raise ValueError(
                    f"candidate layer {l} out of range [0, {self.num_blocks}]"
                )
        if not self.candidate_layers:
            raise ValueError("search.candidate_layers must not be empty")

        # One INDEPENDENT copy of the mild block B per candidate layer, plus one
        # gate each. ModuleDict/ParameterDict need str keys, hence str(l).
        self.boundaries = nn.ModuleDict()
        self.alphas = nn.ParameterDict()
        for l in self.candidate_layers:
            self.boundaries[str(l)] = build_boundary_module(
                boundary_cfg=boundary_cfg,
                seq_len=self.seq_len,
                dim=self.embed_dim,
            )
            self.alphas[str(l)] = nn.Parameter(torch.tensor(float(alpha_init)))

        print(
            f"[GatedSupernet] {model_name} | blocks={self.num_blocks} | "
            f"seq_len={self.seq_len} | candidates={self.candidate_layers} | "
            f"alpha_init={alpha_init}"
        )

    # -- copied from DeiTWithBoundary so the two models stay behaviourally identical
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

    def _gated(self, l: int, z: torch.Tensor) -> torch.Tensor:
        """
        The step-2 formula:   Z_out = alpha_l * B_l(Z)

        The boundary modules return (z_out, u); B_l(Z) is z_out. u (the exposed
        bottleneck) is ignored here -- it is what a privacy reward r_l would
        consume once that term is implemented.
        """
        key = str(l)
        if self.alphas[key] <= 1e-2:
            return z
        b_out, _u = self.boundaries[key](z)
        return self.alphas[key] * b_out

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit.patch_embed(x)
        x = self._pos_embed(x)

        if 0 in self.candidate_layers:
            x = self._gated(0, x)

        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            l = i + 1


            if l in self.candidate_layers:
                x = self._gated(l, x)

        x = self.vit.norm(x)

        if getattr(self.vit, "dist_token", None) is not None:
            feat = (x[:, 0] + x[:, 1]) / 2
        else:
            feat = x[:, 0]
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit.head(self.forward_features(x))

    # -- convenience -----------------------------------------------------------
    def alpha_dict(self) -> Dict[int, float]:
        """Current gate values as plain floats, for logging/ranking."""
        return {int(l): float(a.detach().cpu()) for l, a in self.alphas.items()}


# -----------------------------------------------------------------------------
# The three pieces of the objective
# -----------------------------------------------------------------------------

def compute_task_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    mode: str = "cross_entropy",
) -> torch.Tensor:
    """
    L_task.

    mode="cross_entropy"  (DEFAULT, and the correct choice for classification)
        L = (1/N) sum_i [ -logits[i, y_i] + log sum_k exp(logits[i, k]) ]

    mode="mse"  -- the literal "Y - Y'" variant, for your own comparison.
        Turns the integer label y into a one-hot vector Y (e.g. class 7 of 100
        -> a 100-long vector that is 1.0 at index 7 and 0.0 everywhere else),
        turns the logits into probabilities Y' via softmax, then takes the mean
        squared difference:
            L = (1/(N*K)) sum_i sum_k (Y'[i,k] - Y[i,k])^2
        This is what your supervisor's "Y - Y'" literally describes. It trains,
        but it is generally worse than cross-entropy for classification: MSE
        treats all K outputs symmetrically and its gradient vanishes when the
        model is confidently wrong (softmax saturates), whereas cross-entropy's
        gradient stays large exactly there. Provided so you can measure the gap
        yourself rather than take that on faith.
    """
    if mode == "cross_entropy":
        return F.cross_entropy(logits, targets)

    if mode == "mse":
        y_true = F.one_hot(targets, num_classes=num_classes).to(logits.dtype)
        y_pred = logits.softmax(dim=1)
        return F.mse_loss(y_pred, y_true)

    raise ValueError(f"Unknown task_loss mode: {mode!r}. Use 'cross_entropy' or 'mse'.")

    # ---- equivalent hand-written forms, if you want to see the arithmetic ----
    #
    # cross-entropy, written out (identical to F.cross_entropy above):
    #     logp = logits.log_softmax(dim=1)              # [N, K]
    #     picked = logp.gather(1, targets[:, None])     # [N, 1]  -> log p_correct
    #     return -picked.mean()
    #
    # "Y - Y'" on raw logits instead of probabilities (another reading of it):
    #     y_true = F.one_hot(targets, num_classes).to(logits.dtype)
    #     return ((logits - y_true) ** 2).mean()


def sparsity_penalty(model: DeiTWithGatedBoundaries) -> torch.Tensor:
    """
    Omega(alpha) = sum_l |alpha_l|      (the L1 norm of the gate vector)

    The absolute value is what makes this a SPARSITY penalty rather than just a
    magnitude penalty: |.| has a constant-magnitude gradient regardless of how
    small alpha_l already is, so gates whose task/privacy pull is weaker than
    lambda get driven to (numerically) zero and stay there, while a few strong
    ones survive. sum(alpha_l) without the abs would be a different (and broken)
    objective -- it would reward driving gates NEGATIVE without bound.
    """
    return torch.stack([a.abs() for a in model.alphas.values()]).sum()


def privacy_reward(model: DeiTWithGatedBoundaries, *args, **kwargs) -> torch.Tensor:
    """
    NOT IMPLEMENTED -- placeholder for the third term of the slide-2 objective:

        R_privacy(alpha) = sum_l alpha_l * r_l

    which enters the loss as  ` - mu * R_privacy `.

    To implement it you need a cheap, differentiable per-layer leakage score
    r_l. Candidates discussed:
      * -dCor(B_l(Z_l), x)  -- distance correlation between the block output
        and the raw input image (this is what NoPeek minimises);
      * the reconstruction error of a small inverse decoder trained jointly to
        map B_l(Z_l) back to pixels.
    A full DRAG run is far too slow to use here (~10+ min/image vs. this term
    being needed every mini-batch).

    Returns 0.0 so the objective silently reduces to task + sparsity.
    """
    return torch.zeros((), device=next(model.parameters()).device)


# -----------------------------------------------------------------------------
# Setup helpers
# -----------------------------------------------------------------------------

def load_baseline_checkpoint(model: nn.Module, cfg: Dict[str, Any], device) -> None:
    """
    Load a trained baseline into the supernet's backbone.

    The baseline .pt contains only `vit.*` keys; `boundaries.*` and `alphas.*`
    do not exist in it, so they stay freshly initialised. strict=False allows
    exactly that, and the printed `missing` list is expected to be all of the
    boundary/gate parameters -- that is not an error.
    """
    ckpt_path = cfg.get("checkpoint", {}).get("load", None)
    if not ckpt_path:
        print("[Info] No checkpoint.load given; backbone stays at init.")
        return

    print(f"[Info] Loading baseline checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    n_gate_missing = sum(1 for k in missing if k.startswith(("boundaries.", "alphas.")))
    print(
        f"[Info] Missing keys: {len(missing)} "
        f"({n_gate_missing} are boundary/gate params -- expected)"
    )
    print(f"[Info] Unexpected keys: {len(unexpected)}")
    if unexpected:
        print("  Unexpected examples:", unexpected[:5])


def apply_gated_freeze_policy(model: DeiTWithGatedBoundaries, cfg: Dict[str, Any]) -> None:
    """
    Which parameters train during the search.

    Default (freeze_backbone: true): freeze the pretrained ViT, train only the
    candidate boundary blocks, the gates, the final norm and the head. This
    mirrors the fixed-placement recipe's spirit -- but note the fixed recipe's
    "unfreeze the tail AFTER the boundary" rule cannot apply here, because with
    a candidate block after every layer there is no single "after".
    """
    train_cfg = cfg["train"]
    freeze_backbone = train_cfg.get("freeze_backbone", True)
    unfreeze_head = train_cfg.get("unfreeze_head", True)

    if freeze_backbone:
        for p in model.vit.parameters():
            p.requires_grad = False
        for p in model.vit.norm.parameters():
            p.requires_grad = True
        if unfreeze_head:
            for p in model.vit.head.parameters():
                p.requires_grad = True
    else:
        for p in model.vit.parameters():
            p.requires_grad = True

    # The blocks being searched over and their gates always train.
    for p in model.boundaries.parameters():
        p.requires_grad = True
    for p in model.alphas.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Info] Trainable params: {trainable/1e6:.3f}M / {total/1e6:.3f}M")


def build_gated_optimizer(model: DeiTWithGatedBoundaries, cfg: Dict[str, Any]):
    """
    Two parameter groups: ordinary weights, and the gates.

    Why the gates get their own group:
      * alpha_lr: gates are ~12 scalars that need to move a long way (0.1 -> 0
        or 0.1 -> ~1) in few epochs, while the weights want small careful steps.
        A single lr for both makes the gates crawl. DARTS-style searches
        routinely use a larger lr for architecture params.
      * alpha_weight_decay: MUST default to 0. AdamW's weight_decay is an L2
        pull toward zero -- applying it to the gates would silently add a SECOND
        sparsity penalty on top of your explicit lambda*L1, contaminating the
        lambda sweep you are trying to run.
    """
    opt_cfg = cfg["optimizer"]
    name = opt_cfg.get("name", "adamw").lower()
    lr = float(opt_cfg.get("lr", 1e-4))
    wd = float(opt_cfg.get("weight_decay", 0.05))
    alpha_lr = float(opt_cfg.get("alpha_lr", lr))
    alpha_wd = float(opt_cfg.get("alpha_weight_decay", 0.0))

    alpha_params = [p for p in model.alphas.parameters() if p.requires_grad]
    alpha_ids = {id(p) for p in alpha_params}
    weight_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in alpha_ids
    ]

    groups = [
        {"params": weight_params, "lr": lr, "weight_decay": wd},
        {"params": alpha_params, "lr": alpha_lr, "weight_decay": alpha_wd},
    ]

    print(
        f"[Info] Optimizer {name}: weights lr={lr} wd={wd} | "
        f"gates lr={alpha_lr} wd={alpha_wd} ({len(alpha_params)} gates)"
    )

    if name == "adamw":
        return optim.AdamW(groups)
    if name == "sgd":
        return optim.SGD(groups, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------

def train_one_epoch_gated(
    model: DeiTWithGatedBoundaries,
    loader,
    optimizer,
    device,
    epoch: int,
    cfg: Dict[str, Any],
    num_classes: int,
) -> Tuple[float, float, float, float]:
    model.train()

    total_meter = AverageMeter()
    task_meter = AverageMeter()
    sparse_meter = AverageMeter()
    acc_meter = AverageMeter()

    use_amp = cfg["train"].get("amp", True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    lam = float(cfg["search"]["lambda_sparsity"])
    task_mode = cfg["search"].get("task_loss", "cross_entropy")

    pbar = tqdm(loader, desc=f"Search epoch {epoch}", ncols=110)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            t_loss = compute_task_loss(logits, targets, num_classes, task_mode)

        # Computed OUTSIDE autocast: the gates are fp32 params and this term is
        # tiny + numerically sensitive; no reason to do it in fp16.
        s_pen = sparsity_penalty(model)

        # L(theta, alpha) = L_task + lambda * sum_l |alpha_l|
        # (+ the privacy reward term once privacy_reward() is implemented:
        #      loss = loss - mu * privacy_reward(model, ...) )
        loss = t_loss.float() + lam * s_pen

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
        acc = accuracy_top1(logits.detach().float(), targets)
        total_meter.update(loss.item(), bs)
        task_meter.update(t_loss.item(), bs)
        sparse_meter.update(s_pen.item(), bs)
        acc_meter.update(acc, bs)

        pbar.set_postfix(
            loss=f"{total_meter.avg:.4f}",
            task=f"{task_meter.avg:.4f}",
            l1=f"{sparse_meter.avg:.3f}",
            acc=f"{acc_meter.avg*100:.2f}%",
        )

    return total_meter.avg, task_meter.avg, sparse_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate_gated(
    model: DeiTWithGatedBoundaries,
    loader,
    device,
    cfg: Dict[str, Any],
    num_classes: int,
) -> Tuple[float, float]:
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    use_amp = cfg["train"].get("amp", True)
    task_mode = cfg["search"].get("task_loss", "cross_entropy")

    for images, targets in tqdm(loader, desc="Eval", ncols=110):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = compute_task_loss(logits, targets, num_classes, task_mode)

        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(accuracy_top1(logits.float(), targets), bs)

    return loss_meter.avg, acc_meter.avg


# -----------------------------------------------------------------------------
# Step 4: selection
# -----------------------------------------------------------------------------

def select_top_k(model: DeiTWithGatedBoundaries, k_max: int):
    """
    Step 4 of the algorithm: keep the k_max layers with the largest gates.

    Ranked by |alpha_l|, not alpha_l: a negative gate still means the block is
    contributing (with a flipped sign), so magnitude -- not signed value -- is
    what measures "how active is this block".
    """
    ranked = sorted(
        ((l, abs(a)) for l, a in model.alpha_dict().items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    selected = [l for l, _ in ranked[:k_max]]
    return selected, ranked


def format_alpha_table(model: DeiTWithGatedBoundaries) -> str:
    d = model.alpha_dict()
    return " | ".join(f"L{l}:{d[l]:+.4f}" for l in sorted(d))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--lambda-sparsity",
        type=float,
        default=None,
        help="Override search.lambda_sparsity without editing the YAML.",
    )
    parser.add_argument(
        "--task-loss",
        type=str,
        default=None,
        choices=["cross_entropy", "mse"],
        help="Override search.task_loss ('mse' is the Y - Y' variant).",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg.setdefault("search", {})
    if args.lambda_sparsity is not None:
        cfg["search"]["lambda_sparsity"] = args.lambda_sparsity
    if args.task_loss is not None:
        cfg["search"]["task_loss"] = args.task_loss

    set_seed(cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")
    print(
        f"[Info] lambda_sparsity={cfg['search']['lambda_sparsity']} | "
        f"task_loss={cfg['search'].get('task_loss', 'cross_entropy')}"
    )

    train_loader, test_loader, num_classes = build_dataloaders(cfg)

    search_cfg = cfg["search"]
    model = DeiTWithGatedBoundaries(
        model_name=cfg["model"].get("name", "deit_tiny_patch16_224"),
        num_classes=num_classes,
        candidate_layers=search_cfg["candidate_layers"],
        boundary_cfg=cfg["boundary"],
        pretrained=cfg["model"].get("pretrained", False),
        alpha_init=float(search_cfg.get("alpha_init", 0.1)),
    )
    model.to(device)

    load_baseline_checkpoint(model, cfg, device)
    apply_gated_freeze_policy(model, cfg)

    optimizer = build_gated_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    epochs = cfg["train"].get("epochs", 20)
    k_max = int(search_cfg.get("k_max", 2))
    best_acc = 0.0
    history = []

    print(f"[Info] alphas @ init: {format_alpha_table(model)}")

    for epoch in range(1, epochs + 1):
        tot, task, l1, tr_acc = train_one_epoch_gated(
            model, train_loader, optimizer, device, epoch, cfg, num_classes
        )
        val_loss, val_acc = evaluate_gated(model, test_loader, device, cfg, num_classes)

        if scheduler is not None:
            scheduler.step()

        print(
            f"[Epoch {epoch}] loss={tot:.4f} (task={task:.4f}, L1={l1:.3f}) "
            f"train_acc={tr_acc*100:.2f}% val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%"
        )
        print(f"           alphas: {format_alpha_table(model)}")

        history.append(
            {
                "epoch": epoch,
                "total_loss": tot,
                "task_loss": task,
                "l1": l1,
                "train_acc": tr_acc,
                "val_acc": val_acc,
                "alphas": model.alpha_dict(),
            }
        )

        if val_acc > best_acc:
            best_acc = val_acc
            if cfg.get("checkpoint", {}).get("save_best", True):
                out_dir = cfg["checkpoint"].get("out_dir", "./checkpoints")
                ensure_dir(out_dir)
                name = cfg["checkpoint"].get("name", "gated_search")
                path = os.path.join(out_dir, f"{name}_epoch{epoch}_acc{best_acc:.4f}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "best_acc": best_acc,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "cfg": cfg,
                        "alphas": model.alpha_dict(),
                    },
                    path,
                )
                print(f"[Info] Saved checkpoint to {path}")

    # ---- Step 4: report the selection ---------------------------------------
    selected, ranked = select_top_k(model, k_max)

    print("\n" + "=" * 68)
    print(f"[Result] Best val acc: {best_acc*100:.2f}%")
    print(f"[Result] Final gate ranking (by |alpha|):")
    for l, mag in ranked:
        mark = " <-- SELECTED" if l in selected else ""
        print(f"           layer {l:>2}:  |alpha| = {mag:.5f}{mark}")
    print(f"[Result] Selected top-{k_max} layers: {selected}")
    print("=" * 68)
    print(
        "[Note] Steps 5-6 (prune to these layers, drop the gates and the\n"
        "       residual path, then fine-tune the final irreversible network)\n"
        "       are NOT done here. Feed `selected` back into\n"
        "       train_deit_boundary_new_boundaries.py as boundary.after_block."
    )

    out_path = cfg.get("search_out", "./search_results/alpha_search.json")
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w") as f:
        json.dump(
            {
                "selected_layers": selected,
                "ranking": [{"layer": l, "abs_alpha": m} for l, m in ranked],
                "final_alphas": model.alpha_dict(),
                "best_val_acc": best_acc,
                "lambda_sparsity": float(cfg["search"]["lambda_sparsity"]),
                "task_loss": cfg["search"].get("task_loss", "cross_entropy"),
                "k_max": k_max,
                "candidate_layers": model.candidate_layers,
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"[Info] Wrote search results to {out_path}")


if __name__ == "__main__":
    main()