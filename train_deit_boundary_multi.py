#!/usr/bin/env python3
"""
train_deit_boundary_multi.py

Deployment trainer for MULTIPLE simultaneous boundaries.

train_deit_boundary_new_boundaries.py places exactly one boundary
(`boundary.after_block: 6`). This places several
(`boundary.after_blocks: [6, 9]`) and is otherwise the same recipe, so its
reported accuracy is directly comparable to the single-boundary table.

WHY THIS IS NOT THE SEARCH SCRIPT
---------------------------------
No gates, no alphas, no supernet. Every listed layer gets a REAL boundary that
is present on every forward pass from the first training step to the last, and
the network trains around exactly that configuration. Train and eval therefore
match, which is the whole reason the single-boundary trainer reports ~80% while
pruning a gated supernet down to a subset collapses to chance: there, the
network was trained with boundaries at positions that the pruned model no
longer has.

Use this AFTER a placement search has told you which layers to use:

    search (alpha-only)  ->  "layers 6 and 9"
    THIS SCRIPT          ->  real checkpoint + honest accuracy
    DRAG                 ->  attack that checkpoint

CONFIG
------
Same as the single-boundary config except:

    boundary:
      after_blocks: [6, 9]      # <-- list, replaces `after_block: 6`
      module: residual_adapter
      bottleneck_dim: 128
      sigma: 0.05
      quant_bits: 8
      ...

All listed layers share one boundary config. If you need per-layer settings,
see the `after_blocks` handling in build_model() -- it is one loop.

USAGE
-----
    python train_deit_boundary_multi.py --config configs/generated/deit_tiny_multi_6_9.yaml
    python train_deit_boundary_multi.py --config <cfg>.yaml --after-blocks 6 9
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import timm
import torch
import torch.nn as nn
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
# Model: N real boundaries, no gates
# -----------------------------------------------------------------------------

class DeiTWithBoundaries(nn.Module):
    """
    DeiT with a REAL boundary after each layer in `after_blocks`.

    Contrast with DeiTWithGatedBoundaries (the search model): that one has a
    ParameterDict of alphas and computes alpha_l * B_l(z). This one computes
    B_l(z) directly -- there is no gate, because a deployed defense does not
    have one.

    Layer indexing matches the single-boundary trainer:
        l = 0  -> before block 1
        l = 6  -> after the 6th transformer block
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        after_blocks: List[int],
        boundary_cfg: Dict[str, Any],
        pretrained: bool = False,
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

        num_patches = self.vit.patch_embed.num_patches
        extra_tokens = 1
        if getattr(self.vit, "dist_token", None) is not None:
            extra_tokens += 1
        self.seq_len = num_patches + extra_tokens

        self.after_blocks = sorted({int(l) for l in after_blocks})
        if not self.after_blocks:
            raise ValueError("boundary.after_blocks must not be empty")
        for l in self.after_blocks:
            if not (0 <= l <= self.num_blocks):
                raise ValueError(f"after_block {l} out of range [0, {self.num_blocks}]")

        self.boundaries = nn.ModuleDict()
        for l in self.after_blocks:
            self.boundaries[str(l)] = build_boundary_module(
                boundary_cfg=boundary_cfg,
                seq_len=self.seq_len,
                dim=self.embed_dim,
            )

        print(
            f"[Model] {model_name} | blocks={self.num_blocks} | seq_len={self.seq_len} "
            f"| boundaries after blocks {self.after_blocks} | NO gates (deployment form)"
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
        return self.vit.pos_drop(x)

    def forward_features(self, x: torch.Tensor, collect_u: bool = False):
        """If collect_u, also return {layer: u} -- the exposed bottlenecks."""
        us: Dict[int, torch.Tensor] = {}
        x = self.vit.patch_embed(x)
        x = self._pos_embed(x)

        if 0 in self.after_blocks:
            x, u = self.boundaries["0"](x)
            if collect_u:
                us[0] = u

        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            l = i + 1
            if l in self.after_blocks:
                x, u = self.boundaries[str(l)](x)   # REAL boundary, no alpha
                if collect_u:
                    us[l] = u

        x = self.vit.norm(x)
        if getattr(self.vit, "dist_token", None) is not None:
            feat = (x[:, 0] + x[:, 1]) / 2
        else:
            feat = x[:, 0]
        return (feat, us) if collect_u else feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit.head(self.forward_features(x))


# -----------------------------------------------------------------------------
# Freeze policy
# -----------------------------------------------------------------------------

def apply_multi_freeze_policy(model: DeiTWithBoundaries, cfg: Dict[str, Any]) -> None:
    """
    Generalises the single-boundary rule "freeze everything before the boundary,
    train the tail after it" to several boundaries: freeze up to the EARLIEST
    boundary, train everything from there on.

    Rationale: every block after the first boundary sees a signal that has been
    through at least one bottleneck, so it has to adapt. Blocks before the first
    boundary see untouched input and can stay frozen.

    freeze_before_first: false  -> train the whole network (slower, usually
                                   slightly better).
    """
    train_cfg = cfg["train"]
    freeze_before_first = train_cfg.get("freeze_before_first", True)
    first = min(model.after_blocks)

    if freeze_before_first:
        for p in model.vit.parameters():
            p.requires_grad = False
        # blocks strictly after the first boundary
        for i, blk in enumerate(model.vit.blocks):
            if (i + 1) > first:
                for p in blk.parameters():
                    p.requires_grad = True
        for p in model.vit.norm.parameters():
            p.requires_grad = True
        for p in model.vit.head.parameters():
            p.requires_grad = True
    else:
        for p in model.vit.parameters():
            p.requires_grad = True

    for p in model.boundaries.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    n_train_blocks = sum(
        1 for i in range(model.num_blocks) if (i + 1) > first
    ) if freeze_before_first else model.num_blocks
    print(
        f"[Info] Trainable: {trainable/1e6:.3f}M / {total/1e6:.3f}M  "
        f"(boundaries + final norm + head + {n_train_blocks} blocks after L{first})"
    )


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]):
    opt_cfg = cfg["optimizer"]
    name = opt_cfg.get("name", "adamw").lower()
    lr = float(opt_cfg.get("lr", 1e-4))
    wd = float(opt_cfg.get("weight_decay", 0.05))
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[Info] Optimizer {name}: lr={lr} weight_decay={wd}")
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {name}")


def load_baseline(model: nn.Module, cfg: Dict[str, Any], device) -> None:
    ckpt_path = cfg.get("checkpoint", {}).get("load", None)
    if not ckpt_path:
        print("[Info] No checkpoint.load; backbone starts at init.")
        return
    print(f"[Info] Loading baseline: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    n_b = sum(1 for k in missing if k.startswith("boundaries."))
    print(f"[Info] Missing {len(missing)} ({n_b} boundary params -- expected), "
          f"unexpected {len(unexpected)}")


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, epoch, cfg):
    model.train()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    use_amp = cfg["train"].get("amp", True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    pbar = tqdm(loader, desc=f"Epoch {epoch}", ncols=110)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, targets)

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
        loss_m.update(loss.item(), bs)
        acc_m.update(accuracy_top1(logits.detach().float(), targets), bs)
        pbar.set_postfix(loss=f"{loss_m.avg:.4f}", acc=f"{acc_m.avg*100:.2f}%")

    return loss_m.avg, acc_m.avg


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    use_amp = cfg["train"].get("amp", True)
    for images, targets in tqdm(loader, desc="Eval", ncols=110):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, targets)
        bs = images.size(0)
        loss_m.update(loss.item(), bs)
        acc_m.update(accuracy_top1(logits.float(), targets), bs)
    return loss_m.avg, acc_m.avg


def auto_ckpt_name(cfg: Dict[str, Any], model: "DeiTWithBoundaries") -> str:
    """
    Build a checkpoint name from the actual config, matching the existing
    convention, e.g.

        deit_tiny_cifar100_b6_b9_residual_adapter_d128_s0p050_q8_lr1em04

    so nothing about the placement or strength is baked into a filename by hand.
    """
    b = cfg["boundary"]
    ds = cfg.get("data", {}).get("dataset", "data")
    mdl = cfg.get("model", {}).get("name", "model").replace("_patch16_224", "")
    blocks = "_".join(f"b{l}" for l in model.after_blocks)
    module = b.get("module", "boundary")
    dim = b.get("bottleneck_dim", "NA")
    sig = str(b.get("sigma", 0)).replace(".", "p")
    q = b.get("quant_bits", "NA")
    lr = f"{float(cfg['optimizer'].get('lr', 1e-4)):.0e}".replace("-", "m").replace("+", "p")
    return f"{mdl}_{ds}_{blocks}_{module}_d{dim}_s{sig}_q{q}_lr{lr}"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--after-blocks", type=int, nargs="+", default=None,
                    help="Override boundary.after_blocks, e.g. --after-blocks 6 9")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.after_blocks is not None:
        cfg["boundary"]["after_blocks"] = list(args.after_blocks)

    # accept the single-boundary key too, so old configs still work
    bcfg = cfg["boundary"]
    if "after_blocks" not in bcfg:
        if "after_block" in bcfg:
            bcfg["after_blocks"] = [int(bcfg["after_block"])]
            print(f"[Info] using single after_block={bcfg['after_block']} as a list")
        else:
            raise ValueError("config needs boundary.after_blocks (list) or after_block (int)")

    set_seed(cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    train_loader, test_loader, num_classes = build_dataloaders(cfg)

    model = DeiTWithBoundaries(
        model_name=cfg["model"].get("name", "deit_tiny_patch16_224"),
        num_classes=num_classes,
        after_blocks=bcfg["after_blocks"],
        boundary_cfg=bcfg,
        pretrained=cfg["model"].get("pretrained", False),
    ).to(device)

    load_baseline(model, cfg, device)
    apply_multi_freeze_policy(model, cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    epochs = cfg["train"].get("epochs", 50)
    best_acc, history = 0.0, []

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, epoch, cfg)
        val_loss, val_acc = evaluate(model, test_loader, device, cfg)
        if scheduler is not None:
            scheduler.step()

        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} train_acc={tr_acc*100:.2f}% "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            if cfg.get("checkpoint", {}).get("save_best", True):
                out_dir = cfg["checkpoint"].get("out_dir", "./checkpoints")
                ensure_dir(out_dir)
                name = cfg["checkpoint"].get("name") or auto_ckpt_name(cfg, model)
                path = os.path.join(out_dir, f"{name}_epoch{epoch}_acc{best_acc:.4f}.pt")
                torch.save({"epoch": epoch, "best_acc": best_acc,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "cfg": cfg, "after_blocks": model.after_blocks}, path)
                print(f"[Info] Saved {path}")
    if cfg.get("checkpoint", {}).get("save_last", True):
        out_dir = cfg["checkpoint"].get("out_dir", "./checkpoints")
        ensure_dir(out_dir)
        name = cfg["checkpoint"].get("name") or auto_ckpt_name(cfg, model)
        path = os.path.join(out_dir, f"{name}_LAST_epoch{epochs}_acc{val_acc:.4f}.pt")
        torch.save({"epoch": epochs, "val_acc": val_acc, "best_acc": best_acc,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": cfg, "after_blocks": model.after_blocks}, path)
        print(f"[Info] Saved FINAL-epoch checkpoint to {path}")

    print("\n" + "=" * 68)
    print(f"[Result] boundaries at {model.after_blocks}  ->  best val acc "
          f"{best_acc*100:.2f}%")
    print("=" * 68)
    print("[Note] This number is directly comparable to the single-boundary table:\n"
          "       same recipe, boundary present throughout training and at eval,\n"
          "       no gates. Feed this checkpoint to DRAG with\n"
          "       boundary_client.split_after_block set to ONE of these layers\n"
          "       (the attacker intercepts at a single split point).")

    out_path = cfg.get("train_out") or f"./search_results/{auto_ckpt_name(cfg, model)}.json"
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w") as f:
        json.dump({"after_blocks": model.after_blocks, "best_val_acc": best_acc,
                   "boundary_cfg": bcfg, "history": history}, f, indent=2)
    print(f"[Info] Wrote {out_path}")


if __name__ == "__main__":
    main()