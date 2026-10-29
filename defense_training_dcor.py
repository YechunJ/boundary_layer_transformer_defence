#!/usr/bin/env python3
"""
defense_training_dcor.py

Alpha-gated placement search with a distance-correlation privacy term (NoPeek),
replacing the rMLE/SSIM attack that returned a content-free floor on ViT tokens.

    L(theta, alpha) = L_task
                    + lambda * sum_l |alpha_l|      sparsity
                    + mu     * sum_l alpha_l * s_l  privacy   (dcor_privacy.py)

        r_mode='marginal'  s_l = - ( dCor2(z_l, x) - dCor2(exposed_l, x) )
        r_mode='raw'       s_l = + dCor2(exposed_l, x)

WHAT DIFFERS FROM defense_training_rmle.py
------------------------------------------
1. NO SEPARATE MEASUREMENT PASS. The training forward already computes the
   exposures at every candidate layer; the privacy term reads them directly.
   One forward pass per batch does both task loss and privacy term.
2. NO CACHING. rMLE cost ~500 forward/backward per point, so scores were
   refreshed every N epochs and reused stale between. dCor is one closed-form
   expression, recomputed on EVERY batch. rmle_iters/lr/lambda_tv/refresh_every
   are all gone.
3. THE GRADIENT REACHES THE BOUNDARY WEIGHTS. rMLE's SSIM was a detached
   constant, so only the gates felt it. dCor is differentiable end to end, so
   with dcor_detach: false the blocks themselves learn to decorrelate their
   output from the input -- NoPeek's Eq. 1.

This is a LOCATION SEARCH: only the alpha gates are trainable
(freeze_all_but_alphas). It reports a ranking of layers, not a deployment
accuracy -- see the note in the results section. Feed the selected layers to
train_deit_boundary_multi.py for a deployable checkpoint and honest accuracy.

Set mu_privacy: 0.0 to search on task + sparsity only.

USAGE
-----
    python defense_training_dcor.py --config configs/generated/deit_tiny_cifar100_dcor.yaml --probe
    python defense_training_dcor.py --config configs/generated/deit_tiny_cifar100_dcor.yaml
    python defense_training_dcor.py --config <cfg>.yaml --mu-privacy 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from train_deit_boundary_new_boundaries import (
    AverageMeter,
    accuracy_top1,
    build_dataloaders,
    build_scheduler,
    ensure_dir,
    load_yaml,
    set_seed,
)
from defense_training import (
    DeiTWithGatedBoundaries,
    build_gated_optimizer,
    compute_task_loss,
    format_alpha_table,
    load_baseline_checkpoint,
    select_top_k,
    sparsity_penalty,
)
from dcor_privacy import privacy_term, format_dcor_table


# -----------------------------------------------------------------------------
# Alpha-only freeze: this is a LOCATION search
# -----------------------------------------------------------------------------

def freeze_all_but_alphas(model: DeiTWithGatedBoundaries) -> None:
    """
    Freeze every parameter, then unfreeze ONLY the gates. The ranking then
    reflects WHERE a boundary should go, not how well each location's boundary
    could be trained. Note: with dcor_detach: false the privacy term's gradient
    would normally reach the boundary weights (NoPeek Eq. 1) -- but here those
    weights are frozen, so in a pure location search dcor_detach is effectively
    forced. To let the blocks learn to decorrelate, unfreeze the boundaries too
    (edit below) and set dcor_detach: false; that becomes a joint search+train,
    not a pure placement search.
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.alphas.parameters():
        p.requires_grad = True

    names = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[Info] Alpha-only search: {n_tr} trainable / {n_all/1e6:.3f}M total")
    print(f"[Info] Trainable tensors: {names}")


# -----------------------------------------------------------------------------
# One forward pass that yields logits AND every exposure point
# -----------------------------------------------------------------------------

def forward_with_exposures(
    self, x: torch.Tensor
) -> Tuple[torch.Tensor, Dict[int, Dict[str, torch.Tensor]]]:
    """
    Ordinary training forward, additionally returning per candidate layer:
        raw  = z_l          activation BEFORE the boundary   (no block)
        zout = B_l(z_l)     boundary output                  (block applied)
        u    = the exposed bottleneck                        (what is sent)
    All captured from ONE traversal, gradients preserved. `x = alpha_l * z_out`
    reuses the z_out already computed rather than re-calling the boundary.
    """
    x = self.vit.patch_embed(x)
    x = self._pos_embed(x)
    exposures: Dict[int, Dict[str, torch.Tensor]] = {}

    if 0 in self.candidate_layers:
        raw = x
        z_out, u = self.boundaries["0"](x)
        exposures[0] = {"raw": raw, "zout": z_out, "u": u}
        x = self.alphas["0"] * z_out

    for i, blk in enumerate(self.vit.blocks):
        x = blk(x)
        l = i + 1
        if l in self.candidate_layers:
            raw = x
            z_out, u = self.boundaries[str(l)](x)
            exposures[l] = {"raw": raw, "zout": z_out, "u": u}
            x = self.alphas[str(l)] * z_out

    x = self.vit.norm(x)
    if getattr(self.vit, "dist_token", None) is not None:
        feat = (x[:, 0] + x[:, 1]) / 2
    else:
        feat = x[:, 0]
    return self.vit.head(feat), exposures


DeiTWithGatedBoundaries.forward_with_exposures = forward_with_exposures


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, epoch, cfg, num_classes):
    model.train()
    tot_m, task_m, l1_m, priv_m, acc_m = (
        AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    )

    sc = cfg["search"]
    lam = float(sc["lambda_sparsity"])
    mu = float(sc.get("mu_privacy", 0.0))
    task_mode = sc.get("task_loss", "cross_entropy")
    kind = str(sc.get("dcor_kind", "u"))
    r_mode = str(sc.get("r_mode", "marginal"))
    detach = bool(sc.get("dcor_detach", False))
    estimator = str(sc.get("dcor_estimator", "u"))

    use_amp = cfg["train"].get("amp", True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    last_diag: Dict[int, Dict[str, float]] = {}

    pbar = tqdm(loader, desc=f"Search epoch {epoch}", ncols=120)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, exposures = model.forward_with_exposures(images)
            t_loss = compute_task_loss(logits, targets, num_classes, task_mode)

        # Privacy and sparsity in fp32, OUTSIDE autocast. For dCor this is not
        # optional: the sum of squares over 150,528 image dims overflows fp16.
        s_pen = sparsity_penalty(model)
        if mu != 0.0:
            p_term, last_diag = privacy_term(
                model, exposures, images, kind=kind, r_mode=r_mode,
                detach=detach, estimator=estimator
            )
        else:
            p_term = torch.zeros((), device=device)

        loss = t_loss.float() + lam * s_pen + mu * p_term

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
        l1_m.update(s_pen.item(), bs)
        priv_m.update(float(p_term.detach()), bs)
        acc_m.update(accuracy_top1(logits.detach().float(), targets), bs)
        pbar.set_postfix(
            loss=f"{tot_m.avg:.4f}", task=f"{task_m.avg:.4f}",
            l1=f"{l1_m.avg:.3f}", priv=f"{priv_m.avg:+.3f}",
            acc=f"{acc_m.avg*100:.2f}%",
        )

    return tot_m.avg, task_m.avg, l1_m.avg, priv_m.avg, acc_m.avg, last_diag


@torch.no_grad()
def evaluate(model, loader, device, cfg, num_classes):
    model.eval()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    use_amp = cfg["train"].get("amp", True)
    task_mode = cfg["search"].get("task_loss", "cross_entropy")
    for images, targets in tqdm(loader, desc="Eval", ncols=120):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, _ = model.forward_with_exposures(images)
            loss = compute_task_loss(logits, targets, num_classes, task_mode)
        bs = images.size(0)
        loss_m.update(loss.item(), bs)
        acc_m.update(accuracy_top1(logits.float(), targets), bs)
    return loss_m.avg, acc_m.avg


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--lambda-sparsity", type=float, default=None)
    ap.add_argument("--mu-privacy", type=float, default=None)
    ap.add_argument("--task-loss", type=str, default=None,
                    choices=["cross_entropy", "mse"])
    ap.add_argument("--probe", action="store_true",
                    help="Report dCor on ONE batch at init and exit. Run this first.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg.setdefault("search", {})
    sc = cfg["search"]
    if args.lambda_sparsity is not None:
        sc["lambda_sparsity"] = args.lambda_sparsity
    if args.mu_privacy is not None:
        sc["mu_privacy"] = args.mu_privacy
    if args.task_loss is not None:
        sc["task_loss"] = args.task_loss

    set_seed(cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu = float(sc.get("mu_privacy", 0.0))
    kind = str(sc.get("dcor_kind", "u"))
    r_mode = str(sc.get("r_mode", "marginal"))
    detach = bool(sc.get("dcor_detach", False))
    estimator = str(sc.get("dcor_estimator", "u"))

    print(f"[Info] Device: {device}")
    print(f"[Info] gating form: Z_out = alpha_l * B_l(Z)")
    print(f"[Info] L = L_task + {sc['lambda_sparsity']}*sum|alpha| + {mu}*sum alpha_l*s_l")
    est_name = "unbiased U-statistic" if estimator == "u" else "classical V-statistic (NoPeek App.B)"
    print(f"[Info] privacy: dCor^2 [{est_name}] | r_mode={r_mode} | kind={kind} | detach={detach}")
    if mu == 0.0:
        print("[Info] mu_privacy = 0 -> privacy term OFF (control run).")

    train_loader, test_loader, num_classes = build_dataloaders(cfg)

    model = DeiTWithGatedBoundaries(
        model_name=cfg["model"].get("name", "deit_tiny_patch16_224"),
        num_classes=num_classes,
        candidate_layers=sc["candidate_layers"],
        boundary_cfg=cfg["boundary"],
        pretrained=cfg["model"].get("pretrained", False),
        alpha_init=float(sc.get("alpha_init", 1.0)),
    ).to(device)

    load_baseline_checkpoint(model, cfg, device)
    freeze_all_but_alphas(model)
    optimizer = build_gated_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # ---- probe mode ---------------------------------------------------------
    if args.probe:
        images = next(iter(test_loader))[0].to(device)
        print(f"\n[Probe] one batch {tuple(images.shape)} at init\n")
        model.eval()
        t0 = time.time()
        with torch.no_grad():
            _, exposures = model.forward_with_exposures(images)
        _, diag = privacy_term(model, exposures, images, kind=kind,
                               r_mode=r_mode, detach=True, estimator=estimator)
        dt = time.time() - t0
        print(f"  {'layer':>6} {'dCor2(raw,x)':>14} {'dCor2(' + kind + ',x)':>14} "
              f"{'reduction':>11} {'s_l':>9}")
        for l in sorted(diag):
            d = diag[l]
            raw = d["dcor_raw"]
            blk = d[f"dcor_{kind}"]
            red = (raw - blk) if raw == raw else float("nan")
            print(f"  {l:>6} {raw:>14.4f} {blk:>14.4f} {red:>11.4f} {d['s']:>+9.4f}")
        print(f"\n[Probe] {dt:.2f}s for one batch, all layers "
              f"(rMLE needed ~30s for the same thing)")
        print("\n[Probe] What to look for:")
        print("  * dCor2(raw,x) should DECREASE with depth. NoPeek Fig. 11 notes")
        print("    the network reduces dCor on its own, so this trend is expected")
        print("    and is why r_mode=marginal (same-depth) is the default.")
        print("  * reduction > 0 means the block lowered dependence at that layer.")
        print("    At init the blocks are untrained, so small values are normal;")
        print("    what matters is that they MOVE during training.")
        print("  * If every number is ~0 or ~1 regardless of layer, something is")
        print("    wrong -- that was the failure mode of the SSIM version.")
        return

    epochs = cfg["train"].get("epochs", 20)
    k_max = int(sc.get("k_max", 2))
    best_acc, history, last_diag = 0.0, [], {}

    print(f"[Info] alphas @ init: {format_alpha_table(model)}")

    for epoch in range(1, epochs + 1):
        tot, task, l1, priv, tr_acc, last_diag = train_one_epoch(
            model, train_loader, optimizer, device, epoch, cfg, num_classes
        )
        val_loss, val_acc = evaluate(model, test_loader, device, cfg, num_classes)
        if scheduler is not None:
            scheduler.step()

        print(f"[Epoch {epoch}] loss={tot:.4f} (task={task:.4f}, L1={l1:.3f}, "
              f"priv={priv:+.3f}) train_acc={tr_acc*100:.2f}% "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")
        print(f"           alphas: {format_alpha_table(model)}")
        if last_diag:
            print(f"           dcor:   {format_dcor_table(last_diag, kind)}")

        history.append({
            "epoch": epoch, "total_loss": tot, "task_loss": task, "l1": l1,
            "privacy_term": priv, "train_acc": tr_acc, "val_acc": val_acc,
            "alphas": model.alpha_dict(),
            "dcor": {str(k): v for k, v in last_diag.items()},
        })

        if val_acc > best_acc:
            best_acc = val_acc
            if cfg.get("checkpoint", {}).get("save_best", True):
                out_dir = cfg["checkpoint"].get("out_dir", "./checkpoints")
                ensure_dir(out_dir)
                name = cfg["checkpoint"].get("name", "gated_dcor")
                path = os.path.join(out_dir, f"{name}_epoch{epoch}_acc{best_acc:.4f}.pt")
                torch.save({"epoch": epoch, "best_acc": best_acc,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(), "cfg": cfg,
                            "alphas": model.alpha_dict(),
                            "dcor": {str(k): v for k, v in last_diag.items()}}, path)
                print(f"[Info] Saved checkpoint to {path}")

    selected, ranked = select_top_k(model, k_max)

    print("\n" + "=" * 72)
    print("[LOCATION SEARCH RESULT]")
    print("=" * 72)
    print(f"  candidates     : {model.candidate_layers}")
    print(f"  k_max          : {k_max}")
    print(f"  lambda_sparsity: {sc['lambda_sparsity']}    mu_privacy: {mu}")
    print(f"  privacy signal : unbiased dCor^2 (NoPeek), r_mode={r_mode}, kind={kind}")
    print()
    print(f"  {'layer':>7} {'|alpha|':>10} {'dcor_'+kind:>11} {'s_l':>9}")
    print(f"  {'-'*7} {'-'*10} {'-'*11} {'-'*9}")
    for l, mag in ranked:
        d = last_diag.get(l, {})
        blk = d.get(f"dcor_{kind}", float("nan"))
        s = d.get("s", float("nan"))
        mark = "  <-- SELECTED" if l in selected else ""
        print(f"  {l:>7} {mag:>10.5f} {blk:>11.4f} {s:>+9.4f}{mark}")
    print()
    print(f"  SELECTED LAYERS: {selected}")
    print("=" * 72)
    print("[NEXT STEP] Locations only -- no deployment accuracy is reported here\n"
          "  (the supernet runs a boundary at every candidate at once; a pruned\n"
          "  subset is a train/eval mismatch). For accuracy and a deployable\n"
          "  checkpoint, hand these layers to the dedicated trainer:\n"
          "\n"
          f"      python train_deit_boundary_multi.py --config <cfg>.yaml "
          f"--after-blocks {' '.join(str(l) for l in selected)}\n"
          "\n"
          "  Then run DRAG against that checkpoint.")

    out_path = cfg.get("search_out", "./search_results/alpha_dcor_search.json")
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w") as f:
        json.dump({
            "gating_form": "alpha * B(Z)",
            "privacy_term": "mu * sum_l alpha_l * s_l, s_l from unbiased dCor^2 "
                            "(NoPeek ICDMW'20, Szekely-Rizzo U-statistic)",
            "selected_layers": selected,
            "ranking": [{"layer": l, "abs_alpha": m} for l, m in ranked],
            "final_alphas": model.alpha_dict(),
            "final_dcor": {str(k): v for k, v in last_diag.items()},
            "supernet_best_val_acc": best_acc,
            "_note_on_accuracy": "supernet accuracy only -- NOT a deployment "
                                 "number. Get accuracy from train_deit_boundary_multi.py.",
            "next_step_cmd": "python train_deit_boundary_multi.py --config <cfg>.yaml "
                             "--after-blocks " + " ".join(str(l) for l in selected),
            "lambda_sparsity": float(sc["lambda_sparsity"]),
            "mu_privacy": mu, "r_mode": r_mode, "dcor_kind": kind,
            "dcor_detach": detach,
            "candidate_layers": model.candidate_layers,
            "history": history,
        }, f, indent=2)
    print(f"[Info] Wrote {out_path}")


if __name__ == "__main__":
    main()