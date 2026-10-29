#!/usr/bin/env python3
"""
defense_training_rmle.py

defense_training.py + the privacy reward term, measured with the rMLE
white-box inversion attack (He et al., ACSAC'19, Algorithm 1).

    L(theta, alpha) = L_task
                    + lambda * sum_l |alpha_l|      sparsity
                    + mu     * sum_l alpha_l * s_l  privacy   <-- NEW

    s_l is set by `r_mode`: raw SSIM at z_out, or the same-depth marginal
    ( SSIM_raw - SSIM_zout ). See the header of rmle_privacy.py.

Per candidate layer l, ONE traversal captures both z_l (no block) and B_l(z_l)
(block applied); rMLE is run against each and the SSIM difference is stored.
`marginal` mode sums those differences, weighted by the gates, into the privacy
term. Because both sides come from the same traversal they share one upstream
noise realisation, so the difference isolates the block at l rather than
upstream randomness.

SCOPE: THIS IS A LOCATION SEARCH ONLY
-------------------------------------
Only the alpha gates are trainable (see freeze_all_but_alphas). The transformer
blocks, the final ViT norm, the classifier head, and every boundary's internals
(down/up projections, LayerNorms, adapter MLP, gamma) are all frozen. Each
candidate keeps its own independently initialised boundary -- they are not
synchronised to a common template.

The output is a RANKING OF LAYERS. This script deliberately reports no
deployment accuracy: the supernet runs a boundary at every candidate at once,
so no subset of it is a network you would ship, and scoring a pruned subset
measures a train/eval mismatch rather than the defense. For accuracy and a
deployable checkpoint, feed the selected layers to train_deit_boundary_multi.py.

WHY THIS IS A SEPARATE FILE
---------------------------
It imports defense_training.py and monkeypatches methods onto its model class.
Nothing in your existing files is edited, so defense_training.py stays runnable
as the no-privacy control and you can compare rankings.

Set mu_privacy: 0.0 to search on task + sparsity only.

USAGE
-----
    # time one measurement before committing to a full run:
    python defense_training_rmle.py --config configs/generated/deit_tiny_cifar100_rmle.yaml --benchmark

    # full run
    python defense_training_rmle.py --config configs/generated/deit_tiny_cifar100_rmle.yaml

    # sweep the privacy weight
    python defense_training_rmle.py --config <cfg>.yaml --mu-privacy 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Optional

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
    evaluate_gated,
    format_alpha_table,
    load_baseline_checkpoint,
    select_top_k,
    sparsity_penalty,
)
from rmle_privacy import measure_scores, privacy_reward_term


# -----------------------------------------------------------------------------
# One shared traversal that captures every exposure point at once
# -----------------------------------------------------------------------------

def forward_collect(self, x: torch.Tensor, stop_at: Optional[int] = None) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    ONE traversal of the network that captures, at EVERY candidate layer, both
    the pre-boundary activation and the boundary output:

        returns { l: {"raw": z_l, "zout": B_l(z_l), "u": u_l}, ... }

    WHY ONE PASS (this replaces forward_to_raw / forward_to_zout / forward_to_u)
    --------------------------------------------------------------------------
    Those took a `target_layer` and each re-ran every preceding block. Two
    problems, both fixed here:

    1. CORRECTNESS. The boundaries draw fresh Gaussian noise on every call
       (`u + sigma * torch.randn_like(u)`), so two separate traversals see
       DIFFERENT upstream noise realisations. Calling the same function twice
       already returned different tensors. The raw-vs-zout difference at layer
       l therefore mixed in upstream noise variance instead of isolating the
       block at l. Here both are read off ONE realisation, so everything
       upstream is bit-identical by construction and the difference isolates
       exactly the block at l.

    2. COST. Scoring |C| candidates needed 2|C| partial traversals; now it is
       one. Note also `x = alpha * z_out` below REUSES the z_out already
       computed rather than invoking the boundary a second time -- the old
       `self._gated(l, x)` call was a third redundant (and separately noised)
       evaluation of the same block.

    stop_at: stop as soon as that candidate layer has been captured. Used by
    the attack loop so a per-layer client map costs no more than it used to.
    """
    x = self.vit.patch_embed(x)
    x = self._pos_embed(x)
    out: Dict[int, Dict[str, torch.Tensor]] = {}

    if 0 in self.candidate_layers:
        raw = x
        z_out, u = self.boundaries["0"](x)
        out[0] = {"raw": raw, "zout": z_out, "u": u}
        if stop_at == 0:
            return out
        x = self.alphas["0"] * z_out          # reuse z_out; no second call

    for i, blk in enumerate(self.vit.blocks):
        x = blk(x)
        l = i + 1
        if l in self.candidate_layers:
            raw = x                            # BEFORE the boundary
            z_out, u = self.boundaries[str(l)](x)
            out[l] = {"raw": raw, "zout": z_out, "u": u}
            if stop_at is not None and l == stop_at:
                return out
            x = self.alphas[str(l)] * z_out    # reuse z_out; no second call

    if stop_at is not None and stop_at not in out:
        raise ValueError(
            f"forward_collect: stop_at={stop_at} is not in candidate_layers "
            f"{self.candidate_layers}"
        )
    return out


DeiTWithGatedBoundaries.forward_collect = forward_collect


# Thin wrappers, kept so nothing else has to change. Each is one collection
# pass stopped at the layer of interest.
#   "raw"  = z_l, the plain activation      -> z_out WITHOUT the defense layer
#   "zout" = B_l(z_l), the boundary output  -> z_out WITH the defense layer
#   "u"    = the exposed bottleneck         -> what actually crosses the wire
# ALPHA IS NOT APPLIED to any of them: the deployed defense has no gate, and
# scaling f by alpha would scale rMLE's ||f(x)-f(x0)||^2 term by alpha^2 while
# leaving lambda_TV*TV(x) alone, so the score would move because the GATE moved.

def forward_to_raw(self, x, target_layer):
    return self.forward_collect(x, stop_at=target_layer)[target_layer]["raw"]


def forward_to_zout(self, x, target_layer):
    return self.forward_collect(x, stop_at=target_layer)[target_layer]["zout"]


def forward_to_u(self, x, target_layer):
    return self.forward_collect(x, stop_at=target_layer)[target_layer]["u"]


DeiTWithGatedBoundaries.forward_to_raw = forward_to_raw
DeiTWithGatedBoundaries.forward_to_zout = forward_to_zout
DeiTWithGatedBoundaries.forward_to_u = forward_to_u


# -----------------------------------------------------------------------------
# Alpha-only freeze: this is a LOCATION search, nothing else
# -----------------------------------------------------------------------------

def freeze_all_but_alphas(model: DeiTWithGatedBoundaries) -> None:
    """
    Freeze every parameter in the model, then unfreeze ONLY the location gates.

    WHY
    ---
    The question this script answers is "where should a boundary go", not "how
    good can a boundary at each location become". If the boundary weights were
    also trainable, each location would tune its own defense and the ranking
    would confound placement quality with how well that particular copy trained.
    Freezing them means every location is judged with its boundary held fixed,
    so the alphas move purely in response to WHERE the block sits.

    WHAT IS FROZEN
    --------------
    Everything: the 12 transformer blocks, the final ViT norm, the classifier
    head, and inside each boundary the down/up projections, LayerNorms, the
    adapter MLP, and gamma. Note this is stricter than the shared
    apply_gated_freeze_policy, which trains the boundaries, vit.norm and
    vit.head as well.

    WHAT IS TRAINED
    ---------------
    model.alphas -- one scalar per candidate layer. That is all.

    Boundaries are NOT synchronised to a common template: each location keeps
    its own independently initialised weights, frozen where they started.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.alphas.parameters():
        parameter.requires_grad = True

    trainable_names = sorted(
        name for name, p in model.named_parameters() if p.requires_grad
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Info] Alpha-only search: {n_trainable} trainable / {n_total/1e6:.3f}M total")
    print(f"[Info] Trainable tensors: {trainable_names}")


# -----------------------------------------------------------------------------
# Training epoch, with the third term
# -----------------------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch: int,
    cfg: Dict[str, Any],
    num_classes: int,
    ssim_cache: Optional[Dict[int, float]] = None,
):
    model.train()

    tot_m, task_m, l1_m, priv_m, acc_m = (
        AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    )

    use_amp = cfg["train"].get("amp", True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    sc = cfg["search"]
    lam = float(sc["lambda_sparsity"])
    mu = float(sc.get("mu_privacy", 0.0))
    task_mode = sc.get("task_loss", "cross_entropy")

    pbar = tqdm(loader, desc=f"Search epoch {epoch}", ncols=120)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            t_loss = compute_task_loss(logits, targets, num_classes, task_mode)

        # Both extra terms are fp32, outside autocast: they are small, gate-only,
        # and numerically sensitive.
        s_pen = sparsity_penalty(model)
        p_term = (
            privacy_reward_term(model, ssim_cache)
            if (mu != 0.0 and ssim_cache)
            else torch.zeros((), device=device)
        )

        # L = L_task + lambda*sum_l |alpha_l| + mu*sum_l alpha_l*s_l
        #
        # THE SIGN LIVES IN s_l, NOT HERE. The three r_modes need opposite
        # signs, so hard-coding one at this line would break the others.
        # rmle_privacy.py sets s_l, and this line expands to:
        #
        #   r_mode=raw           +mu * sum_l alpha_l * SSIM_zout(l)
        #   r_mode=marginal      -mu * sum_l alpha_l * ( SSIM_raw(l) - SSIM_zout(l) )
        #                        ^ the minus, for the default mode
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
            l1=f"{l1_m.avg:.3f}", priv=f"{priv_m.avg:.3f}",
            acc=f"{acc_m.avg*100:.2f}%",
        )

    return tot_m.avg, task_m.avg, l1_m.avg, priv_m.avg, acc_m.avg


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--lambda-sparsity", type=float, default=None)
    ap.add_argument("--mu-privacy", type=float, default=None,
                    help="Weight on sum_l alpha_l*SSIM_l. 0 disables the attack entirely.")
    ap.add_argument("--task-loss", type=str, default=None,
                    choices=["cross_entropy", "mse"])
    ap.add_argument("--benchmark", action="store_true",
                    help="Time ONE rMLE measurement and exit. Run this first.")
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
    rmle_iters = int(sc.get("rmle_iters", 500))
    rmle_lr = float(sc.get("rmle_lr", 1e-2))
    rmle_tv = float(sc.get("rmle_lambda_tv", 1e1))
    probe_bs = int(sc.get("probe_batch", 8))
    r_mode = str(sc.get("r_mode", "raw"))
    attack_kind = str(sc.get("attack_kind", "zout"))
    refresh_every = int(sc.get("refresh_every", 5))

    print(f"[Info] Device: {device}")
    print(f"[Info] gating form: Z_out = alpha_l * B_l(Z)")
    print(f"[Info] L = L_task + {sc['lambda_sparsity']}*sum|alpha| "
          f"+ {mu}*sum alpha_l*SSIM_l")
    if mu == 0.0:
        print("[Info] mu_privacy = 0 -> privacy term OFF (this is the control run).")
    else:
        print(f"[Info] rMLE: {rmle_iters} iters, lr={rmle_lr}, lambda_TV={rmle_tv}, "
              f"probe batch={probe_bs}, refresh every {refresh_every} epoch(s)")
        print(f"[Info] r_mode={r_mode} | attack_kind={attack_kind}")

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

    # A FIXED probe batch: the same images every refresh, so SSIM_l changes
    # reflect the model changing, not the images changing.
    probe_images = None
    if mu != 0.0 or args.benchmark:
        probe_images = next(iter(test_loader))[0][:probe_bs].to(device)
        print(f"[Info] probe batch: {tuple(probe_images.shape)}")

    # ---- benchmark mode -----------------------------------------------------
    if args.benchmark:
        print("\n[Benchmark] one rMLE measurement across all candidate layers...")
        t0 = time.time()
        s, diag = measure_scores(model, probe_images, device, r_mode=r_mode,
                                 attack_kind=attack_kind, n_iters=rmle_iters,
                                 lr=rmle_lr, lambda_tv=rmle_tv)
        dt = time.time() - t0
        n_ref = 1 + (cfg["train"].get("epochs", 20) - 1) // refresh_every
        print(f"\n[Benchmark] {dt:.1f}s for {len(model.candidate_layers)} layers "
              f"x {rmle_iters} iters")
        print(f"[Benchmark] {n_ref} refreshes over the run -> "
              f"~{dt*n_ref/60:.1f} min of attack time total")
        print(f"[Benchmark] scores s_l: " +
              ", ".join(f"L{l}={v:+.4f}" for l, v in sorted(s.items())))
        print(f"[Benchmark] raw measurements:")
        for k, v in sorted(diag["points"].items()):
            if "ssim" in v:
                print(f"              {k:<12} SSIM={v['ssim']:.4f}  PSNR={v['psnr']:.2f}dB")
        print("\n[Benchmark] Sanity checks before trusting these numbers:")
        print("  * SSIM should DECREASE with depth. If it is flat, lambda_TV is")
        print("    probably wrong for ViT tokens -- sweep rmle_lambda_tv over")
        print("    1e0/1e1/1e2/1e3 (the paper needed 1e1 at ReLU22 but 1e3 at ReLU32).")
        print("  * If every SSIM is ~0, the attack is failing outright rather than")
        print("    being defeated; raise rmle_iters or rmle_lr and re-check.")
        return

    epochs = cfg["train"].get("epochs", 20)
    k_max = int(sc.get("k_max", 2))
    best_acc, history, ssim_cache, last_diag = 0.0, [], {}, {}
    attack_time = 0.0

    print(f"[Info] alphas @ init: {format_alpha_table(model)}")

    for epoch in range(1, epochs + 1):
        # ---- refresh r_l ----------------------------------------------------
        if mu != 0.0 and (epoch - 1) % refresh_every == 0:
            print(f"\n[rMLE] measuring privacy scores (epoch {epoch})...")
            t0 = time.time()
            ssim_cache, last_diag = measure_scores(
                model, probe_images, device, r_mode=r_mode,
                attack_kind=attack_kind, n_iters=rmle_iters, lr=rmle_lr, lambda_tv=rmle_tv,
            )
            attack_time += time.time() - t0
            print(f"[rMLE] done in {time.time()-t0:.0f}s "
                  f"(cumulative {attack_time/60:.1f} min)")

        tot, task, l1, priv, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, cfg, num_classes, ssim_cache
        )
        val_loss, val_acc = evaluate_gated(model, test_loader, device, cfg, num_classes)
        if scheduler is not None:
            scheduler.step()

        print(f"[Epoch {epoch}] loss={tot:.4f} (task={task:.4f}, L1={l1:.3f}, "
              f"priv={priv:.3f}) train_acc={tr_acc*100:.2f}% "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")
        print(f"           alphas: {format_alpha_table(model)}")
        if ssim_cache:
            print(f"           s_l:    " +
                  " | ".join(f"L{l}:{v:+.4f}" for l, v in sorted(ssim_cache.items())))

        history.append({
            "epoch": epoch, "total_loss": tot, "task_loss": task, "l1": l1,
            "privacy_term": priv, "train_acc": tr_acc, "val_acc": val_acc,
            "alphas": model.alpha_dict(), "scores": dict(ssim_cache),
        })

        if val_acc > best_acc:
            best_acc = val_acc
            if cfg.get("checkpoint", {}).get("save_best", True):
                out_dir = cfg["checkpoint"].get("out_dir", "./checkpoints")
                ensure_dir(out_dir)
                name = cfg["checkpoint"].get("name", "gated_rmle")
                path = os.path.join(out_dir, f"{name}_epoch{epoch}_acc{best_acc:.4f}.pt")
                torch.save({"epoch": epoch, "best_acc": best_acc,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(), "cfg": cfg,
                            "alphas": model.alpha_dict(),
                            "scores": dict(ssim_cache)}, path)
                print(f"[Info] Saved checkpoint to {path}")

    selected, ranked = select_top_k(model, k_max)

    print("\n" + "=" * 72)
    print("[LOCATION SEARCH RESULT]")
    print("=" * 72)
    print(f"  candidates     : {model.candidate_layers}")
    print(f"  k_max          : {k_max}")
    print(f"  lambda_sparsity: {sc['lambda_sparsity']}    mu_privacy: {mu}")
    print(f"  attack overhead: {attack_time/60:.1f} min")
    print()
    print(f"  {'layer':>7} {'|alpha|':>10} {'s_l':>10}   ")
    print(f"  {'-'*7} {'-'*10} {'-'*10}")
    for l, mag in ranked:
        sc_l = ssim_cache.get(l)
        s_str = f"{sc_l:+.4f}" if sc_l is not None else "     --"
        mark = "  <-- SELECTED" if l in selected else ""
        print(f"  {l:>7} {mag:>10.5f} {s_str:>10}{mark}")
    print()
    print(f"  SELECTED LAYERS: {selected}")
    print("=" * 72)
    print("[NEXT STEP] This script outputs LOCATIONS ONLY -- it deliberately does\n"
          "  not report a deployment accuracy. The supernet runs a boundary at every\n"
          "  candidate simultaneously, so no subset of it is the network you would\n"
          "  ship, and evaluating a pruned subset measures a train/eval mismatch\n"
          "  rather than the defense.\n"
          "\n"
          "  For accuracy AND a deployable checkpoint, hand these layers to the\n"
          "  dedicated trainer, which places real ungated boundaries and trains the\n"
          "  network around exactly that configuration:\n"
          "\n"
          f"      python train_deit_boundary_multi.py --config <cfg>.yaml "
          f"--after-blocks {' '.join(str(l) for l in selected)}\n"
          "\n"
          "  Then run DRAG against the checkpoint that produces.\n"
          "\n"
          "  Caveat on s_l: it came from rMLE, a weaker attack than DRAG. A layer\n"
          "  that resists rMLE has not been shown to resist a diffusion-prior attack.")

    out_path = cfg.get("search_out", "./search_results/alpha_rmle_search.json")
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w") as f:
        json.dump({
            "gating_form": "alpha * B(Z)",
            "privacy_term": "mu * sum_l alpha_l * SSIM_l  (rMLE, He et al. ACSAC'19)",
            "selected_layers": selected,
            "ranking": [{"layer": l, "abs_alpha": m} for l, m in ranked],
            "final_alphas": model.alpha_dict(),
            "final_scores": dict(ssim_cache),
            "final_diagnostics": last_diag,
            "supernet_best_val_acc": best_acc,
            "_note_on_accuracy": "supernet accuracy only -- NOT a deployment number. Get accuracy from train_deit_boundary_multi.py using selected_layers.",
            "next_step_cmd": "python train_deit_boundary_multi.py --config <cfg>.yaml --after-blocks " + " ".join(str(l) for l in selected),
            "lambda_sparsity": float(sc["lambda_sparsity"]),
            "mu_privacy": mu,
            "r_mode": r_mode, "attack_kind": attack_kind,
            "rmle": {"iters": rmle_iters, "lr": rmle_lr, "lambda_tv": rmle_tv,
                     "probe_batch": probe_bs, "refresh_every": refresh_every},
            "attack_time_min": attack_time / 60.0,
            "candidate_layers": model.candidate_layers,
            "history": history,
        }, f, indent=2)
    print(f"[Info] Wrote {out_path}")


if __name__ == "__main__":
    main()