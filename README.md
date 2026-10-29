# Boundary-Layer Transformer Defence

A training-time defence for **split inference with Vision Transformers**. One or more noisy bottlenecks ("boundaries") are inserted inside a DeiT model, so the activation that leaves the client is compressed, power-limited, noised and quantized before the server sees it.

Experiments use **DeiT-Tiny on CIFAR-100** (images upsampled 32 → 224 to keep the pretrained 197-token positional embedding). Defended models are attacked with [DRAG](https://github.com/ntuaislab/DRAG), a diffusion-based data reconstruction attack.

## How a boundary works

```
z [B, T, D]                          (T = 197 tokens, D = 192 for DeiT-Tiny)
  → down-projection  D → D'
  → LayerNorm
  → PowerNorm        (clip per-sample RMS to ≤ √power)
  → + N(0, σ²)       (kept ON at evaluation time)
  → 8-bit fake quantization (straight-through estimator)
  = u  [B, T, D']    ← exposed representation sent to the server
  → up-projection    D' → D
  → residual MLP adapter
  → remaining transformer blocks
```

The original `z` is **never** added back after the boundary, so all downstream computation depends only on `u`. By the data-processing inequality, any leakage bound on `u` also covers everything the server computes from it. Quantizing *after* the noise can only reduce information, so bounds computed on the pre-quantization tensor are conservative.

**Threat model.** Honest-but-curious server that knows the full model (white-box), intercepts the exposed tensor at the split point, and runs an optimization-based reconstruction attack. With multiple boundaries, the deepest one is the split point; the earlier ones add defence in depth in case the split moves or an attacker observes a shallower activation.

## Boundary modules

Set with `boundary.module` in the config:

| Module | Description |
|---|---|
| `identity` | No-op, for debugging insertion |
| `linear_noise_quant` | Token-wise linear bottleneck + noise + quantization |
| `token_mix_noise_quant` | Also compresses the token dimension T → T' |
| `residual_adapter` | Linear bottleneck + MLP adapter after `u` (**used in all experiments**) |
| `cls_latent_token` | Keeps CLS separate; compresses patch tokens into learned latent tokens via cross-attention |

## Repository layout

| File | Purpose |
|---|---|
| `train_deit_boundary_new_boundaries.py` | Baseline fine-tuning and **single-boundary** training. Defines all boundary modules. |
| `train_deit_boundary_multi.py` | Training with **several boundaries at once** (deployment form, no gates) |
| `defense_training.py` | Placement search: gated supernet with an L1 penalty on gates |
| `defense_training_rmle.py` + `rmle_privacy.py` | Placement search with an rMLE inversion-attack privacy term |
| `defense_training_dcor.py` + `dcor_privacy.py` | Placement search with a distance-correlation (NoPeek) privacy term |
| `sigma_training.py` | Learns the per-layer noise level σ for fixed placements |
| `sub_job_new_boundaries.py` | Generates configs and Slurm jobs for sweeps; collects results into Excel |
| `download_cifar.py`, `download_timm_pretrained.py` | Fetch the dataset and ImageNet-pretrained weights |
| `configs/generated/` | Example YAML configs |

The attack side (DeiT boundary client and launchers for DRAG) lives in a separate repository.

## Setup

Requirements: Python 3.11, PyTorch, torchvision, `timm` (tested with 1.0.27), `pyyaml`, `tqdm`, and `openpyxl` (sweep script only).

```bash
pip install torch torchvision timm pyyaml tqdm openpyxl
```

The configs expect CIFAR in a `datasets/` folder **next to** this repository, and the pretrained weights in `./pretrained/`:

```bash
python download_cifar.py --dataset cifar100 --root ../datasets
python download_timm_pretrained.py --model deit_tiny_patch16_224 --out_dir ./pretrained
```

On clusters where compute nodes have no internet access, run these on a login node.

## Usage

**Layer indexing:** `after_block: k` places the boundary after the k-th transformer block. `0` means before the first block; DeiT-Tiny has 12 blocks.

### 1. Fine-tune the baseline (no boundary)

```bash
python train_deit_boundary_new_boundaries.py --config configs/generated/deit_tiny_cifar100_baseline.yaml
```

Saves `checkpoints/deit_tiny_cifar100_baseline_epoch<N>_acc<acc>.pt`.

### 2. Train with a single boundary

Same script with `mode: boundary`, `boundary.enabled: true` and `boundary.after_block: k`. See the `b6` / `b8` configs in `configs/generated/`.

### 3. Train with multiple boundaries

Set `checkpoint.load` to your baseline checkpoint from step 1, then:

```bash
python train_deit_boundary_multi.py --config configs/generated/deit_tiny_cifar100_multi.yaml --after-blocks 3 8 9 10
```

Checkpoint names are derived from the config. Per-epoch train/validation history is written to `search_results/<name>.json`. Set `checkpoint.save_last: true` to also keep the final-epoch checkpoint (use this one for comparisons, not the best-epoch one).

### 4. Optional: search for placements or noise levels

```bash
python defense_training.py      --config configs/generated/deit_tiny_cifar100_gated.yaml
python defense_training_rmle.py --config configs/generated/deit_tiny_cifar100_rmle.yaml
python defense_training_dcor.py --config configs/generated/deit_tiny_cifar100_dcor.yaml
python sigma_training.py        --config configs/generated/deit_tiny_cifar100_sigma.yaml
```

The placement searches output a **ranking of layers** (by |α|), not a deployable model. Retrain the selected layers with step 3 to get a real accuracy number.

### 5. Optional: Slurm sweeps

```bash
python sub_job_new_boundaries.py --create
python sub_job_new_boundaries.py --submit --template-yaml <your_template>.yaml --row-min 2 --row-max 20
python sub_job_new_boundaries.py --collect --row-min 2 --row-max 20
```

Edit the cluster settings at the top of the script (account, paths) before submitting.

## Key config fields

| Field | Meaning |
|---|---|
| `bottleneck_dim` | Exposed width D' per token |
| `sigma` | Std. dev. of the Gaussian noise added to `u` |
| `quant_bits` | Fake-quantization bit width (8 in all experiments) |
| `power`, `power_norm_mode` | RMS limit on `u` before noise (`clip`, `unit` or `none`) |
| `noise_in_eval` | Keep noise on at test time (must be `true` for the privacy claim) |
| `adapter_mlp_ratio`, `adapter_gamma_init` | Size and initial gain of the residual adapter |
| `train.freeze_before_first` | Freeze blocks before the first boundary (see caveats) |
| `search.dcor_estimator` | `u` (unbiased U-statistic, default) or `classical` (V-statistic) |

## Current results

DeiT-Tiny, CIFAR-100, `residual_adapter`, D' = 32, σ = 0.5, 8-bit. DRAG attacks `u` at the deepest boundary (block 10). Accuracy is final-epoch CIFAR-100 test accuracy.

| Placement | Top-1 acc. | DRAG SSIM ↓ | DRAG MS-SSIM ↓ |
|---|---|---|---|
| No defence (baseline) | 83.94% | — | — |
| [1, 2, 3, 10] | 48.01% | 0.807 | 0.703 |
| [7, 8, 9, 10] | 73.62% | 0.339 | 0.136 |
| [3, 8, 9, 10] | 61.62% | 0.297 ± 0.153 | 0.140 ± 0.088 |

[3, 8, 9, 10] is averaged over 5 target images; other rows are single targets.

Observations:
- **Placement matters more than boundary count.** Across seven 4-boundary placements, accuracy and SSIM are strongly anti-correlated (r ≈ −0.95): clustering boundaries early costs accuracy *and* leaks more.
- [3, 8, 9, 10] keeps one shallow boundary for defence in depth (83% of blocks protected vs 50% for [7, 8, 9, 10]) at a ~12-point accuracy cost relative to [7, 8, 9, 10].
- Narrower bottlenecks help privacy and hurt accuracy, as expected (single boundary at block 3: D' 100 → 64 → 32 gives SSIM 0.92 → 0.89 → 0.78).

## Caveats

- **Small attack sample.** Most rows use 1–5 target images; differences under ~0.1 SSIM are not significant yet.
- **Model selection uses the test split.** A held-out validation split is needed before any accuracy number is
