#train_deit_boundary_new_boundaries.py

import os
import math
import yaml
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# -----------------------------
# Load models
# -----------------------------

def load_local_pretrained_if_needed(model: nn.Module, cfg: Dict[str, Any], device: torch.device):
    """
    Load local ImageNet pretrained weights into model.vit.

    This is used when GPU node has no internet access.
    The local checkpoint should be a raw timm state_dict saved from:
        torch.save(timm_model.state_dict(), path)

    Classifier head mismatch is skipped automatically.
    """
    local_path = cfg.get("model", {}).get("local_pretrained_path", None)
    if not local_path:
        return

    if cfg.get("checkpoint", {}).get("load", None):
        print("[Info] checkpoint.load is provided; skip local pretrained loading.")
        return

    print(f"[Info] Loading local pretrained weights from: {local_path}")
    state = torch.load(local_path, map_location=device)

    if "model" in state:
        state = state["model"]
    elif "state_dict" in state:
        state = state["state_dict"]

    target_state = model.vit.state_dict()
    filtered_state = {}

    skipped = []

    for k, v in state.items():
        # Remove possible wrappers.
        kk = k
        if kk.startswith("module."):
            kk = kk[len("module."):]
        if kk.startswith("vit."):
            kk = kk[len("vit."):]

        if kk in target_state and target_state[kk].shape == v.shape:
            filtered_state[kk] = v
        else:
            skipped.append(kk)

    missing, unexpected = model.vit.load_state_dict(filtered_state, strict=False)

    print(f"[Info] Loaded {len(filtered_state)} local pretrained tensors.")
    print(f"[Info] Skipped {len(skipped)} tensors due to missing keys or shape mismatch.")
    print(f"[Info] Missing keys after loading: {len(missing)}")
    print(f"[Info] Unexpected keys after loading: {len(unexpected)}")

    if len(skipped) > 0:
        print("[Info] Skipped examples:", skipped[:10])
    if len(missing) > 0:
        print("[Info] Missing examples:", missing[:10])




# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(1, self.count)


# -----------------------------
# Data
# -----------------------------

def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, int]:
    data_cfg = cfg["data"]
    dataset_name = data_cfg["dataset"].lower()
    image_size = data_cfg.get("image_size", 224)
    batch_size = cfg["train"].get("batch_size", 128)
    num_workers = cfg["train"].get("num_workers", 4)

    # On GPU nodes, set download=false and use local dataset folders.
    download = data_cfg.get("download", False)

    if dataset_name == "cifar10":
        root = data_cfg.get("root", "../datasets/cifar10")
        num_classes = 10
        dataset_cls = datasets.CIFAR10
    elif dataset_name == "cifar100":
        root = data_cfg.get("root", "../datasets/cifar100")
        num_classes = 100
        dataset_cls = datasets.CIFAR100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Use cifar10 or cifar100.")

    print(f"[Info] Dataset: {dataset_name}")
    print(f"[Info] Dataset root: {root}")
    print(f"[Info] Dataset download: {download}")

    # ImageNet-pretrained DeiT expects ImageNet-like normalization.
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    train_set = dataset_cls(
        root=root,
        train=True,
        download=download,
        transform=train_tf,
    )
    test_set = dataset_cls(
        root=root,
        train=False,
        download=download,
        transform=test_tf,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, test_loader, num_classes
# -----------------------------
# Fake quantization with STE
# -----------------------------

class FakeQuantSTE(nn.Module):
    def __init__(self, bits: Optional[int] = 8):
        super().__init__()
        self.bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits is None or self.bits >= 32:
            return x

        qmax = 2 ** (self.bits - 1) - 1
        qmin = -qmax

        # Per-sample scale over token and hidden dimensions.
        scale = x.detach().abs().amax(dim=(1, 2), keepdim=True) / qmax
        scale = scale.clamp(min=1e-8)

        x_q = torch.round(x / scale).clamp(qmin, qmax) * scale

        # Straight-through estimator.
        return x + (x_q - x).detach()


# -----------------------------
# Power normalization
# -----------------------------

class PowerNorm(nn.Module):
    """
    Enforces a bounded feature power condition.

    mode:
      - "none": no normalization
      - "unit": normalize each sample to target RMS sqrt(power)
      - "clip": only scale down samples whose RMS exceeds sqrt(power)
    """
    def __init__(self, power: float = 1.0, mode: str = "clip", eps: float = 1e-6):
        super().__init__()
        self.power = power
        self.mode = mode
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x

        target_rms = math.sqrt(self.power)
        rms = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()

        if self.mode == "unit":
            return x / (rms + self.eps) * target_rms

        if self.mode == "clip":
            scale = torch.clamp(target_rms / (rms + self.eps), max=1.0)
            return x * scale

        raise ValueError(f"Unknown PowerNorm mode: {self.mode}")


# -----------------------------
# Boundary modules
# -----------------------------

class IdentityBoundary(nn.Module):
    """
    No-op boundary. Useful for debugging insertion.
    """
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return z, z


class LinearNoiseQuantBoundary(nn.Module):
    """
    Token-wise boundary:
        z: [B, T, D]
        u = down(z): [B, T, D']
        u = LayerNorm + PowerNorm + Gaussian noise + fake quantization
        z_out = up(u): [B, T, D]
    Exposed representation is u.
    """
    def __init__(
        self,
        dim: int,
        bottleneck_dim: int,
        sigma: float = 0.0,
        quant_bits: Optional[int] = 8,
        power: float = 1.0,
        power_norm_mode: str = "clip",
        noise_in_eval: bool = True,
    ):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.norm = nn.LayerNorm(bottleneck_dim)
        self.power_norm = PowerNorm(power=power, mode=power_norm_mode)
        self.quant = FakeQuantSTE(bits=quant_bits)
        self.up = nn.Linear(bottleneck_dim, dim)

        self.sigma = sigma
        self.noise_in_eval = noise_in_eval

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = self.down(z)
        u = self.norm(u)
        u = self.power_norm(u)

        if self.sigma > 0 and (self.training or self.noise_in_eval):
            u = u + self.sigma * torch.randn_like(u)

        u = self.quant(u)
        z_out = self.up(u)
        return z_out, u


class TokenMixNoiseQuantBoundary(nn.Module):
    """
    ViT-specific token-mixing boundary:
        z: [B, T, D]
        hidden compression: D -> D'
        token compression: T -> T'
        noise + quantization on exposed [B, T', D']
        token expansion: T' -> T
        hidden expansion: D' -> D

    This gives exposed dimension d = T' * D'.
    """
    def __init__(
        self,
        seq_len: int,
        dim: int,
        token_bottleneck: int,
        bottleneck_dim: int,
        sigma: float = 0.0,
        quant_bits: Optional[int] = 8,
        power: float = 1.0,
        power_norm_mode: str = "clip",
        noise_in_eval: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.token_bottleneck = token_bottleneck

        self.hidden_down = nn.Linear(dim, bottleneck_dim)
        self.token_down = nn.Linear(seq_len, token_bottleneck)
        self.norm = nn.LayerNorm(bottleneck_dim)
        self.power_norm = PowerNorm(power=power, mode=power_norm_mode)
        self.quant = FakeQuantSTE(bits=quant_bits)

        self.token_up = nn.Linear(token_bottleneck, seq_len)
        self.hidden_up = nn.Linear(bottleneck_dim, dim)

        self.sigma = sigma
        self.noise_in_eval = noise_in_eval

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # z: [B, T, D]
        u = self.hidden_down(z)          # [B, T, D']
        u = self.norm(u)

        # Token compression: [B, T, D'] -> [B, D', T] -> [B, D', T'] -> [B, T', D']
        u = u.transpose(1, 2)
        u = self.token_down(u)
        u = u.transpose(1, 2)

        u = self.power_norm(u)

        if self.sigma > 0 and (self.training or self.noise_in_eval):
            u = u + self.sigma * torch.randn_like(u)

        u = self.quant(u)                # exposed representation [B, T', D']

        # Expand back to original shape for remaining DeiT blocks.
        z_out = u.transpose(1, 2)        # [B, D', T']
        z_out = self.token_up(z_out)     # [B, D', T]
        z_out = z_out.transpose(1, 2)    # [B, T, D']
        z_out = self.hidden_up(z_out)    # [B, T, D]

        return z_out, u



class ResidualAdapterBoundary(nn.Module):
    """
    Residual-adapter boundary.

    This keeps the same exposed representation as the linear randomized bottleneck:
        u = Q(PowerNorm(LN(down(z))) + noise)
    but uses a stronger deterministic adapter after the exposed representation:
        v = up(u)
        z_out = v + gamma * MLP(LN(v))

    Privacy note: the original z is never added back to z_out. Both z_out and all
    downstream activations are deterministic post-processing of the exposed noisy
    bottleneck u, so the leakage certificate for u still applies by data processing.
    """
    def __init__(
        self,
        dim: int,
        bottleneck_dim: int,
        sigma: float = 0.0,
        quant_bits: Optional[int] = 8,
        power: float = 1.0,
        power_norm_mode: str = "clip",
        noise_in_eval: bool = True,
        adapter_mlp_ratio: float = 2.0,
        adapter_gamma_init: float = 0.1,
        adapter_dropout: float = 0.0,
    ):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.norm = nn.LayerNorm(bottleneck_dim)
        self.power_norm = PowerNorm(power=power, mode=power_norm_mode)
        self.quant = FakeQuantSTE(bits=quant_bits)
        self.up = nn.Linear(bottleneck_dim, dim)

        hidden_dim = max(1, int(dim * adapter_mlp_ratio))
        self.adapter_norm = nn.LayerNorm(dim)
        self.adapter = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(adapter_dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(adapter_dropout),
        )
        self.gamma = nn.Parameter(torch.tensor(float(adapter_gamma_init)))

        self.sigma = sigma
        self.noise_in_eval = noise_in_eval

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = self.down(z)
        u = self.norm(u)
        u = self.power_norm(u)

        if self.sigma > 0 and (self.training or self.noise_in_eval):
            u = u + self.sigma * torch.randn_like(u)

        u = self.quant(u)
        v = self.up(u)
        z_out = v + self.gamma * self.adapter(self.adapter_norm(v))
        return z_out, u


class CLSPreservingLatentTokenBoundary(nn.Module):
    """
    CLS-preserving latent-token boundary for ViT/DeiT.

    Input:
        z: [B, T, D], where the first prefix_tokens are CLS/dist tokens and the
           remaining tokens are patch tokens.

    Exposed representation:
        u: [B, prefix_tokens + latent_tokens, D']

    Design:
      1) Prefix tokens are compressed independently, not mixed with patch tokens.
      2) Patch tokens are compressed into a small set of learnable latent privacy
         tokens through cross-attention.
      3) The concatenated compressed prefix + latent patch tokens are power-normalized,
         noised, and quantized.
      4) A deterministic expansion maps the sanitized representation back to [B,T,D]
         so the remaining DeiT blocks can be reused.

    Privacy note: the exposed dimension is (prefix_tokens + latent_tokens) * D'.
    """
    def __init__(
        self,
        seq_len: int,
        dim: int,
        latent_tokens: int,
        bottleneck_dim: int,
        prefix_tokens: int = 1,
        num_heads: int = 4,
        sigma: float = 0.0,
        quant_bits: Optional[int] = 8,
        power: float = 1.0,
        power_norm_mode: str = "clip",
        noise_in_eval: bool = True,
        adapter_mlp_ratio: float = 1.0,
        adapter_gamma_init: float = 0.1,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if prefix_tokens < 1:
            raise ValueError("prefix_tokens should be at least 1 for CLS-preserving boundary.")
        if prefix_tokens >= seq_len:
            raise ValueError(f"prefix_tokens={prefix_tokens} must be smaller than seq_len={seq_len}.")
        if bottleneck_dim % num_heads != 0:
            raise ValueError(
                f"bottleneck_dim={bottleneck_dim} must be divisible by num_heads={num_heads}."
            )

        self.seq_len = seq_len
        self.prefix_tokens = prefix_tokens
        self.patch_tokens = seq_len - prefix_tokens
        self.latent_tokens = latent_tokens
        self.bottleneck_dim = bottleneck_dim

        self.prefix_down = nn.Linear(dim, bottleneck_dim)
        self.patch_down = nn.Linear(dim, bottleneck_dim)

        self.latent_queries = nn.Parameter(torch.randn(1, latent_tokens, bottleneck_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=bottleneck_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(bottleneck_dim)
        self.power_norm = PowerNorm(power=power, mode=power_norm_mode)
        self.quant = FakeQuantSTE(bits=quant_bits)

        self.prefix_up = nn.Linear(bottleneck_dim, dim)
        self.latent_hidden_up = nn.Linear(bottleneck_dim, dim)
        self.token_up = nn.Linear(latent_tokens, self.patch_tokens)

        hidden_dim = max(1, int(dim * adapter_mlp_ratio))
        self.adapter_norm = nn.LayerNorm(dim)
        self.adapter = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.gamma = nn.Parameter(torch.tensor(float(adapter_gamma_init)))

        self.sigma = sigma
        self.noise_in_eval = noise_in_eval

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # z: [B, T, D]
        if z.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {z.size(1)}")

        z_prefix = z[:, :self.prefix_tokens, :]
        z_patch = z[:, self.prefix_tokens:, :]

        u_prefix = self.prefix_down(z_prefix)      # [B, P, D']
        patch_kv = self.patch_down(z_patch)        # [B, N, D']

        B = z.size(0)
        queries = self.latent_queries.expand(B, -1, -1)  # [B, T', D']
        u_patch, _ = self.cross_attn(queries, patch_kv, patch_kv, need_weights=False)

        u = torch.cat([u_prefix, u_patch], dim=1)  # [B, P+T', D']
        u = self.norm(u)
        u = self.power_norm(u)

        if self.sigma > 0 and (self.training or self.noise_in_eval):
            u = u + self.sigma * torch.randn_like(u)

        u = self.quant(u)

        # Deterministically expand sanitized representation back to original ViT shape.
        u_prefix_q = u[:, :self.prefix_tokens, :]
        u_patch_q = u[:, self.prefix_tokens:, :]

        z_prefix_out = self.prefix_up(u_prefix_q)       # [B, P, D]
        z_patch_latent = self.latent_hidden_up(u_patch_q)  # [B, T', D]

        # Expand latent tokens back to original patch-token count.
        z_patch_out = z_patch_latent.transpose(1, 2)    # [B, D, T']
        z_patch_out = self.token_up(z_patch_out)        # [B, D, N]
        z_patch_out = z_patch_out.transpose(1, 2)       # [B, N, D]

        z_out = torch.cat([z_prefix_out, z_patch_out], dim=1)
        z_out = z_out + self.gamma * self.adapter(self.adapter_norm(z_out))
        return z_out, u

def build_boundary_module(
    boundary_cfg: Dict[str, Any],
    seq_len: int,
    dim: int,
) -> nn.Module:
    name = boundary_cfg.get("module", "identity").lower()

    if name == "identity":
        return IdentityBoundary()

    if name == "linear_noise_quant":
        return LinearNoiseQuantBoundary(
            dim=dim,
            bottleneck_dim=boundary_cfg.get("bottleneck_dim", dim),
            sigma=boundary_cfg.get("sigma", 0.0),
            quant_bits=boundary_cfg.get("quant_bits", 8),
            power=boundary_cfg.get("power", 1.0),
            power_norm_mode=boundary_cfg.get("power_norm_mode", "clip"),
            noise_in_eval=boundary_cfg.get("noise_in_eval", True),
        )

    if name == "token_mix_noise_quant":
        return TokenMixNoiseQuantBoundary(
            seq_len=seq_len,
            dim=dim,
            token_bottleneck=boundary_cfg.get("token_bottleneck", seq_len),
            bottleneck_dim=boundary_cfg.get("bottleneck_dim", dim),
            sigma=boundary_cfg.get("sigma", 0.0),
            quant_bits=boundary_cfg.get("quant_bits", 8),
            power=boundary_cfg.get("power", 1.0),
            power_norm_mode=boundary_cfg.get("power_norm_mode", "clip"),
            noise_in_eval=boundary_cfg.get("noise_in_eval", True),
        )

    if name == "residual_adapter":
        return ResidualAdapterBoundary(
            dim=dim,
            bottleneck_dim=boundary_cfg.get("bottleneck_dim", dim),
            sigma=boundary_cfg.get("sigma", 0.0),
            quant_bits=boundary_cfg.get("quant_bits", 8),
            power=boundary_cfg.get("power", 1.0),
            power_norm_mode=boundary_cfg.get("power_norm_mode", "clip"),
            noise_in_eval=boundary_cfg.get("noise_in_eval", True),
            adapter_mlp_ratio=boundary_cfg.get("adapter_mlp_ratio", 2.0),
            adapter_gamma_init=boundary_cfg.get("adapter_gamma_init", 0.1),
            adapter_dropout=boundary_cfg.get("adapter_dropout", 0.0),
        )

    if name == "cls_latent_token":
        return CLSPreservingLatentTokenBoundary(
            seq_len=seq_len,
            dim=dim,
            latent_tokens=boundary_cfg.get("latent_tokens", boundary_cfg.get("token_bottleneck", 64)),
            bottleneck_dim=boundary_cfg.get("bottleneck_dim", dim),
            prefix_tokens=boundary_cfg.get("prefix_tokens", 1),
            num_heads=boundary_cfg.get("num_heads", 4),
            sigma=boundary_cfg.get("sigma", 0.0),
            quant_bits=boundary_cfg.get("quant_bits", 8),
            power=boundary_cfg.get("power", 1.0),
            power_norm_mode=boundary_cfg.get("power_norm_mode", "clip"),
            noise_in_eval=boundary_cfg.get("noise_in_eval", True),
            adapter_mlp_ratio=boundary_cfg.get("adapter_mlp_ratio", 1.0),
            adapter_gamma_init=boundary_cfg.get("adapter_gamma_init", 0.1),
            attn_dropout=boundary_cfg.get("attn_dropout", 0.0),
        )

    raise ValueError(f"Unsupported boundary module: {name}")


# -----------------------------
# DeiT wrapper with insertable boundary
# -----------------------------

class DeiTWithBoundary(nn.Module):
    """
    Wraps timm VisionTransformer / DeiT and inserts a boundary after a chosen number
    of transformer blocks.

    boundary_after_block:
      - None: no boundary
      - 0: insert boundary before block 1
      - 1: insert after first transformer block
      - ...
      - 12: insert after all 12 blocks for DeiT-Tiny/Small/Base

    This uses "after first k blocks" convention, not zero-based block index.
    """
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        boundary_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        if not hasattr(self.vit, "blocks"):
            raise ValueError("This wrapper expects a timm VisionTransformer-like model with .blocks")

        self.num_blocks = len(self.vit.blocks)
        self.embed_dim = self.vit.embed_dim
        self.num_classes = num_classes

        self.boundary_enabled = False
        self.boundary_after_block = None
        self.boundary = None

        # timm ViT seq length = num patches + cls token, possibly + dist token.
        num_patches = self.vit.patch_embed.num_patches
        extra_tokens = 1
        if getattr(self.vit, "dist_token", None) is not None:
            extra_tokens += 1
        self.seq_len = num_patches + extra_tokens

        if boundary_cfg is not None and boundary_cfg.get("enabled", False):
            self.boundary_enabled = True
            self.boundary_after_block = boundary_cfg.get("after_block", 9)

            if not (0 <= self.boundary_after_block <= self.num_blocks):
                raise ValueError(
                    f"boundary.after_block must be in [0, {self.num_blocks}], "
                    f"got {self.boundary_after_block}"
                )

            self.boundary = build_boundary_module(
                boundary_cfg=boundary_cfg,
                seq_len=self.seq_len,
                dim=self.embed_dim,
            )

        self.last_exposed_feature = None

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Handles DeiT/ViT positional embedding.
        This follows timm's common VisionTransformer structure.
        """
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

    def forward_features(self, x: torch.Tensor, return_exposed: bool = False):
        x = self.vit.patch_embed(x)
        x = self._pos_embed(x)

        exposed = None

        if self.boundary_enabled and self.boundary_after_block == 0:
            x, exposed = self.boundary(x)

        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if self.boundary_enabled and (i + 1) == self.boundary_after_block:
                x, exposed = self.boundary(x)

        x = self.vit.norm(x)

        if getattr(self.vit, "dist_token", None) is not None:
            # DeiT distilled models may use cls and dist tokens.
            feat = (x[:, 0] + x[:, 1]) / 2
        else:
            feat = x[:, 0]

        self.last_exposed_feature = exposed

        if return_exposed:
            return feat, exposed
        return feat

    def forward(self, x: torch.Tensor, return_exposed: bool = False):
        if return_exposed:
            feat, exposed = self.forward_features(x, return_exposed=True)
            logits = self.vit.head(feat)
            return logits, exposed

        feat = self.forward_features(x, return_exposed=False)
        logits = self.vit.head(feat)
        return logits


# -----------------------------
# Checkpoint loading
# -----------------------------

def load_checkpoint_if_needed(model: nn.Module, cfg: Dict[str, Any], device: torch.device):
    ckpt_path = cfg.get("checkpoint", {}).get("load", None)
    if not ckpt_path:
        return

    print(f"[Info] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Info] Missing keys: {len(missing)}")
    print(f"[Info] Unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("  Missing examples:", missing[:10])
    if len(unexpected) > 0:
        print("  Unexpected examples:", unexpected[:10])


def save_checkpoint(model: nn.Module, optimizer, epoch: int, best_acc: float, cfg: Dict[str, Any]):
    out_dir = cfg["checkpoint"].get("out_dir", "./checkpoints")
    ensure_dir(out_dir)

    name = cfg["checkpoint"].get("name", "deit_boundary")
    path = os.path.join(out_dir, f"{name}_epoch{epoch}_acc{best_acc:.4f}.pt")

    torch.save({
        "epoch": epoch,
        "best_acc": best_acc,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg,
    }, path)

    print(f"[Info] Saved checkpoint to {path}")


# -----------------------------
# Freeze policy
# -----------------------------

def apply_freeze_policy(model: DeiTWithBoundary, cfg: Dict[str, Any]):
    train_cfg = cfg["train"]
    mode = cfg.get("mode", "baseline").lower()

    # Default: train everything.
    for p in model.parameters():
        p.requires_grad = True

    if mode == "baseline":
        if train_cfg.get("head_only", False):
            for p in model.parameters():
                p.requires_grad = False
            for p in model.vit.head.parameters():
                p.requires_grad = True
        return

    if mode == "boundary":
        freeze_prefix = train_cfg.get("freeze_prefix", True)
        unfreeze_tail = train_cfg.get("unfreeze_tail", True)
        unfreeze_head = train_cfg.get("unfreeze_head", True)
        unfreeze_boundary = train_cfg.get("unfreeze_boundary", True)

        # Start by freezing everything.
        for p in model.parameters():
            p.requires_grad = False

        # Boundary module.
        if model.boundary is not None and unfreeze_boundary:
            for p in model.boundary.parameters():
                p.requires_grad = True

        # Tail transformer blocks after boundary.
        if unfreeze_tail:
            k = model.boundary_after_block if model.boundary_after_block is not None else model.num_blocks
            for i in range(k, model.num_blocks):
                for p in model.vit.blocks[i].parameters():
                    p.requires_grad = True

            # Final norm is part of tail.
            for p in model.vit.norm.parameters():
                p.requires_grad = True

        # Head.
        if unfreeze_head:
            for p in model.vit.head.parameters():
                p.requires_grad = True

        # Optional: if not freezing prefix, unfreeze all blocks before boundary too.
        if not freeze_prefix:
            for p in model.parameters():
                p.requires_grad = True

        return

    raise ValueError(f"Unsupported mode: {mode}")


def print_trainable_parameters(model: nn.Module):
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    print(f"[Info] Trainable params: {trainable/1e6:.3f}M / {total/1e6:.3f}M")


# -----------------------------
# Optimizer / scheduler
# -----------------------------

def build_optimizer(model: nn.Module, cfg: Dict[str, Any]):
    opt_cfg = cfg["optimizer"]
    lr = opt_cfg.get("lr", 5e-5)
    weight_decay = opt_cfg.get("weight_decay", 0.05)
    name = opt_cfg.get("name", "adamw").lower()

    params = [p for p in model.parameters() if p.requires_grad]

    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, cfg: Dict[str, Any]):
    sched_cfg = cfg.get("scheduler", {})
    name = sched_cfg.get("name", "cosine").lower()
    epochs = cfg["train"].get("epochs", 20)

    if name == "none":
        return None

    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=sched_cfg.get("eta_min", 1e-6),
        )

    if name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 10),
            gamma=sched_cfg.get("gamma", 0.1),
        )

    raise ValueError(f"Unsupported scheduler: {name}")


# -----------------------------
# Train / eval
# -----------------------------

def train_one_epoch(model, loader, optimizer, device, epoch: int, cfg: Dict[str, Any]):
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"Train epoch {epoch}", ncols=100)

    use_amp = cfg["train"].get("amp", True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)

        scaler.scale(loss).backward()

        grad_clip = cfg["train"].get("grad_clip", None)
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        acc = accuracy_top1(logits.detach(), targets)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(acc, bs)

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg*100:.2f}%")

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, device, cfg: Dict[str, Any]):
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    use_amp = cfg["train"].get("amp", True)

    for images, targets in tqdm(loader, desc="Eval", ncols=100):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)

        bs = images.size(0)
        acc = accuracy_top1(logits, targets)

        loss_meter.update(loss.item(), bs)
        acc_meter.update(acc, bs)

    return loss_meter.avg, acc_meter.avg


# -----------------------------
# Optional feature dumping
# -----------------------------

@torch.no_grad()
def dump_boundary_features(model, loader, device, cfg: Dict[str, Any]):
    dump_cfg = cfg.get("dump_features", {})
    if not dump_cfg.get("enabled", False):
        return

    out_path = dump_cfg.get("path", "./boundary_features.pt")
    max_batches = dump_cfg.get("max_batches", 10)

    model.eval()

    all_exposed = []
    all_labels = []

    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="Dump features", ncols=100)):
        if batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        logits, exposed = model(images, return_exposed=True)

        if exposed is None:
            raise RuntimeError("No exposed boundary feature found. Enable boundary first.")

        all_exposed.append(exposed.cpu())
        all_labels.append(targets.cpu())

    payload = {
        "features": torch.cat(all_exposed, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "cfg": cfg,
    }

    ensure_dir(os.path.dirname(out_path) or ".")
    torch.save(payload, out_path)
    print(f"[Info] Dumped boundary features to {out_path}")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    set_seed(cfg.get("seed", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    train_loader, test_loader, num_classes = build_dataloaders(cfg)

    model_cfg = cfg["model"]
    boundary_cfg = cfg.get("boundary", {"enabled": False})

    model = DeiTWithBoundary(
        model_name=model_cfg.get("name", "deit_tiny_patch16_224"),
        pretrained=model_cfg.get("pretrained", True),
        num_classes=num_classes,
        boundary_cfg=boundary_cfg,
    )


    model.to(device)

    load_local_pretrained_if_needed(model, cfg, device)
    load_checkpoint_if_needed(model, cfg, device)

    apply_freeze_policy(model, cfg)
    print_trainable_parameters(model)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    epochs = cfg["train"].get("epochs", 20)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch, cfg)
        val_loss, val_acc = evaluate(model, test_loader, device, cfg)

        if scheduler is not None:
            scheduler.step()

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc*100:.2f}%, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            if cfg.get("checkpoint", {}).get("save_best", True):
                save_checkpoint(model, optimizer, epoch, best_acc, cfg)

    print(f"[Info] Best val acc: {best_acc*100:.2f}%")

    dump_boundary_features(model, test_loader, device, cfg)


if __name__ == "__main__":
    main()