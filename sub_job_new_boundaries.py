#!/usr/bin/env python3
"""
Automate DeiT boundary sweep experiments.

Public functions:
  1) create_sweep_excel(...)
     Create an Excel tracker with one row per experiment.

  2) submit_jobs_from_excel(...)
     Read tracker rows, write YAML configs, write Slurm job scripts, submit jobs,
     and write job IDs / paths back to the Excel file.

Extra utility:
  collect_results_from_logs(...)
     Parse Slurm logs after jobs finish and write best/last validation accuracy back to Excel.

Example:
  python sub_job.py --create
  python sub_job.py --submit --row-min 2 --row-max 20
  python sub_job.py --collect --row-min 2 --row-max 20
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import openpyxl
import yaml


# -----------------------------------------------------------------------------
# Default paths and cluster settings. Edit these once for your project.
# -----------------------------------------------------------------------------

DEFAULT_EXCEL = "train_results.xlsx"
DEFAULT_TEMPLATE_YAML = "deit_tiny_cifar100_boundary.yaml"
DEFAULT_CONFIG_DIR = "configs/generated"
DEFAULT_JOB_DIR = "jobs"
DEFAULT_LOG_DIR = "logs"
DEFAULT_TRAIN_SCRIPT = "train_deit_boundary.py"

DEFAULT_ACCOUNT = "def-enright"
DEFAULT_TIME = "12:00:00"
DEFAULT_CPUS = 8
DEFAULT_MEM = "32G"
DEFAULT_GPUS = 1
DEFAULT_ENV_ACTIVATE = "source ./ENV/bin/activate"

DEFAULT_MODULE_LINES = [
    "module load StdEnv/2023",
    "module load cuda/12.6",
    "module load python/3.11.5",
    "module load gcc arrow",
]

DEFAULT_EXPORT_LINES = [
    "export HF_HUB_OFFLINE=1",
    "export TRANSFORMERS_OFFLINE=1",
    "export TIMM_FUSED_ATTN=0",
]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _mkdir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _sanitize(s: Any) -> str:
    s = "none" if s is None else str(s)
    s = s.replace(".", "p")
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _as_bool(x: Any, default: bool = False) -> bool:
    if x is None or x == "":
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    return default


def _none_if_blank(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, str) and x.strip().lower() in {"", "none", "null", "nan"}:
        return None
    return x


def _float_or_none(x: Any) -> Optional[float]:
    x = _none_if_blank(x)
    if x is None:
        return None
    return float(x)


def _int_or_none(x: Any) -> Optional[int]:
    x = _none_if_blank(x)
    if x is None:
        return None
    return int(x)


def _header_map(ws) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for c in range(1, ws.max_column + 1):
        v = ws.cell(row=1, column=c).value
        if v is not None:
            mapping[str(v).strip()] = c
    return mapping


def _get(row, h: Dict[str, int], key: str, default: Any = None) -> Any:
    c = h.get(key)
    if c is None:
        return default
    v = row[c - 1]
    return default if v is None else v


def _set(ws, row_num: int, h: Dict[str, int], key: str, value: Any) -> None:
    if key not in h:
        ws.cell(row=1, column=ws.max_column + 1, value=key)
        h[key] = ws.max_column
    ws.cell(row=row_num, column=h[key], value=value)


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _write_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    _mkdir(Path(path).parent)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _format_sigma_for_name(sigma: Any) -> str:
    if sigma is None:
        return "none"
    try:
        return f"s{float(sigma):.3f}".replace(".", "p")
    except Exception:
        return _sanitize(sigma)


def _format_lr_for_name(lr: Any) -> str:
    if lr is None:
        return "lrnone"
    try:
        return f"lr{float(lr):.0e}".replace("-", "m").replace("+", "p")
    except Exception:
        return _sanitize(lr)


def _make_run_name(row_dict: Dict[str, Any]) -> str:
    if row_dict.get("run_name"):
        return _sanitize(row_dict["run_name"])

    model_short = _sanitize(row_dict.get("model", "deit_tiny"))
    dataset = _sanitize(row_dict.get("dataset", "cifar100"))
    mode = _sanitize(row_dict.get("mode", "boundary"))

    if mode == "baseline" or not _as_bool(row_dict.get("boundary_enabled"), False):
        return f"{model_short}_{dataset}_baseline"

    module = _sanitize(row_dict.get("module", "linear_noise_quant"))
    after_block = _sanitize(row_dict.get("after_block", "na"))
    bottleneck_dim = _sanitize(row_dict.get("bottleneck_dim", "na"))
    token_bottleneck = row_dict.get("token_bottleneck")
    latent_tokens = row_dict.get("latent_tokens")
    sigma = _format_sigma_for_name(row_dict.get("sigma"))
    q = _sanitize(row_dict.get("quant_bits", "na"))
    lr = _format_lr_for_name(row_dict.get("lr"))

    name = f"{model_short}_{dataset}_b{after_block}_{module}_d{bottleneck_dim}"
    if token_bottleneck not in {None, "", "none"}:
        name += f"_t{_sanitize(token_bottleneck)}"
    if latent_tokens not in {None, "", "none"}:
        name += f"_lat{_sanitize(latent_tokens)}"
    name += f"_{sigma}_q{q}_{lr}"
    return name


def _row_to_dict(ws, row_num: int, h: Dict[str, int]) -> Dict[str, Any]:
    values = [ws.cell(row=row_num, column=c).value for c in range(1, ws.max_column + 1)]
    return {k: _get(values, h, k) for k in h.keys()}


# -----------------------------------------------------------------------------
# Function 1: create Excel tracker
# -----------------------------------------------------------------------------

def create_sweep_excel(
    filename: str = DEFAULT_EXCEL,
    datasets: Iterable[str] = ("cifar100",),
    model: str = "deit_tiny_patch16_224",
    model_alias: str = "deit_tiny",
    data_roots: Optional[Dict[str, str]] = None,
    local_pretrained_path: str = "../pretrained/deit_tiny_patch16_224_imagenet_state_dict.pt",
    baseline_checkpoints: Optional[Dict[str, str]] = None,
) -> str:
    """
    Create an Excel file that records sweep parameters and job/result metadata.

    By default, it creates:
      - one baseline row per dataset;
      - boundary sweep rows for linear_noise_quant;
      - a small token_mix_noise_quant sweep for later comparison.

    Edit the grids below if you want more or fewer experiments.
    """
    data_roots = data_roots or {
        "cifar10": "../datasets/cifar10",
        "cifar100": "../datasets/cifar100",
    }
    baseline_checkpoints = baseline_checkpoints or {
        "cifar10": "checkpoints/deit_tiny_cifar10_baseline_epochXX_accX.XXXX.pt",
        "cifar100": "checkpoints/deit_tiny_cifar100_baseline_epoch39_acc0.8143.pt",
    }

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "jobs"

    headers = [
        # Identity and dataset/model
        "row_id", "run_name", "model", "model_name", "dataset", "mode",
        "data_root", "image_size", "download", "local_pretrained_path",
        "baseline_checkpoint",
        # Boundary parameters
        "boundary_enabled", "after_block", "module", "bottleneck_dim",
        "token_bottleneck", "sigma", "quant_bits", "power", "power_norm_mode",
        "noise_in_eval",
        # Extra boundary parameters for residual_adapter and cls_latent_token
        "adapter_mlp_ratio", "adapter_gamma_init", "adapter_dropout",
        "latent_tokens", "prefix_tokens", "num_heads", "attn_dropout",
        # Training parameters
        "epochs", "batch_size", "num_workers", "amp", "grad_clip",
        "freeze_prefix", "unfreeze_tail", "unfreeze_head", "unfreeze_boundary",
        # Optimizer/scheduler
        "optimizer", "lr", "weight_decay", "scheduler", "eta_min",
        # Checkpoint / dumping
        "checkpoint_out_dir", "dump_enabled", "dump_max_batches",
        # Slurm/job settings
        "submit", "job_time", "account", "cpus", "mem", "gpus",
        # Auto-filled fields
        "yaml_path", "job_path", "job_id", "status", "last_val_acc", "best_acc", "notes",
    ]
    ws.append(headers)

    row_id = 1

    # Baseline rows.
    for ds in datasets:
        ds = ds.lower()
        row = {
            "row_id": row_id,
            "run_name": f"{model_alias}_{ds}_baseline",
            "model": model_alias,
            "model_name": model,
            "dataset": ds,
            "mode": "baseline",
            "data_root": data_roots.get(ds, f"../datasets/{ds}"),
            "image_size": 224,
            "download": False,
            "local_pretrained_path": local_pretrained_path,
            "baseline_checkpoint": None,
            "boundary_enabled": False,
            "after_block": None,
            "module": None,
            "bottleneck_dim": None,
            "token_bottleneck": None,
            "sigma": None,
            "quant_bits": None,
            "power": 1.0,
            "power_norm_mode": "clip",
            "noise_in_eval": True,
            "adapter_mlp_ratio": None,
            "adapter_gamma_init": None,
            "adapter_dropout": None,
            "latent_tokens": None,
            "prefix_tokens": None,
            "num_heads": None,
            "attn_dropout": None,
            "epochs": 50 if ds == "cifar100" else 20,
            "batch_size": 128,
            "num_workers": 4,
            "amp": True,
            "grad_clip": 1.0,
            "freeze_prefix": None,
            "unfreeze_tail": None,
            "unfreeze_head": None,
            "unfreeze_boundary": None,
            "optimizer": "adamw",
            "lr": 3e-5 if ds == "cifar100" else 5e-5,
            "weight_decay": 0.05,
            "scheduler": "cosine",
            "eta_min": 1e-6,
            "checkpoint_out_dir": "./checkpoints",
            "dump_enabled": False,
            "dump_max_batches": 20,
            "submit": True,
            "job_time": DEFAULT_TIME,
            "account": DEFAULT_ACCOUNT,
            "cpus": DEFAULT_CPUS,
            "mem": DEFAULT_MEM,
            "gpus": DEFAULT_GPUS,
            "yaml_path": None,
            "job_path": None,
            "job_id": None,
            "status": "pending",
            "last_val_acc": None,
            "best_acc": None,
            "notes": "baseline",
        }
        ws.append([row.get(hh) for hh in headers])
        row_id += 1

    # Boundary sweep rows. Conservative first, then stronger settings.
    linear_sweeps = [
        # after_block, bottleneck_dim, sigma, quant_bits, epochs, lr, note
        (9, 192, 0.02, 8, 50, 1e-4, "conservative"),
        (9, 128, 0.05, 8, 50, 1e-4, "main"),
        (9, 128, 0.10, 8, 50, 1e-4, "stronger_noise"),
        (9, 64,  0.05, 8, 50, 1e-4, "stronger_compression"),
        (8, 128, 0.05, 8, 50, 1e-4, "earlier_boundary"),
        (6, 192, 0.02, 8, 50, 1e-4, "early_rescue"),
        (6, 128, 0.05, 8, 50, 1e-4, "aggressive"),
    ]

    token_mix_sweeps = [
        # after_block, token_bottleneck, bottleneck_dim, sigma, quant_bits, epochs, lr, note
        (9, 128, 128, 0.05, 8, 50, 1e-4, "token_mix_main"),
        (9, 64,  128, 0.05, 8, 50, 1e-4, "token_mix_lower_d"),
        (8, 128, 128, 0.05, 8, 50, 1e-4, "token_mix_early"),
    ]

    residual_adapter_sweeps = [
        # after_block, bottleneck_dim, sigma, quant_bits, adapter_mlp_ratio, adapter_gamma_init, adapter_dropout, epochs, lr, note
        (9, 192, 0.02, 8, 2.0, 0.1, 0.0, 50, 1e-4, "residual_conservative"),
        (9, 128, 0.05, 8, 2.0, 0.1, 0.0, 50, 1e-4, "residual_main"),
        (9, 128, 0.10, 8, 2.0, 0.1, 0.0, 50, 1e-4, "residual_stronger_noise"),
        (8, 128, 0.05, 8, 2.0, 0.1, 0.0, 50, 1e-4, "residual_early"),
        (6, 192, 0.02, 8, 2.0, 0.1, 0.0, 50, 1e-4, "residual_early_rescue"),
    ]

    cls_latent_sweeps = [
        # after_block, latent_tokens, bottleneck_dim, sigma, quant_bits, prefix_tokens, num_heads, adapter_mlp_ratio, adapter_gamma_init, attn_dropout, epochs, lr, note
        (9, 128, 192, 0.02, 8, 1, 4, 1.0, 0.1, 0.0, 50, 1e-4, "cls_latent_conservative"),
        (9, 64,  128, 0.05, 8, 1, 4, 1.0, 0.1, 0.0, 50, 1e-4, "cls_latent_main"),
        (9, 32,  128, 0.05, 8, 1, 4, 1.0, 0.1, 0.0, 50, 1e-4, "cls_latent_lower_token"),
        (8, 64,  128, 0.05, 8, 1, 4, 1.0, 0.1, 0.0, 50, 1e-4, "cls_latent_early"),
        (6, 128, 192, 0.02, 8, 1, 4, 1.0, 0.1, 0.0, 50, 1e-4, "cls_latent_early_rescue"),
    ]

    for ds in datasets:
        ds = ds.lower()
        for after_block, bottleneck_dim, sigma, qbits, epochs, lr, note in linear_sweeps:
            row = {
                "row_id": row_id,
                "run_name": None,
                "model": model_alias,
                "model_name": model,
                "dataset": ds,
                "mode": "boundary",
                "data_root": data_roots.get(ds, f"../datasets/{ds}"),
                "image_size": 224,
                "download": False,
                "local_pretrained_path": None,
                "baseline_checkpoint": baseline_checkpoints.get(ds),
                "boundary_enabled": True,
                "after_block": after_block,
                "module": "linear_noise_quant",
                "bottleneck_dim": bottleneck_dim,
                "token_bottleneck": None,
                "sigma": sigma,
                "quant_bits": qbits,
                "power": 1.0,
                "power_norm_mode": "clip",
                "noise_in_eval": True,
                "adapter_mlp_ratio": None,
                "adapter_gamma_init": None,
                "adapter_dropout": None,
                "latent_tokens": None,
                "prefix_tokens": None,
                "num_heads": None,
                "attn_dropout": None,
                "epochs": epochs,
                "batch_size": 128,
                "num_workers": 4,
                "amp": True,
                "grad_clip": 1.0,
                "freeze_prefix": True,
                "unfreeze_tail": True,
                "unfreeze_head": True,
                "unfreeze_boundary": True,
                "optimizer": "adamw",
                "lr": lr,
                "weight_decay": 0.05,
                "scheduler": "cosine",
                "eta_min": 1e-6,
                "checkpoint_out_dir": "./checkpoints",
                "dump_enabled": True,
                "dump_max_batches": 20,
                "submit": True,
                "job_time": DEFAULT_TIME,
                "account": DEFAULT_ACCOUNT,
                "cpus": DEFAULT_CPUS,
                "mem": DEFAULT_MEM,
                "gpus": DEFAULT_GPUS,
                "yaml_path": None,
                "job_path": None,
                "job_id": None,
                "status": "pending",
                "last_val_acc": None,
                "best_acc": None,
                "notes": note,
            }
            row["run_name"] = _make_run_name(row)
            ws.append([row.get(hh) for hh in headers])
            row_id += 1

        for after_block, token_bottleneck, bottleneck_dim, sigma, qbits, epochs, lr, note in token_mix_sweeps:
            row = {
                "row_id": row_id,
                "run_name": None,
                "model": model_alias,
                "model_name": model,
                "dataset": ds,
                "mode": "boundary",
                "data_root": data_roots.get(ds, f"../datasets/{ds}"),
                "image_size": 224,
                "download": False,
                "local_pretrained_path": None,
                "baseline_checkpoint": baseline_checkpoints.get(ds),
                "boundary_enabled": True,
                "after_block": after_block,
                "module": "token_mix_noise_quant",
                "bottleneck_dim": bottleneck_dim,
                "token_bottleneck": token_bottleneck,
                "sigma": sigma,
                "quant_bits": qbits,
                "power": 1.0,
                "power_norm_mode": "clip",
                "noise_in_eval": True,
                "adapter_mlp_ratio": None,
                "adapter_gamma_init": None,
                "adapter_dropout": None,
                "latent_tokens": None,
                "prefix_tokens": None,
                "num_heads": None,
                "attn_dropout": None,
                "epochs": epochs,
                "batch_size": 128,
                "num_workers": 4,
                "amp": True,
                "grad_clip": 1.0,
                "freeze_prefix": True,
                "unfreeze_tail": True,
                "unfreeze_head": True,
                "unfreeze_boundary": True,
                "optimizer": "adamw",
                "lr": lr,
                "weight_decay": 0.05,
                "scheduler": "cosine",
                "eta_min": 1e-6,
                "checkpoint_out_dir": "./checkpoints",
                "dump_enabled": True,
                "dump_max_batches": 20,
                "submit": True,
                "job_time": DEFAULT_TIME,
                "account": DEFAULT_ACCOUNT,
                "cpus": DEFAULT_CPUS,
                "mem": DEFAULT_MEM,
                "gpus": DEFAULT_GPUS,
                "yaml_path": None,
                "job_path": None,
                "job_id": None,
                "status": "pending",
                "last_val_acc": None,
                "best_acc": None,
                "notes": note,
            }
            row["run_name"] = _make_run_name(row)
            ws.append([row.get(hh) for hh in headers])
            row_id += 1

        for after_block, bottleneck_dim, sigma, qbits, adapter_mlp_ratio, adapter_gamma_init, adapter_dropout, epochs, lr, note in residual_adapter_sweeps:
            row = {
                "row_id": row_id,
                "run_name": None,
                "model": model_alias,
                "model_name": model,
                "dataset": ds,
                "mode": "boundary",
                "data_root": data_roots.get(ds, f"../datasets/{ds}"),
                "image_size": 224,
                "download": False,
                "local_pretrained_path": None,
                "baseline_checkpoint": baseline_checkpoints.get(ds),
                "boundary_enabled": True,
                "after_block": after_block,
                "module": "residual_adapter",
                "bottleneck_dim": bottleneck_dim,
                "token_bottleneck": None,
                "sigma": sigma,
                "quant_bits": qbits,
                "power": 1.0,
                "power_norm_mode": "clip",
                "noise_in_eval": True,
                "adapter_mlp_ratio": adapter_mlp_ratio,
                "adapter_gamma_init": adapter_gamma_init,
                "adapter_dropout": adapter_dropout,
                "latent_tokens": None,
                "prefix_tokens": None,
                "num_heads": None,
                "attn_dropout": None,
                "epochs": epochs,
                "batch_size": 128,
                "num_workers": 4,
                "amp": True,
                "grad_clip": 1.0,
                "freeze_prefix": True,
                "unfreeze_tail": True,
                "unfreeze_head": True,
                "unfreeze_boundary": True,
                "optimizer": "adamw",
                "lr": lr,
                "weight_decay": 0.05,
                "scheduler": "cosine",
                "eta_min": 1e-6,
                "checkpoint_out_dir": "./checkpoints",
                "dump_enabled": True,
                "dump_max_batches": 20,
                "submit": True,
                "job_time": DEFAULT_TIME,
                "account": DEFAULT_ACCOUNT,
                "cpus": DEFAULT_CPUS,
                "mem": DEFAULT_MEM,
                "gpus": DEFAULT_GPUS,
                "yaml_path": None,
                "job_path": None,
                "job_id": None,
                "status": "pending",
                "last_val_acc": None,
                "best_acc": None,
                "notes": note,
            }
            row["run_name"] = _make_run_name(row)
            ws.append([row.get(hh) for hh in headers])
            row_id += 1

        for after_block, latent_tokens, bottleneck_dim, sigma, qbits, prefix_tokens, num_heads, adapter_mlp_ratio, adapter_gamma_init, attn_dropout, epochs, lr, note in cls_latent_sweeps:
            row = {
                "row_id": row_id,
                "run_name": None,
                "model": model_alias,
                "model_name": model,
                "dataset": ds,
                "mode": "boundary",
                "data_root": data_roots.get(ds, f"../datasets/{ds}"),
                "image_size": 224,
                "download": False,
                "local_pretrained_path": None,
                "baseline_checkpoint": baseline_checkpoints.get(ds),
                "boundary_enabled": True,
                "after_block": after_block,
                "module": "cls_latent_token",
                "bottleneck_dim": bottleneck_dim,
                "token_bottleneck": None,
                "sigma": sigma,
                "quant_bits": qbits,
                "power": 1.0,
                "power_norm_mode": "clip",
                "noise_in_eval": True,
                "adapter_mlp_ratio": adapter_mlp_ratio,
                "adapter_gamma_init": adapter_gamma_init,
                "adapter_dropout": 0.0,
                "latent_tokens": latent_tokens,
                "prefix_tokens": prefix_tokens,
                "num_heads": num_heads,
                "attn_dropout": attn_dropout,
                "epochs": epochs,
                "batch_size": 128,
                "num_workers": 4,
                "amp": True,
                "grad_clip": 1.0,
                "freeze_prefix": True,
                "unfreeze_tail": True,
                "unfreeze_head": True,
                "unfreeze_boundary": True,
                "optimizer": "adamw",
                "lr": lr,
                "weight_decay": 0.05,
                "scheduler": "cosine",
                "eta_min": 1e-6,
                "checkpoint_out_dir": "./checkpoints",
                "dump_enabled": True,
                "dump_max_batches": 20,
                "submit": True,
                "job_time": DEFAULT_TIME,
                "account": DEFAULT_ACCOUNT,
                "cpus": DEFAULT_CPUS,
                "mem": DEFAULT_MEM,
                "gpus": DEFAULT_GPUS,
                "yaml_path": None,
                "job_path": None,
                "job_id": None,
                "status": "pending",
                "last_val_acc": None,
                "best_acc": None,
                "notes": note,
            }
            row["run_name"] = _make_run_name(row)
            ws.append([row.get(hh) for hh in headers])
            row_id += 1

    ws.freeze_panes = "A2"
    for col in range(1, ws.max_column + 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = min(28, max(10, len(str(ws.cell(1, col).value)) + 2))

    wb.save(filename)
    print(f"[Info] Wrote sweep Excel: {filename} ({row_id - 1} rows)")
    return filename


# -----------------------------------------------------------------------------
# YAML and job generation
# -----------------------------------------------------------------------------

def build_yaml_from_row(row: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(template)

    mode = str(row.get("mode") or "boundary").strip().lower()
    dataset = str(row.get("dataset") or "cifar100").strip().lower()
    run_name = _make_run_name(row)

    cfg["seed"] = int(row.get("seed") or 0) if row.get("seed") is not None else 0
    cfg["mode"] = mode

    cfg.setdefault("data", {})
    cfg["data"]["dataset"] = dataset
    cfg["data"]["root"] = row.get("data_root") or f"../datasets/{dataset}"
    cfg["data"]["image_size"] = int(row.get("image_size") or 224)
    cfg["data"]["download"] = _as_bool(row.get("download"), False)

    cfg.setdefault("model", {})
    cfg["model"]["name"] = row.get("model_name") or "deit_tiny_patch16_224"
    cfg["model"]["pretrained"] = False
    cfg["model"]["local_pretrained_path"] = _none_if_blank(row.get("local_pretrained_path"))

    boundary_enabled = _as_bool(row.get("boundary_enabled"), mode == "boundary")
    cfg.setdefault("boundary", {})
    cfg["boundary"]["enabled"] = boundary_enabled

    if boundary_enabled:
        cfg["boundary"]["after_block"] = _int_or_none(row.get("after_block"))
        cfg["boundary"]["module"] = row.get("module") or "linear_noise_quant"
        cfg["boundary"]["bottleneck_dim"] = _int_or_none(row.get("bottleneck_dim"))
        if _none_if_blank(row.get("token_bottleneck")) is not None:
            cfg["boundary"]["token_bottleneck"] = _int_or_none(row.get("token_bottleneck"))
        else:
            cfg["boundary"].pop("token_bottleneck", None)
        cfg["boundary"]["sigma"] = float(row.get("sigma") or 0.0)
        cfg["boundary"]["quant_bits"] = _int_or_none(row.get("quant_bits"))
        cfg["boundary"]["power"] = float(row.get("power") or 1.0)
        cfg["boundary"]["power_norm_mode"] = row.get("power_norm_mode") or "clip"
        cfg["boundary"]["noise_in_eval"] = _as_bool(row.get("noise_in_eval"), True)

        # Optional fields for newer boundary modules. They are harmless if ignored by older modules,
        # but the current train_deit_boundary_new_boundaries.py consumes them for
        # residual_adapter and cls_latent_token.
        if _none_if_blank(row.get("adapter_mlp_ratio")) is not None:
            cfg["boundary"]["adapter_mlp_ratio"] = float(row.get("adapter_mlp_ratio"))
        else:
            cfg["boundary"].pop("adapter_mlp_ratio", None)

        if _none_if_blank(row.get("adapter_gamma_init")) is not None:
            cfg["boundary"]["adapter_gamma_init"] = float(row.get("adapter_gamma_init"))
        else:
            cfg["boundary"].pop("adapter_gamma_init", None)

        if _none_if_blank(row.get("adapter_dropout")) is not None:
            cfg["boundary"]["adapter_dropout"] = float(row.get("adapter_dropout"))
        else:
            cfg["boundary"].pop("adapter_dropout", None)

        if _none_if_blank(row.get("latent_tokens")) is not None:
            cfg["boundary"]["latent_tokens"] = int(row.get("latent_tokens"))
        else:
            cfg["boundary"].pop("latent_tokens", None)

        if _none_if_blank(row.get("prefix_tokens")) is not None:
            cfg["boundary"]["prefix_tokens"] = int(row.get("prefix_tokens"))
        else:
            cfg["boundary"].pop("prefix_tokens", None)

        if _none_if_blank(row.get("num_heads")) is not None:
            cfg["boundary"]["num_heads"] = int(row.get("num_heads"))
        else:
            cfg["boundary"].pop("num_heads", None)

        if _none_if_blank(row.get("attn_dropout")) is not None:
            cfg["boundary"]["attn_dropout"] = float(row.get("attn_dropout"))
        else:
            cfg["boundary"].pop("attn_dropout", None)
    else:
        # Keep the YAML clean for baseline.
        cfg["boundary"] = {"enabled": False}

    cfg.setdefault("train", {})
    cfg["train"]["epochs"] = int(row.get("epochs") or 20)
    cfg["train"]["batch_size"] = int(row.get("batch_size") or 128)
    cfg["train"]["num_workers"] = int(row.get("num_workers") or 4)
    cfg["train"]["amp"] = _as_bool(row.get("amp"), True)
    if _none_if_blank(row.get("grad_clip")) is not None:
        cfg["train"]["grad_clip"] = float(row.get("grad_clip"))

    if mode == "boundary":
        cfg["train"]["freeze_prefix"] = _as_bool(row.get("freeze_prefix"), True)
        cfg["train"]["unfreeze_tail"] = _as_bool(row.get("unfreeze_tail"), True)
        cfg["train"]["unfreeze_head"] = _as_bool(row.get("unfreeze_head"), True)
        cfg["train"]["unfreeze_boundary"] = _as_bool(row.get("unfreeze_boundary"), True)
    else:
        # Baseline-specific field supported by the training script.
        cfg["train"]["head_only"] = False
        for k in ["freeze_prefix", "unfreeze_tail", "unfreeze_head", "unfreeze_boundary"]:
            cfg["train"].pop(k, None)

    cfg.setdefault("optimizer", {})
    cfg["optimizer"]["name"] = row.get("optimizer") or "adamw"
    cfg["optimizer"]["lr"] = float(row.get("lr") or 5e-5)
    cfg["optimizer"]["weight_decay"] = float(row.get("weight_decay") or 0.05)

    cfg.setdefault("scheduler", {})
    cfg["scheduler"]["name"] = row.get("scheduler") or "cosine"
    cfg["scheduler"]["eta_min"] = float(row.get("eta_min") or 1e-6)

    cfg.setdefault("checkpoint", {})
    if mode == "boundary":
        cfg["checkpoint"]["load"] = _none_if_blank(row.get("baseline_checkpoint"))
    else:
        cfg["checkpoint"]["load"] = None
    cfg["checkpoint"]["out_dir"] = row.get("checkpoint_out_dir") or "./checkpoints"
    cfg["checkpoint"]["name"] = run_name
    cfg["checkpoint"]["save_best"] = True

    cfg.setdefault("dump_features", {})
    dump_enabled = _as_bool(row.get("dump_enabled"), mode == "boundary")
    cfg["dump_features"]["enabled"] = dump_enabled
    cfg["dump_features"]["path"] = f"./features/{run_name}.pt"
    cfg["dump_features"]["max_batches"] = int(row.get("dump_max_batches") or 20)

    return cfg


def write_slurm_job(
    row: Dict[str, Any],
    yaml_path: str,
    job_dir: str = DEFAULT_JOB_DIR,
    log_dir: str = DEFAULT_LOG_DIR,
    train_script: str = DEFAULT_TRAIN_SCRIPT,
) -> str:
    _mkdir(job_dir)
    _mkdir(log_dir)

    run_name = _make_run_name(row)
    job_name = run_name[:80]
    job_path = str(Path(job_dir) / f"{run_name}.sh")

    account = row.get("account") or DEFAULT_ACCOUNT
    job_time = row.get("job_time") or DEFAULT_TIME
    cpus = int(row.get("cpus") or DEFAULT_CPUS)
    mem = row.get("mem") or DEFAULT_MEM
    gpus = int(row.get("gpus") or DEFAULT_GPUS)

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --nodes=1",
        f"#SBATCH --gpus-per-node={gpus}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={job_time}",
        f"#SBATCH --account={account}",
        f"#SBATCH --output={log_dir}/%x_%j.out",
        f"#SBATCH --error={log_dir}/%x_%j.err",
        "",
        "set -e",
        "",
        *DEFAULT_MODULE_LINES,
        "",
        DEFAULT_ENV_ACTIVATE,
        "",
        *DEFAULT_EXPORT_LINES,
        "",
        f"echo '[Info] Running {run_name}'",
        f"echo '[Info] Config: {yaml_path}'",
        f"python {train_script} --config {yaml_path}",
        "",
    ]

    with open(job_path, "w") as f:
        f.write("\n".join(lines))
    os.chmod(job_path, 0o755)
    return job_path


# -----------------------------------------------------------------------------
# Function 2: read Excel rows, write YAML, write Slurm jobs, submit
# -----------------------------------------------------------------------------

def submit_jobs_from_excel(
    filename: str = DEFAULT_EXCEL,
    row_min: int = 2,
    row_max: Optional[int] = None,
    template_yaml: str = DEFAULT_TEMPLATE_YAML,
    config_dir: str = DEFAULT_CONFIG_DIR,
    job_dir: str = DEFAULT_JOB_DIR,
    log_dir: str = DEFAULT_LOG_DIR,
    train_script: str = DEFAULT_TRAIN_SCRIPT,
    dry_run: bool = False,
    resubmit: bool = False,
) -> None:
    """
    For each selected Excel row:
      1. read sweep parameters;
      2. write a generated YAML config;
      3. write a Slurm job script;
      4. submit with sbatch unless dry_run=True;
      5. write yaml_path, job_path, job_id, status back to Excel.
    """
    wb = openpyxl.load_workbook(filename)
    ws = wb["jobs"] if "jobs" in wb.sheetnames else wb.active
    h = _header_map(ws)
    template = _read_yaml(template_yaml)

    row_max = row_max or ws.max_row
    _mkdir(config_dir)
    _mkdir(job_dir)
    _mkdir(log_dir)

    for row_num in range(row_min, row_max + 1):
        row = _row_to_dict(ws, row_num, h)
        submit = _as_bool(row.get("submit"), True)
        existing_job_id = _none_if_blank(row.get("job_id"))

        if not submit:
            print(f"[Skip] Row {row_num}: submit=False")
            continue
        if existing_job_id is not None and not resubmit:
            print(f"[Skip] Row {row_num}: already has job_id={existing_job_id}")
            continue

        run_name = _make_run_name(row)
        cfg = build_yaml_from_row(row, template)
        yaml_path = str(Path(config_dir) / f"{run_name}.yaml")
        _write_yaml(cfg, yaml_path)

        job_path = write_slurm_job(
            row=row,
            yaml_path=yaml_path,
            job_dir=job_dir,
            log_dir=log_dir,
            train_script=train_script,
        )

        _set(ws, row_num, h, "run_name", run_name)
        _set(ws, row_num, h, "yaml_path", yaml_path)
        _set(ws, row_num, h, "job_path", job_path)

        if dry_run:
            _set(ws, row_num, h, "status", "dry_run")
            print(f"[Dry-run] Row {row_num}: wrote {yaml_path} and {job_path}")
            continue

        print(f"[Submit] Row {row_num}: {run_name}")
        res = subprocess.run(["sbatch", job_path], capture_output=True, text=True)
        stdout = res.stdout.strip()
        stderr = res.stderr.strip()
        print(stdout)
        if stderr:
            print(stderr)

        if res.returncode != 0:
            _set(ws, row_num, h, "status", "submit_failed")
            _set(ws, row_num, h, "notes", stderr[:250])
            continue

        m = re.search(r"Submitted batch job\s+(\d+)", stdout)
        job_id = m.group(1) if m else stdout.split()[-1]
        _set(ws, row_num, h, "job_id", job_id)
        _set(ws, row_num, h, "status", "submitted")

    wb.save(filename)
    print(f"[Info] Updated Excel: {filename}")


# -----------------------------------------------------------------------------
# Optional result collection
# -----------------------------------------------------------------------------

def _parse_acc_from_log(log_path: str | Path) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (last_val_acc, best_acc) in fraction units, e.g., 0.9668.
    Compatible with log lines like:
      [Epoch 1] ... val_acc=96.68%
      [Info] Best val acc: 96.68%
    """
    if not Path(log_path).exists():
        return None, None

    last_val_acc = None
    best_acc = None
    with open(log_path, "r", encoding="latin-1") as f:
        for line in f:
            m = re.search(r"val_acc=([0-9.]+)%", line)
            if m:
                last_val_acc = float(m.group(1)) / 100.0
            m = re.search(r"Best val acc:\s*([0-9.]+)%", line)
            if m:
                best_acc = float(m.group(1)) / 100.0
    return last_val_acc, best_acc


def collect_results_from_logs(
    filename: str = DEFAULT_EXCEL,
    row_min: int = 2,
    row_max: Optional[int] = None,
    log_dir: str = DEFAULT_LOG_DIR,
) -> None:
    wb = openpyxl.load_workbook(filename)
    ws = wb["jobs"] if "jobs" in wb.sheetnames else wb.active
    h = _header_map(ws)
    row_max = row_max or ws.max_row

    for row_num in range(row_min, row_max + 1):
        row = _row_to_dict(ws, row_num, h)
        job_id = _none_if_blank(row.get("job_id"))
        run_name = _none_if_blank(row.get("run_name"))
        if job_id is None or run_name is None:
            print(f"[Skip] Row {row_num}: missing job_id or run_name")
            continue

        # Slurm output format: logs/%x_%j.out, where %x is job name = truncated run_name.
        candidates = list(Path(log_dir).glob(f"{str(run_name)[:80]}_{job_id}.out"))
        if not candidates:
            # Fallback for default Slurm log names or manually moved logs.
            candidates = list(Path(".").glob(f"slurm-{job_id}.out"))
        if not candidates:
            print(f"[Skip] Row {row_num}: log not found for job_id={job_id}")
            continue

        log_path = candidates[0]
        last_val_acc, best_acc = _parse_acc_from_log(log_path)
        if last_val_acc is not None:
            _set(ws, row_num, h, "last_val_acc", last_val_acc)
        if best_acc is not None:
            _set(ws, row_num, h, "best_acc", best_acc)
            _set(ws, row_num, h, "status", "done")
        else:
            _set(ws, row_num, h, "status", "log_found_no_best_acc")

        print(f"[Result] Row {row_num}: log={log_path}, last={last_val_acc}, best={best_acc}")

    wb.save(filename)
    print(f"[Info] Updated Excel with results: {filename}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true", help="Create sweep Excel")
    parser.add_argument("--submit", action="store_true", help="Generate YAML/jobs and submit")
    parser.add_argument("--collect", action="store_true", help="Collect accuracy from logs")
    parser.add_argument("--excel", type=str, default=DEFAULT_EXCEL)
    parser.add_argument("--template-yaml", type=str, default=DEFAULT_TEMPLATE_YAML)
    parser.add_argument("--row-min", type=int, default=2)
    parser.add_argument("--row-max", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resubmit", action="store_true")
    parser.add_argument("--datasets", type=str, default="cifar100", help="Comma-separated, e.g. cifar10,cifar100")
    args = parser.parse_args()

    if args.create:
        datasets = tuple(x.strip() for x in args.datasets.split(",") if x.strip())
        create_sweep_excel(filename=args.excel, datasets=datasets)

    if args.submit:
        submit_jobs_from_excel(
            filename=args.excel,
            row_min=args.row_min,
            row_max=args.row_max,
            template_yaml=args.template_yaml,
            dry_run=args.dry_run,
            resubmit=args.resubmit,
        )

    if args.collect:
        collect_results_from_logs(
            filename=args.excel,
            row_min=args.row_min,
            row_max=args.row_max,
        )


if __name__ == "__main__":
    main()
