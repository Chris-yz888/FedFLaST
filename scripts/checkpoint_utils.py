"""Utilities for loading FedFLaST/FedDST dense or sparse checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint produced by ``--save_final_checkpoint true``."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    required = {"model_state_dict", "args"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"checkpoint {path} misses keys: {missing}")
    return checkpoint


def build_model(checkpoint: dict, device: torch.device):
    """Rebuild the trained model and restore its final sparse mask, if present."""
    from api.model.cv.resnet import resnet18, resnet56
    from api.pruning.model_pruning import SparseModel

    saved_args = checkpoint.get("args", {})
    model_name = saved_args.get("model", "resnet18")
    dataset_name = saved_args.get("dataset", "cifar10")
    if dataset_name not in {"cifar10", "svhn"}:
        raise ValueError(f"CIFAR-10-C evaluation requires a CIFAR-10 checkpoint, got {dataset_name}")

    if model_name == "resnet18":
        inner_model = resnet18(class_num=10)
    elif model_name == "resnet56":
        inner_model = resnet56(class_num=10)
    else:
        raise ValueError(f"unsupported model: {model_name}")

    mask_dict = checkpoint.get("mask_dict")
    if mask_dict is None:
        model = inner_model
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.mask_dict = {}
    else:
        model = SparseModel(
            inner_model,
            target_density=float(saved_args.get("target_density", 1.0)),
            strategy=saved_args.get("pruning_strategy", "ERK_magnitude"),
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.mask_dict = {
            name: mask.detach().to(device=device, dtype=torch.float32)
            for name, mask in mask_dict.items()
        }

    model.to(device)
    if hasattr(model, "apply_mask"):
        model.apply_mask()
    model.eval()
    return model
