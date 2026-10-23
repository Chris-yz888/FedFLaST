"""Shared final-checkpoint writer for post-training analysis.

This helper is intentionally opt-in.  Normal training behavior is unchanged
unless ``--save_final_checkpoint true`` is passed by an experiment script.
"""

from __future__ import annotations

import copy
import logging
import os

import torch


def save_final_server_checkpoint(args, aggregator, method: str, round_num: int) -> None:
    """Persist the final server model and sparse mask, when requested."""
    if not getattr(args, "save_final_checkpoint", False):
        return

    checkpoint_path = getattr(args, "checkpoint_path", None)
    if not checkpoint_path:
        density = float(getattr(args, "target_density", 1.0))
        checkpoint_path = os.path.join(
            "checkpoints",
            "{}_{}_{}_den{:03d}_seed{}.pt".format(
                method.lower(),
                args.dataset,
                args.model,
                int(round(density * 100)),
                getattr(args, "seed", 0),
            ),
        )

    checkpoint_path = os.path.abspath(checkpoint_path)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    model = aggregator.trainer.model
    if hasattr(model, "apply_mask"):
        model.apply_mask()

    payload = {
        "format_version": 1,
        "method": method,
        "round": round_num,
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "args": dict(vars(args)),
    }
    mask_dict = getattr(model, "mask_dict", None)
    if isinstance(mask_dict, dict):
        payload["mask_dict"] = {
            key: value.detach().cpu().clone()
            for key, value in mask_dict.items()
        }
    layer_density_dict = getattr(model, "layer_density_dict", None)
    if layer_density_dict is not None:
        payload["layer_density_dict"] = copy.deepcopy(layer_density_dict)

    torch.save(payload, checkpoint_path)
    logging.info("[checkpoint] saved final %s model to %s", method, checkpoint_path)
