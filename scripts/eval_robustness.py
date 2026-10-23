#!/usr/bin/env python3
"""Evaluate dense or sparse checkpoints on the standard 15 CIFAR-10-C corruptions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from checkpoint_utils import build_model, load_checkpoint


CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="LABEL:checkpoint.pt; repeat for each model")
    parser.add_argument("--data_dir", required=True,
                        help="extracted CIFAR-10-C directory containing labels.npy")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def parse_spec(spec: str) -> tuple[str, Path]:
    label, path = spec.split(":", 1)
    return label, Path(path).resolve()


@torch.inference_mode()
def evaluate_array(model, images: np.ndarray, labels: np.ndarray,
                   batch_size: int, device: torch.device) -> tuple[float, float]:
    mean = torch.tensor([0.49139968, 0.48215827, 0.44653124], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.24703233, 0.24348505, 0.26158768], device=device).view(1, 3, 1, 1)
    total_loss = 0.0
    total_correct = 0
    total = int(len(labels))
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        array = np.asarray(images[start:end])
        x = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
        x.div_(255.0).sub_(mean).div_(std)
        y = torch.from_numpy(np.asarray(labels[start:end]).copy()).to(device=device, dtype=torch.long)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        total_correct += int(logits.argmax(dim=1).eq(y).sum().item())
    return total_correct / total, total_loss / total


def label_slice(labels: np.ndarray, severity: int) -> np.ndarray:
    if len(labels) == 50000:
        start = (severity - 1) * 10000
        return labels[start:start + 10000]
    if len(labels) == 10000:
        return labels
    raise ValueError(f"unexpected CIFAR-10-C labels shape: {labels.shape}")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    labels = np.load(data_dir / "labels.npy", mmap_mode="r")

    rows = []
    for label, checkpoint_path in [parse_spec(spec) for spec in args.checkpoint]:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint.get("args", {}).get("dataset") != "cifar10":
            raise ValueError(f"{checkpoint_path} is not a CIFAR10 checkpoint")
        model = build_model(checkpoint, device)
        for corruption in CORRUPTIONS:
            corruption_path = data_dir / f"{corruption}.npy"
            if not corruption_path.exists():
                raise FileNotFoundError(corruption_path)
            images = np.load(corruption_path, mmap_mode="r")
            if len(images) != 50000:
                raise ValueError(f"unexpected {corruption} shape: {images.shape}")
            for severity in range(1, 6):
                start = (severity - 1) * 10000
                accuracy, loss = evaluate_array(
                    model,
                    images[start:start + 10000],
                    label_slice(labels, severity),
                    args.batch_size,
                    device,
                )
                rows.append({
                    "model": label,
                    "method": checkpoint.get("method", "unknown"),
                    "density": checkpoint.get("args", {}).get("target_density", 1.0),
                    "seed": checkpoint.get("args", {}).get("seed", 0),
                    "corruption": corruption,
                    "severity": severity,
                    "accuracy": accuracy,
                    "loss": loss,
                    "checkpoint": str(checkpoint_path),
                })
                print(f"[{label}] {corruption} severity={severity}: acc={accuracy:.4f} loss={loss:.4f}", flush=True)

    fields = list(rows[0].keys())
    with (output_dir / "cifar10c_per_corruption.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    labels_seen = list(dict.fromkeys(row["model"] for row in rows))
    for label in labels_seen:
        subset = [row for row in rows if row["model"] == label]
        summary_rows.append({
            "model": label,
            "severity": "all",
            "mean_corruption_accuracy": float(np.mean([row["accuracy"] for row in subset])),
            "mean_corruption_loss": float(np.mean([row["loss"] for row in subset])),
        })
        for severity in range(1, 6):
            sev = [row for row in subset if row["severity"] == severity]
            summary_rows.append({
                "model": label,
                "severity": severity,
                "mean_corruption_accuracy": float(np.mean([row["accuracy"] for row in sev])),
                "mean_corruption_loss": float(np.mean([row["loss"] for row in sev])),
            })

    with (output_dir / "cifar10c_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# CIFAR-10-C checkpoint evaluation",
        "",
        "Metric: mean corruption accuracy over the standard 15 corruptions; this is not normalized mCE.",
        "",
        "| model | severity | mean corruption accuracy | mean corruption loss |",
        "|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | {row['severity']} | "
            f"{row['mean_corruption_accuracy']:.2%} | {row['mean_corruption_loss']:.4f} |"
        )
    (output_dir / "cifar10c_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {output_dir}")


if __name__ == "__main__":
    main()
