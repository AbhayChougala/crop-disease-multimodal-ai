from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from crop_ai.device import get_device
from crop_ai.segmentation.dataset import SegmentationDataset
from crop_ai.segmentation.losses import BCEDiceLoss
from crop_ai.segmentation.metrics import segmentation_metrics
from crop_ai.segmentation.unet import UNet
from crop_ai.training.reproducibility import set_reproducible_seed


def _load_config(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Segmentation config must be a YAML mapping.")
    return payload


def _epoch(model, loader, loss_fn, device, optimizer=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    metric_total = {"iou": 0.0, "dice": 0.0}
    samples = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch = images.shape[0]
        metrics = segmentation_metrics(torch.sigmoid(logits.detach()), targets)
        loss_total += float(loss.item()) * batch
        metric_total["iou"] += metrics["iou"] * batch
        metric_total["dice"] += metrics["dice"] * batch
        samples += batch
    return {
        "loss": loss_total / samples,
        "iou": metric_total["iou"] / samples,
        "dice": metric_total["dice"] / samples,
    }


def train(config_path: Path) -> list[dict[str, object]]:
    config = _load_config(config_path)
    set_reproducible_seed(int(config.get("seed", 42)))
    device = get_device()
    if not isinstance(device, torch.device):
        device = torch.device("cpu")
    root = Path(config["image_root"])
    manifest = Path(config["manifest"])
    image_size = int(config.get("image_size", 256))
    train_data = SegmentationDataset(manifest, root, "train", image_size)
    validation_data = SegmentationDataset(manifest, root, "validation", image_size)
    batch_size = int(config.get("batch_size", 8))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size)
    base = int(config.get("base_channels", 32))
    model = UNet(base=base).to(device)
    optimizer = AdamW(model.parameters(), lr=float(config.get("learning_rate", 0.0005)))
    loss_fn = BCEDiceLoss(float(config.get("dice_weight", 0.5)))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")
    for epoch in range(1, int(config.get("epochs", 30)) + 1):
        row = {
            "epoch": epoch,
            "train": _epoch(model, train_loader, loss_fn, device, optimizer),
            "validation": _epoch(model, validation_loader, loss_fn, device),
        }
        history.append(row)
        if row["validation"]["loss"] < best_loss:
            best_loss = row["validation"]["loss"]
            torch.save(
                {
                    "format_version": 1,
                    "architecture": "unet",
                    "state_dict": model.state_dict(),
                    "base_channels": base,
                    "image_size": image_size,
                    "validated": False,
                    "validation_metrics": row["validation"],
                    "config": config,
                },
                output / "best.pt",
            )
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


def main() -> int:
    parser = argparse.ArgumentParser(description="Train lesion U-Net from paired masks.")
    parser.add_argument("--config", type=Path, default=Path("configs/training/unet.yaml"))
    arguments = parser.parse_args()
    train(arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
