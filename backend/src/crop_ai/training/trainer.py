from __future__ import annotations

import importlib.metadata
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from crop_ai.data.manifest import canonicalize_label, read_manifest
from crop_ai.device import get_device
from crop_ai.models.checkpoints import CheckpointMetadata, read_checkpoint, save_checkpoint
from crop_ai.models.contracts import ClassLabel
from crop_ai.models.factory import (
    ArchitectureName,
    TransformConfig,
    build_inference_transform,
    create_model,
)
from crop_ai.training.augmentations import build_training_transform
from crop_ai.training.config import TrainingConfig
from crop_ai.training.dataset import ManifestDataset
from crop_ai.training.losses import create_loss
from crop_ai.training.reproducibility import set_reproducible_seed


def _packages() -> dict[str, str]:
    result = {}
    for name in ("torch", "torchvision", "numpy", "Pillow"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unavailable"
    return result


class Trainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        set_reproducible_seed(config.seed)
        device = get_device()
        if not isinstance(device, torch.device):
            device = torch.device("cpu")
        self.device = device
        self.transform_config = TransformConfig(image_size=config.image_size)
        records = read_manifest(Path(config.manifest))
        classes = sorted({item.class_id for item in records})
        self.class_to_index = {name: index for index, name in enumerate(classes)}
        self.labels = {
            index: ClassLabel(index, name, *canonicalize_label(name)[:2])
            for name, index in self.class_to_index.items()
        }
        architecture: ArchitectureName = config.architecture  # type: ignore[assignment]
        self.model = create_model(
            architecture,
            len(classes),
            config.pretrained,
            config.dropout,
            config.freeze_backbone,
            config.unfreeze_blocks,
        ).to(device)
        self.train_dataset = ManifestDataset(
            Path(config.manifest),
            Path(config.image_root),
            "train",
            build_training_transform(self.transform_config, config.augmentations),
            self.class_to_index,
        )
        self.validation_dataset = ManifestDataset(
            Path(config.manifest),
            Path(config.image_root),
            "validation",
            build_inference_transform(self.transform_config),
            self.class_to_index,
        )
        self.train_loader = self._loader(self.train_dataset, training=True)
        self.validation_loader = self._loader(self.validation_dataset, training=False)
        class_weights = self._class_weights() if config.class_weighted_loss else None
        self.loss = create_loss(
            config.loss,
            class_weights,
            config.label_smoothing,
            config.focal_gamma,
        )
        parameters = [item for item in self.model.parameters() if item.requires_grad]
        if config.optimizer == "adamw":
            self.optimizer = AdamW(
                parameters, lr=config.learning_rate, weight_decay=config.weight_decay
            )
        elif config.optimizer == "sgd":
            self.optimizer = SGD(
                parameters,
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError("optimizer must be 'adamw' or 'sgd'.")
        self.scheduler = (
            CosineAnnealingLR(self.optimizer, T_max=max(config.epochs, 1))
            if config.scheduler == "cosine"
            else ReduceLROnPlateau(
                self.optimizer, mode="min", patience=max(config.patience // 2, 1)
            )
        )
        if config.scheduler not in {"cosine", "plateau"}:
            raise ValueError("scheduler must be 'cosine' or 'plateau'.")
        self.use_amp = config.mixed_precision and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _loader(self, dataset: ManifestDataset, training: bool) -> DataLoader:
        sampler = None
        shuffle = training
        if training and self.config.class_balanced_sampling:
            counts = Counter(dataset.class_to_index[item.class_id] for item in dataset.records)
            weights = [
                1.0 / counts[dataset.class_to_index[item.class_id]] for item in dataset.records
            ]
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.workers,
            pin_memory=self.device.type == "cuda",
        )

    def _class_weights(self) -> torch.Tensor:
        counts = Counter(self.class_to_index[item.class_id] for item in self.train_dataset.records)
        weights = [
            len(self.train_dataset) / (len(counts) * counts[index]) for index in range(len(counts))
        ]
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(training)
        total_loss = 0.0
        correct = 0
        samples = 0
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        progress = tqdm(loader, desc="train" if training else "validation", leave=False)
        for step, (images, targets) in enumerate(progress, start=1):
            images, targets = images.to(self.device), targets.to(self.device)
            with (
                torch.set_grad_enabled(training),
                torch.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                ),
            ):
                logits = self.model(images)
                loss = self.loss(logits, targets)
                scaled_loss = loss / self.config.gradient_accumulation
            if training:
                self.scaler.scale(scaled_loss).backward()
                if step % self.config.gradient_accumulation == 0 or step == len(loader):
                    self.scaler.unscale_(self.optimizer)
                    if self.config.gradient_clip_norm is not None:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clip_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.item()) * targets.size(0)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            samples += targets.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")
        return {"loss": total_loss / samples, "accuracy": correct / samples}

    def _metadata(self, epoch: int, best_metric: float | None) -> CheckpointMetadata:
        return CheckpointMetadata(
            architecture=self.config.architecture,  # type: ignore[arg-type]
            model_version="1",
            training_dataset=self.config.dataset_id,
            dataset_fingerprint=self.config.dataset_fingerprint,
            seed=self.config.seed,
            epoch=epoch,
            best_metric=best_metric,
            transform=self.transform_config.as_dict(),
            training_config=self.config.as_dict(),
            package_versions=_packages(),
        )

    def fit(self, resume: Path | None = None) -> list[dict[str, Any]]:
        start_epoch = 1
        best_loss = float("inf")
        if resume:
            payload = read_checkpoint(resume, self.device)
            self.model.load_state_dict(payload["state_dict"])
            if payload.get("optimizer_state"):
                self.optimizer.load_state_dict(payload["optimizer_state"])
            start_epoch = int(payload["metadata"]["epoch"]) + 1
            best_loss = float(payload["metadata"].get("best_metric") or best_loss)
        history: list[dict[str, Any]] = []
        stale_epochs = 0
        for epoch in range(start_epoch, self.config.epochs + 1):
            started = time.perf_counter()
            training = self._run_epoch(self.train_loader, training=True)
            validation = self._run_epoch(self.validation_loader, training=False)
            if self.config.scheduler == "plateau":
                self.scheduler.step(validation["loss"])
            else:
                self.scheduler.step()
            row = {
                "epoch": epoch,
                "train": training,
                "validation": validation,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
            history.append(row)
            improved = validation["loss"] < best_loss
            if improved:
                best_loss = validation["loss"]
                stale_epochs = 0
                save_checkpoint(
                    self.output_dir / "best.pt",
                    self.model,
                    self.labels,
                    self._metadata(epoch, best_loss),
                    self.optimizer.state_dict(),
                )
            else:
                stale_epochs += 1
            save_checkpoint(
                self.output_dir / "last.pt",
                self.model,
                self.labels,
                self._metadata(epoch, best_loss),
                self.optimizer.state_dict(),
            )
            (self.output_dir / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            if stale_epochs >= self.config.patience:
                break
        return history
