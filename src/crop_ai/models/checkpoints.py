from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from crop_ai.models.contracts import ClassLabel, normalize_class_mapping
from crop_ai.models.factory import ArchitectureName, TransformConfig, create_model


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointMetadata:
    architecture: ArchitectureName
    model_version: str
    training_dataset: str | None
    dataset_fingerprint: str | None
    seed: int
    epoch: int
    best_metric: float | None
    transform: dict[str, Any]
    temperature: float | None = None
    training_config: dict[str, Any] = field(default_factory=dict)
    package_versions: dict[str, str] = field(default_factory=dict)


def checkpoint_identifier(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_mapping: dict[int, ClassLabel],
    metadata: CheckpointMetadata,
    optimizer_state: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "state_dict": model.state_dict(),
        "class_mapping": {str(index): label.as_dict() for index, label in class_mapping.items()},
        "metadata": asdict(metadata),
        "optimizer_state": optimizer_state,
    }
    torch.save(payload, path)


def read_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise CheckpointError(f"Checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"Checkpoint could not be read safely: {exc}") from exc
    required = {"format_version", "state_dict", "class_mapping", "metadata"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise CheckpointError(f"Checkpoint is missing required fields: {sorted(required)}")
    if payload["format_version"] != 1:
        raise CheckpointError(f"Unsupported checkpoint format: {payload['format_version']}")
    return payload


def load_model_from_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[int, ClassLabel], CheckpointMetadata, TransformConfig]:
    payload = read_checkpoint(path, device)
    try:
        metadata = CheckpointMetadata(**payload["metadata"])
        mapping = normalize_class_mapping(payload["class_mapping"])
        transform = TransformConfig(**metadata.transform)
        model = create_model(metadata.architecture, len(mapping), pretrained=False)
        model.load_state_dict(payload["state_dict"], strict=True)
    except Exception as exc:
        raise CheckpointError(f"Checkpoint metadata or weights are incompatible: {exc}") from exc
    model.to(device)
    model.eval()
    return model, mapping, metadata, transform
