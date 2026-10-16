from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    experiment_name: str
    architecture: str
    manifest: str
    image_root: str
    output_dir: str
    dataset_id: str
    dataset_fingerprint: str = "pending"
    image_size: int = 224
    pretrained: bool = False
    dropout: float = 0.2
    freeze_backbone: bool = True
    unfreeze_blocks: int = 0
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 0.001
    optimizer: str = "adamw"
    weight_decay: float = 0.0001
    scheduler: str = "cosine"
    patience: int = 5
    loss: str = "cross_entropy"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weighted_loss: bool = False
    class_balanced_sampling: bool = False
    gradient_accumulation: int = 1
    gradient_clip_norm: float | None = 1.0
    mixed_precision: bool = True
    workers: int = 0
    seed: int = 42
    augmentations: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_training_config(path: Path) -> TrainingConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: install the project ML extras.") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Training config does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Training config must be a YAML mapping.")
    return TrainingConfig(**payload)
