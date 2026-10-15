from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ClassLabel:
    index: int
    class_id: str
    crop: str
    disease: str

    @classmethod
    def from_value(cls, index: int, value: str | dict[str, Any]) -> ClassLabel:
        if isinstance(value, str):
            parts = value.split("__", maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Class '{value}' must use the canonical 'crop__disease' format.")
            crop, disease = parts
            return cls(
                index, value, crop.replace("_", " ").title(), disease.replace("_", " ").title()
            )
        class_id = str(value.get("class_id", ""))
        crop = str(value.get("crop", ""))
        disease = str(value.get("disease", ""))
        if not class_id or not crop or not disease:
            raise ValueError(f"Class mapping entry {index} is incomplete.")
        return cls(index, class_id, crop, disease)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RankedClass:
    rank: int
    class_id: str
    crop: str
    disease: str
    confidence: float


@dataclass(frozen=True)
class ClassifierPrediction:
    crop: str | None
    disease: str | None
    class_id: str | None
    raw_confidence: float
    calibrated_confidence: float | None
    calibration_status: Literal["not_configured", "temperature_scaled"]
    top_k: tuple[RankedClass, ...]
    architecture: str
    model_version: str
    checkpoint_id: str
    training_dataset: str | None
    runtime_device: str
    latency_ms: float
    inference_mode: Literal["real"] = "real"
    abstained: bool = False
    abstention_reason: str | None = None


def normalize_class_mapping(
    mapping: dict[int | str, str | dict[str, Any]],
) -> dict[int, ClassLabel]:
    normalized = {
        int(index): ClassLabel.from_value(int(index), value) for index, value in mapping.items()
    }
    if sorted(normalized) != list(range(len(normalized))):
        raise ValueError("Class mapping indices must be contiguous and start at zero.")
    return normalized
