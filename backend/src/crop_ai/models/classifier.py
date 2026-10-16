from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from crop_ai.models.checkpoints import (
    CheckpointMetadata,
    checkpoint_identifier,
    load_model_from_checkpoint,
)
from crop_ai.models.contracts import ClassifierPrediction, ClassLabel, RankedClass
from crop_ai.models.factory import TransformConfig, build_inference_transform


class InferenceError(RuntimeError):
    pass


class TorchClassifier:
    def __init__(
        self,
        model: nn.Module,
        labels: dict[int, ClassLabel],
        metadata: CheckpointMetadata,
        transform_config: TransformConfig,
        device: torch.device,
        checkpoint_id: str,
        confidence_threshold: float = 0.65,
    ) -> None:
        if len(labels) < 2:
            raise ValueError("Classifier taxonomy must contain at least two classes.")
        self.model = model.to(device).eval()
        self.labels = labels
        self.metadata = metadata
        self.transform = build_inference_transform(transform_config)
        self.device = device
        self.checkpoint_id = checkpoint_id
        self.confidence_threshold = confidence_threshold

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: torch.device,
        confidence_threshold: float = 0.65,
    ) -> TorchClassifier:
        model, labels, metadata, transform = load_model_from_checkpoint(checkpoint_path, device)
        return cls(
            model,
            labels,
            metadata,
            transform,
            device,
            checkpoint_identifier(checkpoint_path),
            confidence_threshold,
        )

    def _tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise InferenceError("Expected a three-channel BGR image.")
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

    def predict(self, image_bgr: np.ndarray, top_k: int = 5) -> ClassifierPrediction:
        tensor = self._tensor(image_bgr)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(tensor)
        if logits.ndim != 2 or logits.shape[1] != len(self.labels):
            raise InferenceError(
                f"Model returned shape {tuple(logits.shape)} for {len(self.labels)} classes."
            )
        raw_probabilities = torch.softmax(logits, dim=1)
        temperature = self.metadata.temperature
        probabilities = (
            torch.softmax(logits / temperature, dim=1)
            if temperature is not None and temperature > 0
            else raw_probabilities
        )
        count = min(max(top_k, 1), len(self.labels))
        values, indices = torch.topk(probabilities[0], count)
        raw_best = float(raw_probabilities[0, int(indices[0])].item())
        ranked = tuple(
            RankedClass(
                rank=rank,
                class_id=self.labels[int(index)].class_id,
                crop=self.labels[int(index)].crop,
                disease=self.labels[int(index)].disease,
                confidence=float(value),
            )
            for rank, (value, index) in enumerate(
                zip(values.tolist(), indices.tolist(), strict=False), start=1
            )
        )
        best = ranked[0]
        abstained = best.confidence < self.confidence_threshold
        latency = (time.perf_counter() - started) * 1000
        return ClassifierPrediction(
            crop=None if abstained else best.crop,
            disease=None if abstained else best.disease,
            class_id=None if abstained else best.class_id,
            raw_confidence=raw_best,
            calibrated_confidence=best.confidence if temperature else None,
            calibration_status="temperature_scaled" if temperature else "not_configured",
            top_k=ranked,
            architecture=self.metadata.architecture,
            model_version=self.metadata.model_version,
            checkpoint_id=self.checkpoint_id,
            training_dataset=self.metadata.training_dataset,
            runtime_device=str(self.device),
            latency_ms=round(latency, 3),
            abstained=abstained,
            abstention_reason=(
                f"Confidence {best.confidence:.3f} is below threshold "
                f"{self.confidence_threshold:.3f}."
                if abstained
                else None
            ),
        )

    def predict_batch(
        self, images_bgr: list[np.ndarray], top_k: int = 5
    ) -> list[ClassifierPrediction]:
        if not images_bgr:
            return []
        return [self.predict(image, top_k=top_k) for image in images_bgr]
