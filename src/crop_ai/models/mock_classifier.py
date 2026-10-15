from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RankedPrediction:
    crop: str
    disease: str
    confidence: float

    @property
    def label(self) -> str:
        return f"{self.crop} - {self.disease}"


@dataclass(frozen=True)
class MockPrediction:
    crop: str
    disease: str
    confidence: float
    top_k: tuple[RankedPrediction, ...]
    is_mock: bool = True


class MockClassifier:
    """Deterministic UI placeholder. Outputs must never be used as research results."""

    labels = (
        ("Tomato", "Early Blight"),
        ("Tomato", "Leaf Mold"),
        ("Potato", "Late Blight"),
        ("Maize", "Common Rust"),
        ("Tomato", "Healthy"),
    )

    def predict(self, image_bgr: np.ndarray) -> MockPrediction:
        thumbnail = cv2.resize(image_bgr, (32, 32), interpolation=cv2.INTER_AREA)
        digest = hashlib.sha256(thumbnail.tobytes()).digest()
        selected = digest[0] % len(self.labels)
        ordered = [self.labels[(selected + offset) % len(self.labels)] for offset in range(3)]
        confidences = (0.74, 0.17, 0.09)
        ranked = tuple(
            RankedPrediction(crop, disease, confidence)
            for (crop, disease), confidence in zip(ordered, confidences, strict=True)
        )
        best = ranked[0]
        return MockPrediction(best.crop, best.disease, best.confidence, ranked)
