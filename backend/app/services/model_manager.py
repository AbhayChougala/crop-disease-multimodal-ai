from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Any, Literal

import numpy as np

from backend.app.core.config import get_settings
from crop_ai.models.mock_classifier import MockClassifier

RuntimeState = Literal["DEMO", "REAL", "UNAVAILABLE"]


@dataclass(frozen=True)
class RuntimeStatus:
    state: RuntimeState
    loaded: bool
    architecture: str
    checkpoint_id: str | None
    error: str | None


class ModelManager:
    """Loads a configured checkpoint once and never silently falls back to mock mode."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.mock: MockClassifier | None = None
        self.real: Any | None = None
        self.error: str | None = None
        self._inference_lock = Lock()
        if self.settings.model_mode == "mock":
            self.mock = MockClassifier()
            self.state: RuntimeState = "DEMO"
        else:
            self.state = "UNAVAILABLE"
            self._load_real()

    def _load_real(self) -> None:
        try:
            import torch

            from crop_ai.device import get_device
            from crop_ai.models.classifier import TorchClassifier

            device = get_device()
            if not isinstance(device, torch.device):
                device = torch.device("cpu")
            self.real = TorchClassifier.from_checkpoint(
                self.settings.model_path,
                device,
                self.settings.confidence_threshold,
            )
            self.state = "REAL"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = "UNAVAILABLE"

    @property
    def status(self) -> RuntimeStatus:
        return RuntimeStatus(
            state=self.state,
            loaded=self.state == "REAL",
            architecture=(
                self.real.metadata.architecture
                if self.real is not None
                else self.settings.model_arch
            ),
            checkpoint_id=(self.real.checkpoint_id if self.real is not None else None),
            error=self.error,
        )

    def predict(self, image_bgr: np.ndarray, top_k: int = 5):
        if self.state == "DEMO" and self.mock is not None:
            return self.mock.predict(image_bgr)
        if self.state == "REAL" and self.real is not None:
            with self._inference_lock:
                return self.real.predict(image_bgr, top_k)
        raise RuntimeError(
            "Real model inference is unavailable. Fix MODEL_PATH/dependencies or explicitly "
            "set MODEL_MODE=mock for demonstration mode."
        )

    def gradcam(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float] | None:
        if self.state != "REAL" or self.real is None:
            return None
        from crop_ai.explainability.gradcam import GradCAM, overlay_heatmap
        from crop_ai.models.factory import get_target_layer

        with self._inference_lock:
            tensor = self.real._tensor(image_bgr)
            layer = get_target_layer(self.real.model, self.real.metadata.architecture)
            with GradCAM(self.real.model, layer) as cam:
                result = cam.generate(tensor)
        return overlay_heatmap(image_bgr, result.heatmap), result.target_score


@lru_cache
def get_model_manager() -> ModelManager:
    return ModelManager()
