from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


class GradCAMError(RuntimeError):
    pass


@dataclass(frozen=True)
class GradCAMResult:
    heatmap: np.ndarray
    target_class: int
    target_score: float
    is_untrained_demo: bool = False


class GradCAM:
    """Class-specific Grad-CAM with explicit hook lifecycle."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._capture_activation),
            target_layer.register_full_backward_hook(self._capture_gradient),
        ]
        self.closed = False

    def _capture_activation(self, _module, _inputs, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _capture_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        is_untrained_demo: bool = False,
    ) -> GradCAMResult:
        if self.closed:
            raise GradCAMError("Grad-CAM hooks have already been removed.")
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise GradCAMError("Grad-CAM currently requires a batch containing exactly one image.")
        self.activations = None
        self.gradients = None
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(input_tensor)
            if logits.ndim != 2:
                raise GradCAMError(
                    f"Expected two-dimensional logits, received {tuple(logits.shape)}."
                )
            selected = int(logits.argmax(dim=1).item()) if target_class is None else target_class
            if selected < 0 or selected >= logits.shape[1]:
                raise GradCAMError(f"Target class {selected} is outside the model output taxonomy.")
            score = logits[0, selected]
            score.backward()
        if self.activations is None or self.gradients is None:
            raise GradCAMError("Target layer did not produce both activations and gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = functional.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        minimum, maximum = cam.min(), cam.max()
        if float(maximum - minimum) > 1e-12:
            cam = (cam - minimum) / (maximum - minimum)
        else:
            cam = torch.zeros_like(cam)
        heatmap = cam.detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(heatmap).all():
            raise GradCAMError("Grad-CAM produced non-finite values.")
        return GradCAMResult(heatmap, selected, float(score.detach().cpu()), is_untrained_demo)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.closed = True

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def overlay_heatmap(
    image_bgr: np.ndarray, heatmap: np.ndarray, opacity: float = 0.45
) -> np.ndarray:
    if not 0 <= opacity <= 1:
        raise ValueError("opacity must be between zero and one.")
    if heatmap.ndim != 2:
        raise ValueError("heatmap must be two-dimensional.")
    resized = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    colored = cv2.applyColorMap(np.uint8(np.clip(resized, 0, 1) * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1 - opacity, colored, opacity, 0)
