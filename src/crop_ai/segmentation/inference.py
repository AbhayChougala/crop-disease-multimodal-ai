from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional

from crop_ai.segmentation.unet import UNet


class SegmentationUnavailable(RuntimeError):
    pass


class LesionSegmenter:
    def __init__(self, model: UNet, device: torch.device, image_size: int = 256) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.image_size = image_size

    @classmethod
    def from_checkpoint(cls, path: Path, device: torch.device) -> LesionSegmenter:
        if not path.is_file():
            raise SegmentationUnavailable(f"Segmentation checkpoint does not exist: {path}")
        try:
            payload = torch.load(path, map_location=device, weights_only=True)
            if payload.get("validated") is not True:
                raise SegmentationUnavailable(
                    "Segmentation checkpoint is not marked as evaluated against ground-truth masks."
                )
            model = UNet(base=int(payload.get("base_channels", 32)))
            model.load_state_dict(payload["state_dict"], strict=True)
            return cls(model, device, int(payload.get("image_size", 256)))
        except SegmentationUnavailable:
            raise
        except Exception as exc:
            raise SegmentationUnavailable(
                f"Segmentation checkpoint is incompatible: {exc}"
            ) from exc

    def predict(self, image_bgr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((self.image_size, self.image_size))
        tensor = functional.to_tensor(pil).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probability = torch.sigmoid(self.model(tensor))[0, 0]
        mask = np.uint8(probability.cpu().numpy() >= threshold) * 255
        return cv2.resize(
            mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST
        )
