from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import torch

from crop_ai.device import get_device
from crop_ai.models.classifier import TorchClassifier


def main() -> int:
    parser = argparse.ArgumentParser(description="Run structured inference for one image.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    arguments = parser.parse_args()
    image = cv2.imread(str(arguments.image))
    if image is None:
        parser.error(f"Image could not be decoded: {arguments.image}")
    device = get_device()
    if not isinstance(device, torch.device):
        device = torch.device("cpu")
    classifier = TorchClassifier.from_checkpoint(
        arguments.checkpoint,
        device,
        arguments.confidence_threshold,
    )
    print(json.dumps(asdict(classifier.predict(image, arguments.top_k)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
