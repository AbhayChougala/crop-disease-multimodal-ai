"""Offline CPU smoke test for model, checkpoint, inference, Grad-CAM, and U-Net code."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from crop_ai.explainability.gradcam import GradCAM
from crop_ai.models.checkpoints import (
    CheckpointMetadata,
    load_model_from_checkpoint,
    save_checkpoint,
)
from crop_ai.models.classifier import TorchClassifier
from crop_ai.models.contracts import ClassLabel
from crop_ai.models.factory import TransformConfig, create_model, get_target_layer
from crop_ai.models.registry import SUPPORTED_ARCHITECTURES
from crop_ai.segmentation.unet import UNet


def main() -> int:
    torch.manual_seed(42)
    for architecture in SUPPORTED_ARCHITECTURES:
        model = create_model(architecture, 3, pretrained=False).eval()
        with torch.inference_mode():
            assert model(torch.randn(1, 3, 64, 64)).shape == (1, 3)

    model = create_model("mobilenet_v2", 2, pretrained=False)
    loss = torch.nn.functional.cross_entropy(model(torch.randn(2, 3, 64, 64)), torch.tensor([0, 1]))
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    labels = {
        0: ClassLabel(0, "tomato__healthy", "Tomato", "Healthy"),
        1: ClassLabel(1, "tomato__early_blight", "Tomato", "Early Blight"),
    }
    metadata = CheckpointMetadata(
        architecture="mobilenet_v2",
        model_version="synthetic-smoke",
        training_dataset="synthetic-smoke",
        dataset_fingerprint="synthetic",
        seed=42,
        epoch=1,
        best_metric=None,
        transform=TransformConfig(image_size=64).as_dict(),
    )
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "toy.pt"
        save_checkpoint(checkpoint, model, labels, metadata)
        restored, mapping, _, _ = load_model_from_checkpoint(checkpoint, torch.device("cpu"))
        assert len(mapping) == 2
        classifier = TorchClassifier.from_checkpoint(
            checkpoint, torch.device("cpu"), confidence_threshold=1.0
        )
        image = np.full((64, 64, 3), 120, dtype=np.uint8)
        prediction = classifier.predict(image, top_k=2)
        assert prediction.abstained and len(prediction.top_k) == 2
        tensor = classifier._tensor(image)
        with GradCAM(restored, get_target_layer(restored, "mobilenet_v2")) as gradcam:
            heatmap = gradcam.generate(tensor, is_untrained_demo=True).heatmap
        assert heatmap.shape == (64, 64) and np.isfinite(heatmap).all()

    unet_output = UNet(base=4)(torch.randn(1, 3, 65, 67))
    assert unet_output.shape == (1, 1, 65, 67)
    print("ML smoke test passed (synthetic data, weights=None, CPU).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
