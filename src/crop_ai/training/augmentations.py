from __future__ import annotations

from typing import Any

from torchvision import transforms

from crop_ai.models.factory import TransformConfig


def build_training_transform(
    config: TransformConfig, options: dict[str, Any]
) -> transforms.Compose:
    operations: list[Any] = [
        transforms.RandomResizedCrop(
            config.image_size,
            scale=tuple(options.get("crop_scale", (0.8, 1.0))),
        )
    ]
    if options.get("horizontal_flip", True):
        operations.append(transforms.RandomHorizontalFlip())
    if options.get("vertical_flip", False):
        operations.append(transforms.RandomVerticalFlip())
    rotation = float(options.get("rotation_degrees", 15))
    if rotation:
        operations.append(transforms.RandomRotation(rotation))
    jitter = float(options.get("color_jitter", 0.1))
    if jitter:
        operations.append(transforms.ColorJitter(jitter, jitter, jitter, min(jitter / 2, 0.2)))
    blur_probability = float(options.get("gaussian_blur_probability", 0.0))
    if blur_probability:
        operations.append(transforms.RandomApply([transforms.GaussianBlur(3)], blur_probability))
    operations.extend([transforms.ToTensor(), transforms.Normalize(config.mean, config.std)])
    return transforms.Compose(operations)
