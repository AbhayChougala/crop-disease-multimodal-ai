"""Generate tiny geometric fixtures for smoke tests, never agricultural research."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def generate(output: Path, images_per_class: int = 6) -> None:
    classes = {"synthetic_green__healthy": (30, 160, 40), "synthetic_blue__spot": (160, 60, 30)}
    for class_id, color in classes.items():
        directory = output / class_id
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(images_per_class):
            image = np.full((96, 96, 3), 235, dtype=np.uint8)
            cv2.ellipse(image, (48, 48), (25 + index, 35), index * 4, 0, 360, color, -1)
            if "spot" in class_id:
                cv2.circle(image, (48, 48), 8, (30, 30, 100), -1)
            if not cv2.imwrite(str(directory / f"fixture-{index}.png"), image):
                raise RuntimeError("Synthetic image could not be written.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tmp/synthetic_dataset"))
    parser.add_argument("--images-per-class", type=int, default=6)
    arguments = parser.parse_args()
    generate(arguments.output, arguments.images_per_class)
    print(f"Synthetic-only fixtures written under {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
