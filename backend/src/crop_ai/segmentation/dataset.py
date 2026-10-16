from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional


class SegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, manifest: Path, root: Path, split: str, image_size: int = 256) -> None:
        if not manifest.is_file():
            raise FileNotFoundError(f"Segmentation manifest does not exist: {manifest}")
        with manifest.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        required = {"image_path", "mask_path", "split"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"Segmentation manifest requires columns: {sorted(required)}")
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No segmentation records found for split '{split}'.")
        self.root = root.resolve()
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, relative: str) -> Path:
        result = (self.root / relative).resolve()
        if not result.is_relative_to(self.root):
            raise ValueError(f"Segmentation path escapes dataset root: {relative}")
        return result

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        with Image.open(self._path(row["image_path"])) as image:
            image = image.convert("RGB").resize((self.image_size, self.image_size))
            image_tensor = functional.to_tensor(image)
        with Image.open(self._path(row["mask_path"])) as mask:
            mask = mask.convert("L").resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.NEAREST,
            )
            mask_array = np.asarray(mask)
        unique = set(np.unique(mask_array).tolist())
        if not unique.issubset({0, 1, 255}):
            raise ValueError(f"Mask contains non-binary values: {sorted(unique)[:10]}")
        mask_tensor = torch.from_numpy((mask_array > 0).astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor
