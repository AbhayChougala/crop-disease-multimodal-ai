from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from crop_ai.data.manifest import ManifestRecord, read_manifest


class ManifestDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        manifest_path: Path,
        image_root: Path,
        split: str,
        transform: Callable,
        class_to_index: dict[str, int] | None = None,
    ) -> None:
        self.records = [item for item in read_manifest(manifest_path) if item.split == split]
        if not self.records:
            raise ValueError(f"Manifest has no records for split '{split}'.")
        classes = sorted({item.class_id for item in read_manifest(manifest_path)})
        self.class_to_index = class_to_index or {name: index for index, name in enumerate(classes)}
        unknown = {item.class_id for item in self.records} - set(self.class_to_index)
        if unknown:
            raise ValueError(f"Manifest contains classes missing from mapping: {sorted(unknown)}")
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record: ManifestRecord = self.records[index]
        path = (self.image_root / record.image_path).resolve()
        root = self.image_root.resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Manifest path escapes dataset root: {record.image_path}")
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.class_to_index[record.class_id]
