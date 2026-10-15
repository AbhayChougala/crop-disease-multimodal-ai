"""Run one offline CPU epoch on generated fixtures; outputs are not research results."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from crop_ai.data.manifest import scan_imagefolder, stratified_group_split, write_manifest
from crop_ai.training.config import TrainingConfig
from crop_ai.training.trainer import Trainer
from scripts.generate_synthetic_dataset import generate


def main() -> int:
    os.environ["CROP_AI_DEVICE"] = "cpu"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        images = root / "images"
        generate(images)
        records, invalid = scan_imagefolder(images, "synthetic-smoke")
        assert not invalid
        manifest = root / "manifest.csv"
        write_manifest(manifest, stratified_group_split(records, seed=42))
        config = TrainingConfig(
            experiment_name="synthetic-one-epoch",
            architecture="mobilenet_v2",
            manifest=str(manifest),
            image_root=str(images),
            output_dir=str(root / "models"),
            dataset_id="synthetic-smoke",
            dataset_fingerprint="synthetic-only",
            image_size=64,
            pretrained=False,
            freeze_backbone=True,
            batch_size=2,
            epochs=1,
            learning_rate=0.001,
            workers=0,
            mixed_precision=False,
            augmentations={"rotation_degrees": 0, "color_jitter": 0},
        )
        history = Trainer(config).fit()
        assert len(history) == 1
        assert (root / "models" / "synthetic-one-epoch" / "best.pt").is_file()
    print("One-epoch CPU training smoke test passed; synthetic metrics were discarded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
