from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from PIL import Image, UnidentifiedImageError

SplitName = Literal["train", "validation", "test", "unassigned"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MANIFEST_FIELDS = (
    "image_path",
    "dataset",
    "crop",
    "disease",
    "class_id",
    "split",
    "source_id",
    "group_id",
    "sha256",
    "width",
    "height",
)


@dataclass(frozen=True)
class ManifestRecord:
    image_path: str
    dataset: str
    crop: str
    disease: str
    class_id: str
    split: SplitName = "unassigned"
    source_id: str = ""
    group_id: str = ""
    sha256: str = ""
    width: int = 0
    height: int = 0


@dataclass(frozen=True)
class InvalidImage:
    image_path: str
    reason: str


@dataclass(frozen=True)
class ManifestReport:
    records: int
    classes: dict[str, int]
    splits: dict[str, int]
    duplicate_hashes: dict[str, list[str]]
    split_leakage: dict[str, list[str]]
    fingerprint: str


def canonicalize_label(value: str) -> tuple[str, str, str]:
    normalized = value.strip().lower().replace(" ", "_").replace("___", "__")
    if "__" not in normalized:
        raise ValueError(
            f"Class folder '{value}' must use a crop__disease name or an explicit taxonomy map."
        )
    crop, disease = normalized.split("__", maxsplit=1)
    if not crop or not disease:
        raise ValueError(f"Invalid canonical label: {value}")
    return crop.replace("_", " ").title(), disease.replace("_", " ").title(), normalized


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path, max_pixels: int = 40_000_000) -> tuple[int, int, str]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Image cannot be decoded: {exc}") from exc
    if width < 32 or height < 32:
        raise ValueError("Image is smaller than 32 x 32 pixels.")
    if width * height > max_pixels:
        raise ValueError(f"Image exceeds the {max_pixels:,}-pixel safety limit.")
    return width, height, file_sha256(path)


def scan_imagefolder(
    root: Path,
    dataset: str,
    taxonomy: dict[str, str] | None = None,
) -> tuple[list[ManifestRecord], list[InvalidImage]]:
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {root}. "
            "Configure the local path before preparing it."
        )
    records: list[ManifestRecord] = []
    invalid: list[InvalidImage] = []
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories were found under {root}.")
    for class_dir in class_dirs:
        mapped = taxonomy.get(class_dir.name, class_dir.name) if taxonomy else class_dir.name
        crop, disease, class_id = canonicalize_label(mapped)
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                width, height, sha256 = inspect_image(image_path)
            except ValueError as exc:
                invalid.append(InvalidImage(str(image_path.relative_to(root)), str(exc)))
                continue
            relative = image_path.relative_to(root).as_posix()
            source_id = image_path.stem
            records.append(
                ManifestRecord(
                    image_path=relative,
                    dataset=dataset,
                    crop=crop,
                    disease=disease,
                    class_id=class_id,
                    source_id=source_id,
                    group_id=source_id,
                    sha256=sha256,
                    width=width,
                    height=height,
                )
            )
    if not records:
        raise ValueError(f"No valid images were found under {root}.")
    return records, invalid


def stratified_group_split(
    records: list[ManifestRecord],
    seed: int = 42,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
) -> list[ManifestRecord]:
    if train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Split ratios must leave non-zero train and test partitions.")
    by_class: dict[str, dict[str, list[ManifestRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        group = record.group_id or record.source_id or record.image_path
        by_class[record.class_id][group].append(record)

    assigned: list[ManifestRecord] = []
    for class_id in sorted(by_class):
        groups = list(by_class[class_id].items())
        random.Random(f"{seed}:{class_id}").shuffle(groups)
        group_count = len(groups)
        if group_count < 3:
            raise ValueError(
                f"Class '{class_id}' has {group_count} source groups; at least three are "
                "required for leakage-safe train/validation/test splits."
            )
        train_groups = max(1, round(group_count * train_ratio))
        validation_groups = max(1, round(group_count * validation_ratio))
        if train_groups + validation_groups >= group_count:
            train_groups = group_count - 2
            validation_groups = 1
        for group_index, (_, items) in enumerate(groups):
            if group_index < train_groups:
                split: SplitName = "train"
            elif group_index < train_groups + validation_groups:
                split = "validation"
            else:
                split = "test"
            assigned.extend(replace(item, split=split) for item in items)
    return sorted(assigned, key=lambda item: item.image_path)


def manifest_fingerprint(records: Iterable[ManifestRecord]) -> str:
    canonical = [asdict(record) for record in sorted(records, key=lambda item: item.image_path)]
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def analyze_manifest(records: list[ManifestRecord]) -> ManifestReport:
    hashes: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        if record.sha256:
            hashes[record.sha256].append(record)
    duplicates = {
        digest: [item.image_path for item in items]
        for digest, items in hashes.items()
        if len(items) > 1
    }
    leakage = {
        digest: sorted({item.split for item in items})
        for digest, items in hashes.items()
        if len({item.split for item in items}) > 1
    }
    return ManifestReport(
        records=len(records),
        classes=dict(sorted(Counter(item.class_id for item in records).items())),
        splits=dict(sorted(Counter(item.split for item in records).items())),
        duplicate_hashes=duplicates,
        split_leakage=leakage,
        fingerprint=manifest_fingerprint(records),
    )


def write_manifest(path: Path, records: list[ManifestRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def read_manifest(path: Path) -> list[ManifestRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(MANIFEST_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        return [
            ManifestRecord(
                **{
                    **row,
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                }
            )
            for row in reader
        ]
