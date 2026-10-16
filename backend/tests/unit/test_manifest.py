from dataclasses import replace

import pytest

from crop_ai.data.manifest import (
    ManifestRecord,
    analyze_manifest,
    canonicalize_label,
    stratified_group_split,
)


def _records() -> list[ManifestRecord]:
    return [
        ManifestRecord(
            image_path=f"tomato__early_blight/{index}.jpg",
            dataset="synthetic-test",
            crop="Tomato",
            disease="Early Blight",
            class_id="tomato__early_blight",
            source_id=str(index),
            group_id=str(index),
            sha256=f"hash-{index}",
            width=64,
            height=64,
        )
        for index in range(8)
    ]


def test_class_mapping_requires_canonical_separator() -> None:
    assert canonicalize_label("tomato__early_blight") == (
        "Tomato",
        "Early Blight",
        "tomato__early_blight",
    )
    with pytest.raises(ValueError):
        canonicalize_label("ambiguous-label")


def test_split_is_reproducible_and_keeps_all_partitions() -> None:
    first = stratified_group_split(_records(), seed=11)
    second = stratified_group_split(_records(), seed=11)

    assert first == second
    assert {item.split for item in first} == {"train", "validation", "test"}


def test_duplicate_hash_leakage_is_reported() -> None:
    records = stratified_group_split(_records(), seed=9)
    test_index = next(index for index, item in enumerate(records) if item.split == "test")
    train_item = next(item for item in records if item.split == "train")
    records[test_index] = replace(records[test_index], sha256=train_item.sha256)

    report = analyze_manifest(records)

    assert train_item.sha256 in report.duplicate_hashes
    assert train_item.sha256 in report.split_leakage
