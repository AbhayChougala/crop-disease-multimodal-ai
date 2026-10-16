from __future__ import annotations

import argparse
import json
from pathlib import Path

from crop_ai.data.manifest import read_manifest


def class_overlap(first: Path, second: Path) -> dict[str, list[str]]:
    first_classes = {item.class_id for item in read_manifest(first)}
    second_classes = {item.class_id for item in read_manifest(second)}
    return {
        "overlap": sorted(first_classes & second_classes),
        "first_only": sorted(first_classes - second_classes),
        "second_only": sorted(second_classes - first_classes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Report compatible classes across two manifests.")
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(class_overlap(arguments.first, arguments.second), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
