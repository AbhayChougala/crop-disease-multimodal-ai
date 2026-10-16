from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from crop_ai.data.manifest import (
    analyze_manifest,
    scan_imagefolder,
    stratified_group_split,
    write_manifest,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for dataset configuration: pip install PyYAML"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(f"Dataset config does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("Dataset configuration must be a YAML mapping.")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a reproducible classification manifest.")
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        config = _load_yaml(arguments.config)
        root = Path(config["root"]).expanduser()
        records, invalid = scan_imagefolder(root, str(config["dataset"]), config.get("taxonomy"))
        split = config.get("split", {})
        records = stratified_group_split(
            records,
            seed=int(config.get("seed", 42)),
            train_ratio=float(split.get("train", 0.7)),
            validation_ratio=float(split.get("validation", 0.15)),
        )
        output = Path(config["output_manifest"])
        write_manifest(output, records)
        report = analyze_manifest(records)
        labels = {class_id: index for index, class_id in enumerate(sorted(report.classes))}
        output.with_suffix(".labels.json").write_text(
            json.dumps(labels, indent=2), encoding="utf-8"
        )
        report_path = output.with_suffix(".report.json")
        report_path.write_text(
            json.dumps(
                {**report.__dict__, "invalid_images": [item.__dict__ for item in invalid]}, indent=2
            ),
            encoding="utf-8",
        )
        print(f"Wrote {len(records)} records to {output}")
        print(f"Dataset fingerprint: {report.fingerprint}")
        return 0
    except (KeyError, ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
