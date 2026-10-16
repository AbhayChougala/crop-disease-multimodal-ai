from __future__ import annotations

import argparse
import json
from pathlib import Path

from crop_ai.data.manifest import analyze_manifest, read_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a dataset manifest and leakage risks.")
    parser.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        report = analyze_manifest(read_manifest(arguments.manifest))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report.__dict__, indent=2))
    return 1 if report.split_leakage else 0


if __name__ == "__main__":
    raise SystemExit(main())
