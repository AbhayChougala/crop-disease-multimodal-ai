from __future__ import annotations

import argparse
from pathlib import Path

from crop_ai.rag.ingest import source_hash
from crop_ai.rag.manifest import read_source_manifest


def validate_manifest(manifest: Path, source_root: Path) -> list[str]:
    errors: list[str] = []
    root = source_root.resolve()
    for source in read_source_manifest(manifest):
        path = (root / source.local_file_path).resolve()
        if not path.is_relative_to(root):
            errors.append(f"{source.document_id}: path escapes source root")
        elif not path.is_file():
            errors.append(f"{source.document_id}: source file is missing")
        elif source.version_hash and source.version_hash != source_hash(path):
            errors.append(f"{source.document_id}: version hash mismatch")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local knowledge sources and hashes.")
    parser.add_argument("--manifest", type=Path, default=Path("knowledge/source_manifest.csv"))
    parser.add_argument("--source-root", type=Path, default=Path("knowledge/source_docs"))
    arguments = parser.parse_args()
    errors = validate_manifest(arguments.manifest, arguments.source_root)
    if errors:
        parser.error("; ".join(errors))
    print("Knowledge manifest is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
