from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from crop_ai.rag.store import TfidfStore, load_chunks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or search the local TF-IDF knowledge index."
    )
    parser.add_argument("--chunks", type=Path, default=Path("data/vectorstore/chunks.json"))
    parser.add_argument("--index", type=Path, default=Path("data/vectorstore/tfidf.joblib"))
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--query")
    parser.add_argument("--crop")
    parser.add_argument("--disease")
    parser.add_argument("--top-k", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.build:
        store = TfidfStore.build(load_chunks(arguments.chunks))
        store.save(arguments.index)
        print(f"Wrote local retrieval index to {arguments.index}")
        return 0
    if not arguments.query:
        parser.error("Provide --build or --query.")
    passages = TfidfStore.load(arguments.index).search(
        arguments.query,
        arguments.top_k,
        arguments.crop,
        arguments.disease,
    )
    print(json.dumps([asdict(item) for item in passages], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
