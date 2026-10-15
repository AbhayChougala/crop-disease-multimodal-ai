from __future__ import annotations

import argparse
import json
from pathlib import Path

from crop_ai.rag.store import TfidfStore


def evaluate(index: Path, cases_path: Path, top_k: int = 4) -> dict[str, object]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Retrieval evaluation cases must be a non-empty JSON list.")
    store = TfidfStore.load(index)
    rows = []
    reciprocal_ranks = []
    for case in cases:
        expected = str(case["expected_document_id"])
        results = store.search(
            str(case["query"]),
            top_k=top_k,
            crop=case.get("crop"),
            disease=case.get("disease"),
        )
        ranked_ids = [item.document_id for item in results]
        rank = ranked_ids.index(expected) + 1 if expected in ranked_ids else None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        rows.append({"query": case["query"], "expected": expected, "rank": rank})
    return {
        "case_count": len(rows),
        "hit_rate_at_k": sum(row["rank"] is not None for row in rows) / len(rows),
        "mean_reciprocal_rank": sum(reciprocal_ranks) / len(rows),
        "top_k": top_k,
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval using documented queries.")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    arguments = parser.parse_args()
    result = evaluate(arguments.index, arguments.cases, arguments.top_k)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote measured retrieval results to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
