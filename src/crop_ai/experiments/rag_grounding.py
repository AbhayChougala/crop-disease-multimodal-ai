from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def evaluate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Grounding evaluation requires at least one reviewed record.")
    required = {
        "correctness_score",
        "groundedness_score",
        "citation_support_score",
        "completeness_score",
        "unsafe_or_unsupported_claim",
        "reviewer_notes",
        "reviewer",
    }
    for record in records:
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Grounding record is missing human-review fields: {sorted(missing)}")
    fields = (
        "correctness_score",
        "groundedness_score",
        "citation_support_score",
        "completeness_score",
    )
    unsupported_citations = sum(
        len(set(record.get("citation_ids", [])) - set(record.get("retrieved_source_ids", [])))
        for record in records
    )
    abstention_cases = [
        record
        for record in records
        if record.get("answerability") in {"ANSWERABLE", "PARTIALLY_ANSWERABLE", "UNANSWERABLE"}
    ]
    correct_abstentions = sum(
        bool(record.get("abstained")) == (record.get("answerability") == "UNANSWERABLE")
        for record in abstention_cases
    )
    return {
        "evaluation_status": "HUMAN_REVIEWED",
        "record_count": len(records),
        "mean_scores": {
            field: sum(float(record[field]) for record in records) / len(records)
            for field in fields
        },
        "reviewers": sorted({str(record["reviewer"]) for record in records}),
        "unsupported_citation_count": unsupported_citations,
        "fabricated_citation_count": unsupported_citations,
        "abstention_accuracy": (
            correct_abstentions / len(abstention_cases) if abstention_cases else None
        ),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate human-reviewed grounding records.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    records = json.loads(arguments.input.read_text(encoding="utf-8"))
    report = evaluate_records(records)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote reviewed grounding evaluation to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
