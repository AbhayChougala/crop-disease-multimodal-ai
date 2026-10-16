from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from crop_ai.rag.base import Retriever
from crop_ai.rag.contracts import RetrieverFilters
from crop_ai.rag.factory import load_retriever


def load_cases(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            cases = list(csv.DictReader(handle))
        for case in cases:
            case["expected_document_ids"] = [
                value.strip()
                for value in case.get("expected_document_ids", "").split("|")
                if value.strip()
            ]
    else:
        cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Retrieval evaluation cases must be a non-empty CSV or JSON list.")
    return cases


def evaluate_retriever(
    retriever: Retriever,
    cases: list[dict[str, Any]],
    *,
    top_k: int = 5,
    retriever_name: str = "unknown",
) -> dict[str, object]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        expected = set(case.get("expected_document_ids", []))
        if not expected and case.get("expected_document_id"):
            expected = {str(case["expected_document_id"])}
        if not expected:
            raise ValueError("Each case requires one or more expected document IDs.")
        results = retriever.search(
            str(case["query"]),
            top_k,
            RetrieverFilters(
                crop=case.get("crop") or None,
                disease=case.get("disease") or None,
            ),
        )
        ranked_ids = list(dict.fromkeys(item.document_id for item in results))
        relevant = [index + 1 for index, item in enumerate(ranked_ids) if item in expected]
        retrieved_relevant = len(set(ranked_ids) & expected)
        rows.append(
            {
                "case_id": case.get("case_id"),
                "query": case["query"],
                "expected_document_ids": sorted(expected),
                "retrieved_document_ids": ranked_ids,
                "precision_at_k": retrieved_relevant / top_k,
                "recall_at_k": retrieved_relevant / len(expected),
                "hit_at_k": bool(relevant),
                "reciprocal_rank": 1 / min(relevant) if relevant else 0.0,
            }
        )
    count = len(rows)
    unique_expected = {item for row in rows for item in row["expected_document_ids"]}
    unique_retrieved_relevant = {
        item
        for row in rows
        for item in row["retrieved_document_ids"]
        if item in row["expected_document_ids"]
    }
    return {
        "evaluation_status": (
            "SAMPLE_FIXTURE_ONLY"
            if all(
                str(case.get("is_sample", "")).casefold() in {"true", "1", "yes"} for case in cases
            )
            else "MEASURED_ON_DECLARED_BENCHMARK"
        ),
        "retriever": retriever_name,
        "case_count": count,
        "top_k": top_k,
        "precision_at_k": sum(row["precision_at_k"] for row in rows) / count,
        "recall_at_k": sum(row["recall_at_k"] for row in rows) / count,
        "hit_rate_at_k": sum(row["hit_at_k"] for row in rows) / count,
        "mean_reciprocal_rank": sum(row["reciprocal_rank"] for row in rows) / count,
        "source_coverage": len(unique_retrieved_relevant) / len(unique_expected),
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a declared retrieval benchmark.")
    parser.add_argument("--retriever", choices=["tfidf", "semantic", "hybrid"], default="tfidf")
    parser.add_argument("--index", type=Path, default=Path("data/vectorstore/tfidf.joblib"))
    parser.add_argument(
        "--semantic-index", type=Path, default=Path("data/vectorstore/semantic.faiss")
    )
    parser.add_argument(
        "--semantic-metadata", type=Path, default=Path("data/vectorstore/semantic.metadata.json")
    )
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table-output", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    arguments = parser.parse_args()
    retriever = load_retriever(
        arguments.retriever,
        lexical_index=arguments.index,
        semantic_index=arguments.semantic_index,
        semantic_metadata=arguments.semantic_metadata,
        embedding_model=arguments.embedding_model,
    )
    result = evaluate_retriever(
        retriever,
        load_cases(arguments.cases),
        top_k=arguments.top_k,
        retriever_name=arguments.retriever,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if arguments.table_output:
        arguments.table_output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.table_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "case_id",
                    "query",
                    "expected_document_ids",
                    "retrieved_document_ids",
                    "precision_at_k",
                    "recall_at_k",
                    "hit_at_k",
                    "reciprocal_rank",
                ),
            )
            writer.writeheader()
            for row in result["cases"]:
                writer.writerow(
                    {
                        **row,
                        "expected_document_ids": "|".join(row["expected_document_ids"]),
                        "retrieved_document_ids": "|".join(row["retrieved_document_ids"]),
                    }
                )
    print(f"Wrote retrieval evaluation to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
