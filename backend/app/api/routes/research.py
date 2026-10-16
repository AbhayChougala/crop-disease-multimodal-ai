import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["research"])


@router.get("/research/rag-results")
async def rag_results() -> dict[str, Any]:
    root = Path("research/results")
    results = []
    for path in sorted(root.glob("rag_retrieval_*.json")) + sorted(root.glob("sample_*_rag.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            results.append(
                {
                    "file": path.name,
                    "evaluation_status": payload.get("evaluation_status"),
                    "retriever": payload.get("retriever"),
                    "case_count": payload.get("case_count"),
                    "top_k": payload.get("top_k"),
                    "precision_at_k": payload.get("precision_at_k"),
                    "recall_at_k": payload.get("recall_at_k"),
                    "hit_rate_at_k": payload.get("hit_rate_at_k"),
                    "mean_reciprocal_rank": payload.get("mean_reciprocal_rank"),
                    "source_coverage": payload.get("source_coverage"),
                }
            )
    return {"status": "results_available" if results else "no_results", "results": results}


@router.get("/research/summary")
async def research_summary() -> dict[str, Any]:
    summary_path = Path("research/results/summary.json")
    if summary_path.is_file():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "status": "no_results",
        "metrics": [],
        "message": "No validated experiment result files are available yet.",
        "planned_experiments": [
            "model comparison",
            "preprocessing ablation",
            "cross-domain robustness",
            "Grad-CAM qualitative analysis",
            "severity validation",
            "RAG versus no-RAG",
            "realtime latency",
        ],
    }
