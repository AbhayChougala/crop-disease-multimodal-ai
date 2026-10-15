from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["research"])


@router.get("/research/summary")
async def research_summary() -> dict[str, Any]:
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
