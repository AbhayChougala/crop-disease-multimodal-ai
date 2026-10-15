import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["research"])


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
