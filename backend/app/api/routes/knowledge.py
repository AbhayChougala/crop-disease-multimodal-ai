from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from backend.app.services.knowledge_service import get_knowledge_service

router = APIRouter(tags=["knowledge"])


@router.get("/knowledge/search")
def search_knowledge(
    query: Annotated[str, Query(min_length=2, max_length=500)],
    crop: Annotated[str | None, Query(max_length=100)] = None,
    disease: Annotated[str | None, Query(max_length=100)] = None,
) -> dict[str, object]:
    service = get_knowledge_service()
    try:
        passages = service.search(query, crop=crop, disease=disease)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "available" if passages else "no_relevant_source",
        "passages": [asdict(item) for item in passages],
    }
