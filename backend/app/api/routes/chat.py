from fastapi import APIRouter

from backend.app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    return ChatResponse(
        answer=(
            "DEMO MODE: follow-up generation is available only as an interface placeholder. "
            "No agricultural corpus was retrieved, so this response cannot provide grounded advice."
        ),
        sources=[],
        warnings=[
            "The mock chat provider does not answer agricultural questions.",
            f"Session context received for {request.session_id}.",
        ],
        is_mock=True,
    )
