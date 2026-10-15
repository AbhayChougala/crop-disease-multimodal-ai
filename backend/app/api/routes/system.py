from typing import Annotated

from fastapi import APIRouter, Depends

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.system import CapabilitiesResponse
from backend.app.services.knowledge_service import get_knowledge_service
from backend.app.services.model_manager import get_model_manager
from crop_ai.device import get_device_info
from crop_ai.models.registry import SUPPORTED_ARCHITECTURES

router = APIRouter(tags=["system"])


@router.get("/system/capabilities", response_model=CapabilitiesResponse)
async def capabilities(
    settings: Annotated[Settings, Depends(get_settings)],
) -> CapabilitiesResponse:
    device = get_device_info()
    runtime = get_model_manager().status
    return CapabilitiesResponse(
        device=device.selected,
        cuda_available=device.cuda_available,
        mps_available=device.mps_available,
        torch_available=device.torch_available,
        model_loaded=runtime.loaded,
        model_mode=runtime.state.lower(),
        rag_available=get_knowledge_service().available,
        llm_provider=settings.llm_provider,
        gradcam_available=runtime.state == "REAL",
    )


@router.get("/models")
async def models() -> dict[str, object]:
    runtime = get_model_manager().status
    return {
        "supported_architectures": list(SUPPORTED_ARCHITECTURES),
        "runtime": runtime.__dict__,
        "softmax_notice": "Confidence is not a verified probability of correctness.",
    }
