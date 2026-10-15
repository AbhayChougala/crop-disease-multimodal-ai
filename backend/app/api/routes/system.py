from typing import Annotated

from fastapi import APIRouter, Depends

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.system import CapabilitiesResponse
from crop_ai.device import get_device_info

router = APIRouter(tags=["system"])


@router.get("/system/capabilities", response_model=CapabilitiesResponse)
async def capabilities(
    settings: Annotated[Settings, Depends(get_settings)],
) -> CapabilitiesResponse:
    device = get_device_info()
    model_loaded = settings.model_mode == "real" and settings.model_path.exists()
    return CapabilitiesResponse(
        device=device.selected,
        cuda_available=device.cuda_available,
        mps_available=device.mps_available,
        torch_available=device.torch_available,
        model_loaded=model_loaded,
        model_mode=settings.model_mode,
        rag_available=False,
        llm_provider=settings.llm_provider,
        gradcam_available=model_loaded,
    )
