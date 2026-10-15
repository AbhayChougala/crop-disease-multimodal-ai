from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import chat, diagnosis, feedback, health, realtime, research, system
from backend.app.core.config import get_settings
from crop_ai.device import get_device_info

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("crop-disease-api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    device = get_device_info()
    logger.info(
        "startup device=%s model_mode=%s model_loaded=%s rag_enabled=%s llm_provider=%s",
        device.selected,
        settings.model_mode,
        settings.model_mode == "real" and settings.model_path.exists(),
        settings.rag_enabled,
        settings.llm_provider,
    )
    yield


settings = get_settings()
app = FastAPI(
    title="Crop Disease Detection and Diagnosis Using Multimodal AI",
    version="0.1.0",
    description="Research prototype. Mock outputs are not diagnoses.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

routes = (
    health.router,
    system.router,
    diagnosis.router,
    chat.router,
    feedback.router,
    research.router,
)
for route in routes:
    app.include_router(route, prefix="/api/v1")
app.include_router(realtime.router, prefix="/api/v1")
