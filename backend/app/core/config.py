from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    frontend_origin: str = "http://localhost:5173"
    model_mode: Literal["mock", "real"] = "mock"
    model_path: Path = Path("artifacts/models/best_model.pt")
    model_arch: str = "efficientnet_b0"
    model_labels_path: Path = Path("artifacts/models/labels.json")
    confidence_threshold: float = Field(default=0.65, ge=0, le=1)
    cv_preprocessing_mode: Literal["raw", "standard", "segmented"] = "standard"
    severity_mode: Literal["heuristic", "disabled"] = "heuristic"
    severity_low_max: float = Field(default=5.0, ge=0, le=100)
    severity_moderate_max: float = Field(default=20.0, ge=0, le=100)
    severity_high_max: float = Field(default=40.0, ge=0, le=100)
    rag_enabled: bool = False
    rag_collection: str = "crop_disease_knowledge"
    rag_index_path: Path = Path("data/vectorstore/tfidf.joblib")
    rag_top_k: int = Field(default=4, ge=1, le=20)
    llm_provider: Literal["mock", "openai", "ollama", "local"] = "mock"
    llm_model: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_timeout: float = Field(default=30.0, ge=1, le=300)
    openai_api_key: str = ""
    openai_model: str = ""
    local_llm_base_url: str = ""
    local_llm_model: str = ""
    max_upload_mb: int = Field(default=10, ge=1, le=50)


@lru_cache
def get_settings() -> Settings:
    return Settings()
