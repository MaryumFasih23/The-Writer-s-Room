from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    output_dir: Path = Field(default=Path("outputs"))
    image_backend: str = Field(default_factory=lambda: os.getenv("IMAGE_BACKEND", "auto"))
    comfyui_base_url: str | None = Field(default_factory=lambda: os.getenv("COMFYUI_BASE_URL"))
    comfyui_checkpoint: str = Field(
        default_factory=lambda: os.getenv("COMFYUI_CHECKPOINT", "v1-5-pruned-emaonly.safetensors")
    )
    sd_webui_base_url: str | None = Field(default_factory=lambda: os.getenv("SD_WEBUI_BASE_URL"))
    hf_api_token: str | None = Field(default_factory=lambda: os.getenv("HF_API_TOKEN"))
    hf_image_model: str = Field(
        default_factory=lambda: os.getenv("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
    )
    hf_image_model_fallbacks: list[str] = Field(
        default_factory=lambda: [
            item.strip()
            for item in os.getenv(
                "HF_IMAGE_MODEL_FALLBACKS",
                "black-forest-labs/FLUX.1-schnell",
            ).split(",")
            if item.strip()
        ]
    )
    groq_api_key: str | None = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    groq_model: str = Field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    image_width: int = Field(default_factory=lambda: int(os.getenv("IMAGE_WIDTH", "768")))
    image_height: int = Field(default_factory=lambda: int(os.getenv("IMAGE_HEIGHT", "768")))
    image_steps: int = Field(default_factory=lambda: int(os.getenv("IMAGE_STEPS", "28")))
    image_cfg_scale: float = Field(default_factory=lambda: float(os.getenv("IMAGE_CFG_SCALE", "7.0")))
    image_timeout_s: int = Field(default_factory=lambda: int(os.getenv("IMAGE_TIMEOUT_S", "120")))
    image_max_attempts: int = Field(default_factory=lambda: int(os.getenv("IMAGE_MAX_ATTEMPTS", "3")))

    @property
    def images_dir(self) -> Path:
        return Path("image_assets")

    @property
    def scene_manifest_path(self) -> Path:
        return self.output_dir / "scene_manifest.json"

    @property
    def character_db_path(self) -> Path:
        return self.output_dir / "character_db.json"

    @property
    def memory_store_path(self) -> Path:
        return self.output_dir / "memory_store.json"

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
