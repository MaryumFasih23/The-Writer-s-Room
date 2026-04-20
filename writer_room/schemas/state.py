from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

from writer_room.schemas.screenplay import CharacterProfile, ImageAsset, SceneManifest


class WorkflowState(TypedDict, total=False):
    mode: Literal["manual", "prompt"]
    raw_script: str
    prompt_text: str
    auto_approve: bool
    output_dir: Path
    validated: bool
    approved: bool
    current_route: str
    scene_manifest: SceneManifest | None
    characters: list[CharacterProfile]
    image_assets: list[ImageAsset]
    image_backend_used: str
    memory_hits: list[str]
    errors: list[str]
