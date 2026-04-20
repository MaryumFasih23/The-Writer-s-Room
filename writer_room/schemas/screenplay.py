from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DialogueLine(BaseModel):
    character: str = Field(..., description="Speaker name.")
    text: str = Field(..., description="Spoken line.")
    emotion: str = Field(default="neutral", description="Performance tone.")


class Scene(BaseModel):
    scene_number: int
    heading: str
    summary: str
    action: list[str] = Field(default_factory=list)
    dialogue: list[DialogueLine] = Field(default_factory=list)
    visual_cues: list[str] = Field(default_factory=list)


class SceneManifest(BaseModel):
    title: str
    premise: str
    mode: Literal["manual", "prompt"]
    scenes: list[Scene]


class CharacterProfile(BaseModel):
    name: str
    personality_traits: list[str] = Field(default_factory=list)
    age_descriptor: str = "young adult"
    gender_presentation: str = "person"
    appearance_description: str
    wardrobe: str = "practical cinematic wardrobe"
    environment_hint: str = "story-relevant dramatic environment"
    body_language: str = "expressive, emotionally grounded posture"
    reference_style: str
    source_scenes: list[int] = Field(default_factory=list)


class CharacterDB(BaseModel):
    title: str
    characters: list[CharacterProfile] = Field(default_factory=list)


class ImageAsset(BaseModel):
    character: str
    image_path: str
    prompt_used: str
    backend_used: str
    quality_score: float | None = None
