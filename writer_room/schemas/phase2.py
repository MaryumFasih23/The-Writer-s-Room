from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field

from writer_room.schemas.screenplay import CharacterDB, DialogueLine, SceneManifest


class SceneTask(BaseModel):
    scene_number: int
    heading: str
    summary: str
    action: list[str] = Field(default_factory=list)
    dialogue: list[DialogueLine] = Field(default_factory=list)
    visual_cues: list[str] = Field(default_factory=list)
    character_names: list[str] = Field(default_factory=list)
    environment_description: str


class TaskGraphLog(BaseModel):
    scene_number: int
    task_id: str
    task_breakdown: list[str] = Field(default_factory=list)
    log_path: str


class DialogueCue(BaseModel):
    line_index: int
    character: str
    text: str
    emotion: str = "neutral"
    start_s: float
    end_s: float
    duration_s: float
    voice_name: str | None = None
    backend: str = "unknown"


class AudioTrack(BaseModel):
    scene_number: int
    characters: list[str] = Field(default_factory=list)
    audio_path: str
    transcript: str
    cue_path: str = ""
    cues: list[DialogueCue] = Field(default_factory=list)


class RawSceneVideo(BaseModel):
    scene_number: int
    video_path: str
    footage_prompt: str
    character_images: list[str] = Field(default_factory=list)
    shot_plan: list[str] = Field(default_factory=list)
    frame_dir: str = ""


class FaceSwappedVideo(BaseModel):
    scene_number: int
    video_path: str
    source_video_path: str
    validated: bool
    mapped_characters: list[str] = Field(default_factory=list)
    frame_dir: str = ""


class LipSyncedVideo(BaseModel):
    scene_number: int
    video_path: str
    source_video_path: str
    audio_path: str
    frame_dir: str = ""
    alignment_path: str = ""


class Phase2State(TypedDict, total=False):
    scene_manifest_path: Path
    character_db_path: Path
    raw_scenes_dir: Path
    face_swapped_dir: Path
    audio_dir: Path
    synced_scenes_dir: Path
    task_graph_dir: Path
    scene_manifest: SceneManifest
    character_db: CharacterDB
    scene_tasks: list[SceneTask]
    task_graph_logs: Annotated[list[TaskGraphLog], operator.add]
    audio_outputs: list[AudioTrack]
    raw_scene_outputs: list[RawSceneVideo]
    face_swap_outputs: list[FaceSwappedVideo]
    lip_sync_outputs: list[LipSyncedVideo]
    errors: Annotated[list[str], operator.add]
