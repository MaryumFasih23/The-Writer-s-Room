from __future__ import annotations

import re
from pathlib import Path

from writer_room.agents.base import BaseAgent
from writer_room.schemas.phase2 import Phase2State, RawSceneVideo


def _slug_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _find_character_image(name: str) -> str:
    slug = _slug_name(name)
    candidates = [
        Path("image_assets") / f"{name.lower()}.png",
        Path("image_assets") / f"{slug}.png",
        Path("outputs") / "images" / f"{name.lower()}.png",
        Path("outputs") / "images" / f"{slug}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(Path("image_assets") / f"{slug}.png")


class VideoGenerationAgent(BaseAgent):
    def run(self, state: Phase2State) -> Phase2State:
        footage_tool = self.resolve_tool_name(["phase2", "video", "stock-footage"])
        commit_tool = self.resolve_tool_name(["memory", "persistence"])

        character_lookup = {character.name: character for character in state["character_db"].characters}
        raw_outputs: list[RawSceneVideo] = []

        for task in state.get("scene_tasks", []):
            first_character = task.character_names[0] if task.character_names else f"Scene{task.scene_number}"
            profile = character_lookup.get(first_character)
            appearance = profile.appearance_description if profile else task.environment_description
            character_images = [_find_character_image(name) for name in task.character_names]
            output_path = Path(state["raw_scenes_dir"]) / f"scene_{task.scene_number:02d}.mp4"
            result = self.registry.invoke(
                footage_tool,
                {
                    "character_name": first_character,
                    "appearance_description": appearance,
                    "scene_number": task.scene_number,
                    "scene_heading": task.heading,
                    "scene_summary": task.summary,
                    "visual_cues": task.visual_cues,
                    "environment_description": task.environment_description,
                    "character_images": character_images,
                    "output_path": str(output_path),
                },
            )
            raw_video = RawSceneVideo(
                scene_number=task.scene_number,
                video_path=result.video_path,
                footage_prompt=result.footage_prompt,
                character_images=character_images,
                shot_plan=result.shot_plan,
                frame_dir=getattr(result, "frame_dir", ""),
            )
            raw_outputs.append(raw_video)
            self.registry.invoke(
                commit_tool,
                {
                    "category": "video_output",
                    "text": f"Scene {task.scene_number} raw video created at {raw_video.video_path}",
                    "metadata": raw_video.model_dump(),
                },
            )

        return {"raw_scene_outputs": raw_outputs}
