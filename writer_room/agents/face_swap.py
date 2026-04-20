from __future__ import annotations

from pathlib import Path

from writer_room.agents.base import BaseAgent
from writer_room.schemas.phase2 import FaceSwappedVideo, Phase2State


class FaceSwapAgent(BaseAgent):
    def run(self, state: Phase2State) -> Phase2State:
        validator_tool = self.resolve_tool_name(["phase2", "video", "identity"])
        swap_tool = self.resolve_tool_name(["phase2", "video", "face-swap"])
        commit_tool = self.resolve_tool_name(["memory", "persistence"])

        face_outputs: list[FaceSwappedVideo] = []
        task_map = {task.scene_number: task for task in state.get("scene_tasks", [])}

        for raw_video in state.get("raw_scene_outputs", []):
            task = task_map[raw_video.scene_number]
            validation = self.registry.invoke(
                validator_tool,
                {
                    "character_names": task.character_names,
                    "character_images": raw_video.character_images,
                },
            )
            if not validation.valid:
                raise RuntimeError(validation.notes)

            output_path = Path(state["face_swapped_dir"]) / f"scene_{task.scene_number:02d}_faceswapped.mp4"
            result = self.registry.invoke(
                swap_tool,
                {
                    "scene_number": task.scene_number,
                    "source_video_path": raw_video.video_path,
                    "character_names": task.character_names,
                    "character_images": raw_video.character_images,
                    "scene_text": " ".join([task.heading, task.summary, *task.visual_cues]).strip(),
                    "output_path": str(output_path),
                },
            )
            mapped = FaceSwappedVideo(
                scene_number=task.scene_number,
                video_path=result.video_path,
                source_video_path=raw_video.video_path,
                validated=validation.valid,
                mapped_characters=result.mapped_characters,
                frame_dir=getattr(result, "frame_dir", ""),
            )
            face_outputs.append(mapped)
            self.registry.invoke(
                commit_tool,
                {
                    "category": "video_output",
                    "text": f"Scene {task.scene_number} face swap output created at {mapped.video_path}",
                    "metadata": mapped.model_dump(),
                },
            )

        return {"face_swap_outputs": face_outputs}
