from __future__ import annotations

from pathlib import Path

from writer_room.agents.base import BaseAgent
from writer_room.schemas.phase2 import LipSyncedVideo, Phase2State


class LipSyncAgent(BaseAgent):
    def run(self, state: Phase2State) -> Phase2State:
        lip_sync_tool = self.resolve_tool_name(["phase2", "video", "audio", "lip-sync"])
        commit_tool = self.resolve_tool_name(["memory", "persistence"])

        audio_map = {track.scene_number: track for track in state.get("audio_outputs", [])}
        raw_map = {video.scene_number: video for video in state.get("raw_scene_outputs", [])}
        lip_sync_outputs: list[LipSyncedVideo] = []

        for mapped_video in state.get("face_swap_outputs", []):
            audio_track = audio_map[mapped_video.scene_number]
            raw_video = raw_map.get(mapped_video.scene_number)
            output_path = Path(state["synced_scenes_dir"]) / f"scene_{mapped_video.scene_number:02d}_synced.mp4"
            result = self.registry.invoke(
                lip_sync_tool,
                {
                    "scene_number": mapped_video.scene_number,
                    "source_video_path": mapped_video.video_path,
                    "audio_path": audio_track.audio_path,
                    "cue_path": audio_track.cue_path,
                    "cues": [cue.model_dump() for cue in audio_track.cues],
                    "character_names": mapped_video.mapped_characters,
                    "character_images": raw_video.character_images if raw_video else [],
                    "output_path": str(output_path),
                },
            )
            synced = LipSyncedVideo(
                scene_number=mapped_video.scene_number,
                video_path=result.video_path,
                source_video_path=mapped_video.video_path,
                audio_path=result.audio_path,
                frame_dir=getattr(result, "frame_dir", ""),
                alignment_path=getattr(result, "alignment_path", ""),
            )
            lip_sync_outputs.append(synced)
            self.registry.invoke(
                commit_tool,
                {
                    "category": "video_output",
                    "text": f"Scene {mapped_video.scene_number} lip-synced output created at {synced.video_path}",
                    "metadata": synced.model_dump(),
                },
            )

        return {"lip_sync_outputs": lip_sync_outputs}
