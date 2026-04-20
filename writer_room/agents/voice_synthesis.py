from __future__ import annotations

from pathlib import Path

from writer_room.agents.base import BaseAgent
from writer_room.schemas.phase2 import AudioTrack, DialogueCue, Phase2State


class VoiceSynthesisAgent(BaseAgent):
    def run(self, state: Phase2State) -> Phase2State:
        voice_tool = self.resolve_tool_name(["phase2", "audio", "tts"])
        commit_tool = self.resolve_tool_name(["memory", "persistence"])

        audio_outputs: list[AudioTrack] = []
        for task in state.get("scene_tasks", []):
            transcript = " ".join(f"{line.character}: {line.text}" for line in task.dialogue)
            output_path = Path(state["audio_dir"]) / f"scene_{task.scene_number:02d}.wav"
            result = self.registry.invoke(
                voice_tool,
                {
                    "scene_number": task.scene_number,
                    "characters": task.character_names,
                    "transcript": transcript,
                    "dialogue": [
                        {
                            "line_index": index,
                            "character": line.character,
                            "text": line.text,
                            "emotion": line.emotion,
                        }
                        for index, line in enumerate(task.dialogue)
                    ],
                    "output_path": str(output_path),
                },
            )
            track = AudioTrack(
                scene_number=task.scene_number,
                characters=task.character_names,
                audio_path=result.audio_path,
                transcript=result.transcript,
                cue_path=getattr(result, "cue_path", ""),
                cues=[DialogueCue(**cue.model_dump()) for cue in getattr(result, "cues", [])],
            )
            audio_outputs.append(track)
            self.registry.invoke(
                commit_tool,
                {
                    "category": "audio_output",
                    "text": f"Scene {task.scene_number} audio synthesized at {track.audio_path}",
                    "metadata": track.model_dump(),
                },
            )

        return {"audio_outputs": audio_outputs}
