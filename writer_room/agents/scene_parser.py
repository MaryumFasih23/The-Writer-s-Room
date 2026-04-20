from __future__ import annotations

import json
from pathlib import Path

from writer_room.agents.base import BaseAgent
from writer_room.schemas.phase2 import Phase2State, SceneTask, TaskGraphLog
from writer_room.schemas.screenplay import CharacterDB, SceneManifest


class SceneParserAgent(BaseAgent):
    def run(self, state: Phase2State) -> Phase2State:
        manifest = SceneManifest(**json.loads(Path(state["scene_manifest_path"]).read_text(encoding="utf-8")))
        character_db = CharacterDB(**json.loads(Path(state["character_db_path"]).read_text(encoding="utf-8")))

        task_tool = self.resolve_tool_name(["phase2", "task-graph", "scene-parser"])
        commit_tool = self.resolve_tool_name(["memory", "persistence"])

        scene_tasks: list[SceneTask] = []
        task_logs: list[TaskGraphLog] = []

        for scene in manifest.scenes:
            character_names = sorted({line.character for line in scene.dialogue})
            environment_description = " ".join([scene.heading, scene.summary, *scene.visual_cues]).strip()
            task = SceneTask(
                scene_number=scene.scene_number,
                heading=scene.heading,
                summary=scene.summary,
                action=scene.action,
                dialogue=scene.dialogue,
                visual_cues=scene.visual_cues,
                character_names=character_names,
                environment_description=environment_description,
            )
            scene_tasks.append(task)

            log_path = Path(state["task_graph_dir"]) / f"scene_{scene.scene_number:02d}_task_graph.json"
            tool_result = self.registry.invoke(
                task_tool,
                {
                    "scene_number": scene.scene_number,
                    "heading": scene.heading,
                    "summary": scene.summary,
                    "action": scene.action,
                    "visual_cues": scene.visual_cues,
                    "dialogue_characters": character_names,
                    "output_path": str(log_path),
                },
            )
            task_log = TaskGraphLog(
                scene_number=scene.scene_number,
                task_id=tool_result.task_id,
                task_breakdown=tool_result.task_breakdown,
                log_path=tool_result.log_path,
            )
            task_logs.append(task_log)
            self.registry.invoke(
                commit_tool,
                {
                    "category": "task_graph",
                    "text": f"Scene {scene.scene_number} task graph created for {scene.heading}",
                    "metadata": task_log.model_dump(),
                },
            )

        return {
            "scene_manifest": manifest,
            "character_db": character_db,
            "scene_tasks": scene_tasks,
            "task_graph_logs": task_logs,
        }
