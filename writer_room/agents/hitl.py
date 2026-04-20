from __future__ import annotations

from writer_room.agents.base import BaseAgent
from writer_room.schemas.state import WorkflowState


class HITLAgent(BaseAgent):
    def run(self, state: WorkflowState) -> WorkflowState:
        manifest = state.get("scene_manifest")
        if manifest is None:
            return {"approved": False, "errors": state.get("errors", []) + ["No manifest available for review."]}

        if not state.get("validated", False):
            return {"approved": False}

        if state.get("auto_approve", False):
            return {"approved": True}

        print("\nHITL REVIEW CHECKPOINT")
        print(f"Title: {manifest.title}")
        print(f"Scenes: {len(manifest.scenes)}")
        for scene in manifest.scenes:
            print(f" - Scene {scene.scene_number}: {scene.heading}")
        decision = input("Approve and continue? [y/N]: ").strip().lower()
        return {"approved": decision == "y"}
