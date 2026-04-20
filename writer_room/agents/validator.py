from __future__ import annotations

from writer_room.agents.base import BaseAgent
from writer_room.schemas.state import WorkflowState


class ValidatorAgent(BaseAgent):
    def run(self, state: WorkflowState) -> WorkflowState:
        tool_name = self.resolve_tool_name(["validator", "validation"])
        result = self.registry.invoke(tool_name, {"raw_script": state["raw_script"]})
        errors = state.get("errors", []) + result.errors
        return {"validated": result.valid, "scene_manifest": result.scene_manifest, "errors": errors}
