from __future__ import annotations

from writer_room.agents.base import BaseAgent
from writer_room.schemas.state import WorkflowState


class ScriptwriterAgent(BaseAgent):
    def run(self, state: WorkflowState) -> WorkflowState:
        search_tool = self.resolve_tool_name(["memory", "retrieval"])
        memory_hits = self.registry.invoke(search_tool, {"query": state["prompt_text"], "top_k": 2}).hits

        script_tool = self.resolve_tool_name(["scriptwriter", "generation"])
        manifest = self.registry.invoke(script_tool, {"prompt": state["prompt_text"], "memory_context": memory_hits})

        return {
            "scene_manifest": manifest,
            "validated": True,
            "errors": state.get("errors", []),
            "memory_hits": memory_hits,
        }
