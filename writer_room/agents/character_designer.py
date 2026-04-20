from __future__ import annotations

from writer_room.agents.base import BaseAgent
from writer_room.schemas.state import WorkflowState


class CharacterDesignerAgent(BaseAgent):
    def run(self, state: WorkflowState) -> WorkflowState:
        manifest = state["scene_manifest"]
        extract_tool = self.resolve_tool_name(["character", "extraction"])
        query_tool = self.resolve_tool_name(["character", "reference"])
        commit_tool = self.resolve_tool_name(["memory", "persistence"])

        profiles = self.registry.invoke(extract_tool, {"scene_manifest": manifest}).characters
        enriched = []

        for profile in profiles:
            reference = self.registry.invoke(
                query_tool,
                {"character_name": profile.name, "appearance_description": profile.appearance_description},
            )
            profile.reference_style = reference.reference_style
            self.registry.invoke(
                commit_tool,
                {
                    "category": "character",
                    "text": f"{profile.name}: {profile.appearance_description}",
                    "metadata": profile.model_dump(),
                },
            )
            enriched.append(profile)

        return {"characters": enriched}
