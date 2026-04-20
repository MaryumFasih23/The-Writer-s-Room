from __future__ import annotations

from writer_room.agents.base import BaseAgent
from writer_room.schemas.state import WorkflowState


class ImageSynthesizerAgent(BaseAgent):
    def run(self, state: WorkflowState) -> WorkflowState:
        commit_tool = self.resolve_tool_name(["memory", "persistence"])
        assets = []
        errors = list(state.get("errors", []))
        backend_used = state.get("image_backend_used", "none")

        real_tools = self.registry.list_tools(required_tags=["image", "generation", "real", "preferred"])

        if not real_tools:
            return {
                "image_assets": assets,
                "image_backend_used": "none",
                "errors": errors + ["No real MCP image generation tool is available for this run."],
            }

        for character in state.get("characters", []):
            payload = {
                "character_name": character.name,
                "personality_traits": character.personality_traits,
                "age_descriptor": character.age_descriptor,
                "gender_presentation": character.gender_presentation,
                "wardrobe": character.wardrobe,
                "environment_hint": character.environment_hint,
                "body_language": character.body_language,
                "appearance_description": character.appearance_description,
                "reference_style": character.reference_style,
            }

            try:
                asset = self.registry.invoke(real_tools[0].name, payload)
                backend_used = self.registry.context.get("last_image_backend_used", "real")
            except Exception as exc:
                return {
                    "image_assets": assets,
                    "image_backend_used": "none",
                    "errors": errors
                    + [
                        "Real image generation failed. Configure `COMFYUI_BASE_URL`, `SD_WEBUI_BASE_URL`, "
                        "or `HF_API_TOKEN` for a real image backend.",
                        f"{character.name}: {exc}",
                    ],
                }

            self.registry.invoke(
                commit_tool,
                {
                    "category": "image_asset",
                    "text": f"{asset.character} image asset saved at {asset.image_path}",
                    "metadata": asset.model_dump(),
                },
            )
            assets.append(asset)

        return {"image_assets": assets, "image_backend_used": backend_used, "errors": errors}
