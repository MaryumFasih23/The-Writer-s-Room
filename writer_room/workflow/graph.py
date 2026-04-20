from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, StateGraph

from writer_room.agents import CharacterDesignerAgent, HITLAgent, ImageSynthesizerAgent, ScriptwriterAgent, ValidatorAgent
from writer_room.config import AppConfig
from writer_room.mcp.registry import MCPRegistry
from writer_room.schemas.screenplay import CharacterDB
from writer_room.schemas.state import WorkflowState


def build_initial_state(
    *,
    mode: str,
    raw_script: str,
    prompt_text: str,
    auto_approve: bool,
    output_dir: Path,
) -> WorkflowState:
    return WorkflowState(
        mode=mode,
        raw_script=raw_script,
        prompt_text=prompt_text,
        auto_approve=auto_approve,
        output_dir=output_dir,
        validated=False,
        approved=False,
        current_route=mode,
        scene_manifest=None,
        characters=[],
        image_assets=[],
        image_backend_used="none",
        memory_hits=[],
        errors=[],
    )


def build_workflow(config: AppConfig):
    registry = MCPRegistry(config)
    validator = ValidatorAgent(registry)
    scriptwriter = ScriptwriterAgent(registry)
    hitl = HITLAgent(registry)
    character = CharacterDesignerAgent(registry)
    image = ImageSynthesizerAgent(registry)

    graph = StateGraph(WorkflowState)

    def mode_selector_node(state: WorkflowState) -> WorkflowState:
        return {"current_route": state["mode"]}

    def validator_node(state: WorkflowState) -> WorkflowState:
        return validator.run(state)

    def scriptwriter_node(state: WorkflowState) -> WorkflowState:
        return scriptwriter.run(state)

    def hitl_node(state: WorkflowState) -> WorkflowState:
        return hitl.run(state)

    def character_node(state: WorkflowState) -> WorkflowState:
        return character.run(state)

    def image_node(state: WorkflowState) -> WorkflowState:
        return image.run(state)

    def memory_commit_node(state: WorkflowState) -> WorkflowState:
        commit_tool = registry.list_tools(required_tags=["memory", "persistence"])[0].name
        manifest = state["scene_manifest"]
        characters = state.get("characters", [])

        config.scene_manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        config.character_db_path.write_text(
            CharacterDB(title=manifest.title, characters=characters).model_dump_json(indent=2),
            encoding="utf-8",
        )

        registry.invoke(
            commit_tool,
            {
                "category": "script",
                "text": f"{manifest.title}: {manifest.premise}",
                "metadata": manifest.model_dump(),
            },
        )
        return state

    def route_by_mode(state: WorkflowState) -> str:
        return "validator_node" if state["current_route"] == "manual" else "scriptwriter_node"

    def route_after_validation(state: WorkflowState):
        return "hitl_node" if state.get("validated", False) and state.get("scene_manifest") else END

    def route_after_hitl(state: WorkflowState):
        return "character_node" if state.get("approved", False) else END

    graph.add_node("mode_selector_node", mode_selector_node)
    graph.add_node("validator_node", validator_node)
    graph.add_node("scriptwriter_node", scriptwriter_node)
    graph.add_node("hitl_node", hitl_node)
    graph.add_node("character_node", character_node)
    graph.add_node("image_node", image_node)
    graph.add_node("memory_commit_node", memory_commit_node)

    graph.set_entry_point("mode_selector_node")
    graph.add_conditional_edges("mode_selector_node", route_by_mode)
    graph.add_conditional_edges("validator_node", route_after_validation)
    graph.add_edge("scriptwriter_node", "hitl_node")
    graph.add_conditional_edges("hitl_node", route_after_hitl)
    graph.add_edge("character_node", "image_node")
    graph.add_edge("image_node", "memory_commit_node")
    graph.add_edge("memory_commit_node", END)

    return graph.compile()
