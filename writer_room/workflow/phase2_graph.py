from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, StateGraph

from writer_room.agents.face_swap import FaceSwapAgent
from writer_room.agents.lip_sync import LipSyncAgent
from writer_room.agents.scene_parser import SceneParserAgent
from writer_room.agents.video_generation import VideoGenerationAgent
from writer_room.agents.voice_synthesis import VoiceSynthesisAgent
from writer_room.config import AppConfig
from writer_room.mcp.registry import MCPRegistry
from writer_room.schemas.phase2 import Phase2State


def build_phase2_initial_state(
    *,
    scene_manifest_path: Path,
    character_db_path: Path,
    raw_scenes_dir: Path,
    face_swapped_dir: Path,
    audio_dir: Path,
    synced_scenes_dir: Path,
    task_graph_dir: Path,
) -> Phase2State:
    return Phase2State(
        scene_manifest_path=scene_manifest_path,
        character_db_path=character_db_path,
        raw_scenes_dir=raw_scenes_dir,
        face_swapped_dir=face_swapped_dir,
        audio_dir=audio_dir,
        synced_scenes_dir=synced_scenes_dir,
        task_graph_dir=task_graph_dir,
        scene_tasks=[],
        task_graph_logs=[],
        audio_outputs=[],
        raw_scene_outputs=[],
        face_swap_outputs=[],
        lip_sync_outputs=[],
        errors=[],
    )


def build_phase2_workflow(config: AppConfig):
    registry = MCPRegistry(config)
    scene_parser = SceneParserAgent(registry)
    voice_synth = VoiceSynthesisAgent(registry)
    video_gen = VideoGenerationAgent(registry)
    face_swap = FaceSwapAgent(registry)
    lip_sync = LipSyncAgent(registry)

    graph = StateGraph(Phase2State)

    graph.add_node("scene_parser_node", scene_parser.run)
    graph.add_node("voice_synth_node", voice_synth.run)
    graph.add_node("video_gen_node", video_gen.run)
    graph.add_node("face_swap_node", face_swap.run)
    graph.add_node("lip_sync_node", lip_sync.run)

    graph.set_entry_point("scene_parser_node")
    graph.add_edge("scene_parser_node", "voice_synth_node")
    graph.add_edge("scene_parser_node", "video_gen_node")
    graph.add_edge("video_gen_node", "face_swap_node")
    graph.add_edge("voice_synth_node", "lip_sync_node")
    graph.add_edge("face_swap_node", "lip_sync_node")
    graph.add_edge("lip_sync_node", END)

    return graph.compile()


def run_phase2_pipeline(config: AppConfig, state: Phase2State) -> Phase2State:
    registry = MCPRegistry(config)
    agents = [
        SceneParserAgent(registry),
        VoiceSynthesisAgent(registry),
        VideoGenerationAgent(registry),
        FaceSwapAgent(registry),
        LipSyncAgent(registry),
    ]

    current_state = dict(state)
    for agent in agents:
        current_state.update(agent.run(current_state))
    return current_state
