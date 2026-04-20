from __future__ import annotations

import argparse
from pathlib import Path

from writer_room.config import AppConfig
from writer_room.workflow.phase2_graph import (
    build_phase2_initial_state,
    build_phase2_workflow,
    run_phase2_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PROJECT MONTAGE - Phase 2 Studio Floor")
    parser.add_argument("--scene-manifest", type=Path, default=Path("outputs/scene_manifest.json"))
    parser.add_argument("--character-db", type=Path, default=Path("outputs/character_db.json"))
    parser.add_argument("--raw-scenes-dir", type=Path, default=Path("raw_scenes"))
    parser.add_argument("--face-swapped-dir", type=Path, default=Path("face_swapped"))
    parser.add_argument("--audio-dir", type=Path, default=Path("audio"))
    parser.add_argument("--synced-scenes-dir", type=Path, default=Path("synced_scenes"))
    parser.add_argument("--task-graph-dir", type=Path, default=Path("task_graph_logs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def expected_phase2_outputs(state: dict) -> list[Path]:
    paths: list[Path] = []
    for item in state.get("task_graph_logs", []):
        paths.append(Path(item.log_path))
    for item in state.get("audio_outputs", []):
        paths.append(Path(item.audio_path))
    for item in state.get("raw_scene_outputs", []):
        paths.append(Path(item.video_path))
    for item in state.get("face_swap_outputs", []):
        paths.append(Path(item.video_path))
    for item in state.get("lip_sync_outputs", []):
        paths.append(Path(item.video_path))
    return paths


def outputs_materialized(state: dict) -> bool:
    expected = expected_phase2_outputs(state)
    return bool(expected) and all(path.exists() for path in expected)


def main() -> None:
    args = parse_args()
    config = AppConfig(output_dir=args.output_dir)
    config.ensure_directories()
    ensure_dirs(args.raw_scenes_dir, args.face_swapped_dir, args.audio_dir, args.synced_scenes_dir, args.task_graph_dir)

    state = build_phase2_initial_state(
        scene_manifest_path=args.scene_manifest,
        character_db_path=args.character_db,
        raw_scenes_dir=args.raw_scenes_dir,
        face_swapped_dir=args.face_swapped_dir,
        audio_dir=args.audio_dir,
        synced_scenes_dir=args.synced_scenes_dir,
        task_graph_dir=args.task_graph_dir,
    )
    app = build_phase2_workflow(config)
    result = app.invoke(state)
    if not outputs_materialized(result):
        print("Graph run returned incomplete files on disk. Re-running deterministic Phase 2 pipeline...")
        result = run_phase2_pipeline(config, state)

    print("Phase 2 execution complete.")
    print(f"Task graphs: {len(result.get('task_graph_logs', []))}")
    print(f"Audio outputs: {len(result.get('audio_outputs', []))}")
    print(f"Raw scene outputs: {len(result.get('raw_scene_outputs', []))}")
    print(f"Face swap outputs: {len(result.get('face_swap_outputs', []))}")
    print(f"Lip-synced outputs: {len(result.get('lip_sync_outputs', []))}")
    print(f"Raw scenes dir: {args.raw_scenes_dir}")
    print(f"Face-swapped dir: {args.face_swapped_dir}")
    print(f"Audio dir: {args.audio_dir}")
    print(f"Synced scenes dir: {args.synced_scenes_dir}")
    if result.get("errors"):
        print("Errors:")
        for err in result["errors"]:
            print(f" - {err}")


if __name__ == "__main__":
    main()
