from __future__ import annotations

import argparse
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency fallback
    def load_dotenv() -> bool:
        return False

from writer_room.config import AppConfig
from writer_room.workflow.graph import build_initial_state, build_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PROJECT MONTAGE - Writer's Room")
    parser.add_argument("--mode", choices=["manual", "prompt"], required=True)
    parser.add_argument("--script-file", type=Path)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--auto-approve", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def load_text(path: Path | None, inline_text: str | None = None) -> str:
    if inline_text:
        return inline_text.strip()
    if path:
        return path.read_text(encoding="utf-8").strip()
    return ""


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = AppConfig(output_dir=args.output_dir)
    config.ensure_directories()

    raw_script = load_text(args.script_file) if args.mode == "manual" else ""
    prompt_text = load_text(args.prompt_file, args.prompt) if args.mode == "prompt" else ""

    state = build_initial_state(
        mode=args.mode,
        raw_script=raw_script,
        prompt_text=prompt_text,
        auto_approve=args.auto_approve,
        output_dir=config.output_dir,
    )

    app = build_workflow(config)
    result = app.invoke(state)

    print("Execution complete.")
    print(f"Approved: {result['approved']}")
    print(f"Scene manifest: {config.scene_manifest_path}")
    print(f"Character DB: {config.character_db_path}")
    print(f"Images directory: {config.images_dir}")
    print(f"Image backend: {result.get('image_backend_used', 'unknown')}")
    if result.get("errors"):
        print("Errors:")
        for err in result["errors"]:
            print(f" - {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
