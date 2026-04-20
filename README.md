# PROJECT MONTAGE - Phase 1

This repository implements the "Writer's Room" phase from the assignment as a complete, runnable Python project.

## Features

- Multi-agent system with Scriptwriter, Validator, HITL, Character Designer, and Image Synthesizer agents
- LangGraph `StateGraph` orchestration
- MCP-style dynamic tool discovery with runtime registry lookup
- Shared persistent vector memory using a simple local vector store
- Two modes:
  - Manual script validation
  - Prompt to screenplay generation
- Required outputs:
  - `outputs/scene_manifest.json`
  - `outputs/character_db.json`
  - `image_assets/*.png`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Optional but recommended: create a `.env` file in the project root.

Example `.env`:

```env
COMFYUI_BASE_URL=http://127.0.0.1:8188
SD_WEBUI_BASE_URL=http://127.0.0.1:7860
HF_API_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_key
IMAGE_BACKEND=auto
IMAGE_WIDTH=768
IMAGE_HEIGHT=768
IMAGE_STEPS=28
IMAGE_CFG_SCALE=7.0
IMAGE_MAX_ATTEMPTS=3
```

## Run

### Real image generation setup

This project now expects a real local image backend by default.

Option 1: ComfyUI

```powershell
$env:COMFYUI_BASE_URL="http://127.0.0.1:8188"
```

Option 2: Stable Diffusion WebUI / AUTOMATIC1111

```powershell
$env:SD_WEBUI_BASE_URL="http://127.0.0.1:7860"
```

Option 3: Hugging Face Inference API

```powershell
$env:HF_API_TOKEN="your_token_here"
$env:IMAGE_BACKEND="huggingface"
```

Optional prompt enhancement with Groq:

```powershell
$env:GROQ_API_KEY="your_groq_key"
```

Optional image settings:

```powershell
$env:IMAGE_BACKEND="auto"
$env:IMAGE_WIDTH="768"
$env:IMAGE_HEIGHT="768"
$env:IMAGE_STEPS="28"
$env:IMAGE_CFG_SCALE="7.0"
$env:IMAGE_MAX_ATTEMPTS="3"
```

### Manual script validation

```bash
python main.py --mode manual --script-file examples/manual_script.txt --auto-approve
```

### Prompt to screenplay generation

```bash
python main.py --mode prompt --prompt-file examples/prompt_input.txt --auto-approve
```

Omit `--auto-approve` to use the interactive HITL checkpoint.

## Outputs

After execution, the system writes:

- `outputs/scene_manifest.json`
- `outputs/character_db.json`
- `image_assets/`
- `outputs/memory_store.json`

## Phase 2 Studio Floor

Phase 2 converts the Phase 1 screenplay and character database into audiovisual scene assets.

```bash
python phase2_main.py
```

The Phase 2 workflow keeps the assignment's parallel audio/video architecture:

- `SceneParserAgent` loads `outputs/scene_manifest.json` and `outputs/character_db.json`, then writes one MCP task graph log per scene.
- `VoiceSynthesisAgent` generates scene WAV files plus dialogue cue timing JSON.
- `VideoGenerationAgent` renders raw cinematic scene videos from the Phase 1 character portraits.
- `FaceSwapAgent` validates the character image references before rendering face-mapped scene videos.
- `LipSyncAgent` aligns audio amplitude and dialogue cue timings to animated mouth shapes, then muxes the audio into the final synced MP4.

Phase 2 writes:

- `task_graph_logs/scene_XX_task_graph.json`
- `audio/scene_XX.wav`
- `audio/scene_XX.cues.json`
- `raw_scenes/scene_XX.mp4`
- `raw_scenes/frames/scene_XX/frame_*.png`
- `face_swapped/scene_XX_faceswapped.mp4`
- `face_swapped/frames/scene_XX_faceswapped/frame_*.png`
- `synced_scenes/scene_XX_synced.mp4`
- `synced_scenes/scene_XX_synced.alignment.json`
- `synced_scenes/frames/scene_XX_synced/frame_*.png`
- `outputs/mouth_anchors.json`

No paid LLM or paid TTS API is required for Phase 2. The voice tool now uses Edge-TTS neural English voices by default and writes the generated speech back to WAV for the existing lip-sync stage. `outputs/voice_map.json` keeps each character on a consistent voice, such as `en-US-JennyNeural` for Mira and `en-US-GuyNeural` for Rowan. If Edge-TTS is unavailable, the system falls back to the previous free local/offline synthesis paths so the pipeline remains runnable.

The lip-sync aligner computes a waveform RMS window for each video frame using `sample_rate / fps`; the resulting values are stored in each `*.alignment.json` file under `audio_frame_alignment`. Mouth placement comes from calibrated character anchors in `outputs/mouth_anchors.json`, with automatic detection as a fallback for new characters.

## Example files

- `examples/manual_script.txt`
- `examples/prompt_input.txt`
- `examples/example_scene_manifest.json`
- `examples/example_character_db.json`
