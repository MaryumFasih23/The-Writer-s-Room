from __future__ import annotations

import base64
import time
from pathlib import Path
from uuid import uuid4

import requests
from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError
from pydantic import BaseModel, Field

from writer_room.config import AppConfig
from writer_room.mcp.base import mcp_tool
from writer_room.schemas.screenplay import ImageAsset


class SynthesizeCharacterImageInput(BaseModel):
    character_name: str
    personality_traits: list[str] = Field(default_factory=list)
    age_descriptor: str = "young adult"
    gender_presentation: str = "person"
    wardrobe: str = "practical cinematic wardrobe"
    environment_hint: str = "story-relevant dramatic environment"
    body_language: str = "expressive body language"
    appearance_description: str = ""
    reference_style: str = ""


class QualityCheckResult(BaseModel):
    accepted: bool
    score: float
    reasons: list[str] = Field(default_factory=list)


def _safe_stem(name: str) -> str:
    return name.lower().replace(" ", "_")


def _active_backend(config: AppConfig) -> str | None:
    backend = config.image_backend.lower()
    if backend == "comfyui" and config.comfyui_base_url:
        return "comfyui"
    if backend == "sd_webui" and config.sd_webui_base_url:
        return "sd_webui"
    if backend == "huggingface" and config.hf_api_token:
        return "huggingface"
    if backend == "auto":
        if config.comfyui_base_url:
            return "comfyui"
        if config.sd_webui_base_url:
            return "sd_webui"
        if config.hf_api_token:
            return "huggingface"
    return None


def _negative_prompt() -> str:
    return (
        "flat vector art, icon, cartoon blob, placeholder avatar, ui panel, interface card, layout frame, "
        "text overlay, watermark, infographic, simple silhouette, low detail, blurry, deformed anatomy, "
        "duplicate limbs, cropped face, empty background, full body, wide shot, long shot, extra fingers, "
        "bad hands, malformed hands, asymmetrical eyes, distorted face, old-fashioned costume, oversized clothing"
    )


def _build_base_prompt(data: SynthesizeCharacterImageInput) -> str:
    traits = ", ".join(data.personality_traits) if data.personality_traits else "layered, expressive"
    return (
        f"cinematic portrait of a {data.age_descriptor} {data.gender_presentation}, {traits}, wearing "
        f"{data.wardrobe}, in {data.environment_hint}, {data.body_language}, detailed clothing, "
        f"realistic lighting, soft light, rim light, dramatic shadows, highly detailed beautiful face, "
        f"sharp eyes, natural skin texture, symmetrical features, elegant facial structure, modern grounded styling, "
        f"cinematic editorial portrait, film still, 4k, head-and-shoulders to upper-torso portrait, "
        f"subject centered, close crop, face is the primary focus, hands mostly out of frame, "
        f"realistic or high-end stylized character concept art, {data.reference_style}, {data.appearance_description}"
    )


def _refine_prompt_with_groq(base_prompt: str, data: SynthesizeCharacterImageInput, config: AppConfig) -> str:
    if not config.groq_api_key:
        return base_prompt

    payload = {
        "model": config.groq_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You improve prompts for cinematic character portrait generation. "
                    "Return exactly one rich prompt sentence and no commentary."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Improve this image prompt for a diffusion model. Keep it realistic, cinematic, and character-focused. "
                    f"Character: {data.character_name}. Prompt: {base_prompt}"
                ),
            },
        ],
        "temperature": 0.6,
        "max_tokens": 180,
    }
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {config.groq_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        return content or base_prompt
    except Exception:
        return base_prompt


def _build_attempt_prompt(data: SynthesizeCharacterImageInput, config: AppConfig, attempt: int) -> str:
    base_prompt = _build_base_prompt(data)
    enriched = _refine_prompt_with_groq(base_prompt, data, config)
    if attempt == 1:
        return enriched
    if attempt == 2:
        return (
            f"{enriched}, visible facial texture, cinematic background depth, lifelike skin, "
            "rich wardrobe detail, believable environment, no poster framing, no card layout, "
            "close portrait only, elegant composition, face-forward cinematic beauty"
        )
    return (
        f"{enriched}, premium cinematic still, emotionally intense portrait photography, natural pose, "
        "complex scene lighting, photoreal detail, no framing device, no design panel, "
        "beauty-focused portrait realism, strong facial coherence, upper torso only, refined facial detail"
    )


def _save_binary_image(binary: bytes, file_path: Path) -> None:
    file_path.write_bytes(binary)


def _generate_via_sd_webui(
    render_prompt: str,
    file_path: Path,
    negative_prompt: str,
    context: dict,
) -> str:
    config: AppConfig = context["config"]
    base_url = config.sd_webui_base_url
    if not base_url:
        raise RuntimeError("`SD_WEBUI_BASE_URL` is not configured.")

    payload = {
        "prompt": render_prompt,
        "negative_prompt": negative_prompt,
        "width": config.image_width,
        "height": config.image_height,
        "steps": config.image_steps,
        "cfg_scale": config.image_cfg_scale,
        "sampler_name": "DPM++ 2M Karras",
    }

    response = requests.post(
        f"{base_url.rstrip('/')}/sdapi/v1/txt2img",
        json=payload,
        timeout=config.image_timeout_s,
    )
    response.raise_for_status()
    body = response.json()
    if not body.get("images"):
        raise RuntimeError("Stable Diffusion WebUI returned no images.")

    encoded = body["images"][0].split(",", 1)[-1]
    _save_binary_image(base64.b64decode(encoded), file_path)
    return "sd_webui"


def _generate_via_comfyui(
    render_prompt: str,
    file_path: Path,
    negative_prompt: str,
    context: dict,
) -> str:
    config: AppConfig = context["config"]
    base_url = config.comfyui_base_url
    if not base_url:
        raise RuntimeError("`COMFYUI_BASE_URL` is not configured.")

    filename_prefix = f"writer_room_{file_path.stem}_{uuid4().hex[:8]}"
    workflow = {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": config.comfyui_checkpoint}},
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": config.image_width, "height": config.image_height, "batch_size": 1},
        },
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": render_prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["4", 1]}},
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(time.time()) + int(uuid4().hex[:4], 16),
                "steps": config.image_steps,
                "cfg": config.image_cfg_scale,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": ["8", 0]}},
    }

    response = requests.post(
        f"{base_url.rstrip('/')}/prompt",
        json={"prompt": workflow, "client_id": uuid4().hex},
        timeout=config.image_timeout_s,
    )
    response.raise_for_status()
    prompt_id = response.json()["prompt_id"]

    deadline = time.time() + config.image_timeout_s
    history_payload = None
    while time.time() < deadline:
        history_response = requests.get(f"{base_url.rstrip('/')}/history/{prompt_id}", timeout=15)
        history_response.raise_for_status()
        history_payload = history_response.json().get(prompt_id)
        if history_payload and history_payload.get("outputs"):
            break
        time.sleep(2)

    if not history_payload:
        raise RuntimeError("ComfyUI generation timed out before producing an output.")

    output_images = history_payload["outputs"].get("9", {}).get("images", [])
    if not output_images:
        raise RuntimeError("ComfyUI completed but returned no saved images.")

    first_image = output_images[0]
    binary = requests.get(
        f"{base_url.rstrip('/')}/view",
        params={
            "filename": first_image["filename"],
            "subfolder": first_image.get("subfolder", ""),
            "type": first_image.get("type", "output"),
        },
        timeout=30,
    )
    binary.raise_for_status()
    _save_binary_image(binary.content, file_path)
    return "comfyui"


def _generate_via_huggingface(
    render_prompt: str,
    file_path: Path,
    negative_prompt: str,
    context: dict,
) -> str:
    config: AppConfig = context["config"]
    if not config.hf_api_token:
        raise RuntimeError("`HF_API_TOKEN` is not configured.")

    payload = {
        "inputs": render_prompt,
        "parameters": {
            "negative_prompt": negative_prompt,
            "width": config.image_width,
            "height": config.image_height,
            "num_inference_steps": config.image_steps,
            "guidance_scale": config.image_cfg_scale,
        },
        "options": {"wait_for_model": True},
    }
    headers = {
        "Authorization": f"Bearer {config.hf_api_token}",
        "Content-Type": "application/json",
        "Accept": "image/png",
    }

    candidate_models: list[str] = []
    for model in [config.hf_image_model, *config.hf_image_model_fallbacks]:
        if model and model not in candidate_models:
            candidate_models.append(model)

    last_error = "Hugging Face image generation timed out."
    for model_name in candidate_models:
        endpoint = f"https://router.huggingface.co/hf-inference/models/{model_name}"
        deadline = time.time() + config.image_timeout_s
        while time.time() < deadline:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=config.image_timeout_s)
            if response.status_code == 503:
                time.sleep(5)
                continue
            if response.status_code in {404, 410}:
                last_error = f"Hugging Face model `{model_name}` is unavailable ({response.status_code})."
                break
            response.raise_for_status()
            if response.headers.get("content-type", "").startswith("image/"):
                _save_binary_image(response.content, file_path)
                context["last_hf_model_used"] = model_name
                return "huggingface"
            last_error = f"Hugging Face returned a non-image response for `{model_name}`: {response.text[:300]}"
            break

    raise RuntimeError(last_error)


def _evaluate_image_quality(file_path: Path) -> QualityCheckResult:
    try:
        image = Image.open(file_path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError) as exc:
        return QualityCheckResult(accepted=False, score=0.0, reasons=[f"Invalid image output: {exc}"])

    gray = image.convert("L")
    width, height = image.size
    center = gray.crop((width // 4, height // 5, width * 3 // 4, height * 4 // 5))
    edge_image = gray.filter(ImageFilter.FIND_EDGES)
    entropy = gray.entropy()
    center_entropy = center.entropy()
    edge_stats = ImageStat.Stat(edge_image)
    edge_mean = edge_stats.mean[0]
    edge_std = edge_stats.stddev[0]

    quantized = image.quantize(colors=256)
    colors = quantized.getcolors(256) or []
    color_count = len(colors)

    border = Image.new("L", (width, height), 0)
    border.paste(gray.crop((0, 0, width, max(8, height // 12))), (0, 0))
    border.paste(gray.crop((0, height - max(8, height // 12), width, height)), (0, height - max(8, height // 12)))
    border.paste(gray.crop((0, 0, max(8, width // 12), height)), (0, 0))
    border.paste(gray.crop((width - max(8, width // 12), 0, width, height)), (width - max(8, width // 12), 0))
    border_std = ImageStat.Stat(border).stddev[0]

    score = min(1.0, max(0.0, (
        (entropy / 8.0) * 0.32
        + (center_entropy / 8.0) * 0.28
        + min(edge_mean / 35.0, 1.0) * 0.20
        + min(edge_std / 40.0, 1.0) * 0.10
        + min(color_count / 180.0, 1.0) * 0.10
    )))

    reasons: list[str] = []
    hard_fail = False
    if entropy < 5.0:
        reasons.append("overall image entropy is too low and suggests flat/minimal output")
        if entropy < 4.4:
            hard_fail = True
    if center_entropy < 4.7:
        reasons.append("center composition is too simple for a cinematic portrait")
        if center_entropy < 4.2:
            hard_fail = True
    if edge_mean < 10.0 or edge_std < 12.0:
        reasons.append("detail density is too low and may indicate a placeholder-like render")
    if color_count < 70:
        reasons.append("palette diversity is too limited for a rich portrait render")
        if color_count < 40:
            hard_fail = True
    if border_std < 15.0 and entropy < 5.8:
        reasons.append("borders look too uniform and may indicate a framed card or panel")
        hard_fail = True

    accepted = (score >= 0.72 and not hard_fail) or score >= 0.82
    return QualityCheckResult(accepted=accepted, score=score, reasons=reasons)


def _generate_with_quality_control(data: SynthesizeCharacterImageInput, context: dict) -> ImageAsset:
    config: AppConfig = context["config"]
    backend = _active_backend(config)
    if not backend:
        raise RuntimeError(
            "No real image backend is configured. Set `COMFYUI_BASE_URL`, `SD_WEBUI_BASE_URL`, "
            "or `HF_API_TOKEN` for Hugging Face inference."
        )

    generators = {
        "comfyui": _generate_via_comfyui,
        "sd_webui": _generate_via_sd_webui,
        "huggingface": _generate_via_huggingface,
    }
    generator = generators[backend]
    negative_prompt = _negative_prompt()
    final_path = Path(context["images_dir"]) / f"{_safe_stem(data.character_name)}.png"
    last_reasons: list[str] = []
    last_score = 0.0
    last_prompt = ""

    for attempt in range(1, config.image_max_attempts + 1):
        render_prompt = _build_attempt_prompt(data, config, attempt)
        candidate_path = final_path.with_name(f"{final_path.stem}_attempt{attempt}.png")
        backend_used = generator(render_prompt, candidate_path, negative_prompt, context)
        quality = _evaluate_image_quality(candidate_path)
        last_reasons = quality.reasons
        last_score = quality.score
        last_prompt = render_prompt

        if quality.accepted:
            if final_path.exists():
                final_path.unlink()
            candidate_path.replace(final_path)
            context["last_image_backend_used"] = backend_used
            return ImageAsset(
                character=data.character_name,
                image_path=final_path.as_posix(),
                prompt_used=render_prompt,
                backend_used=backend_used,
                quality_score=quality.score,
            )

        if candidate_path.exists():
            candidate_path.unlink()

    raise RuntimeError(
        "Generated images failed quality checks after regeneration. "
        + "; ".join(last_reasons)
        + f". Last quality score: {last_score:.2f}. Last prompt: {last_prompt}"
    )


@mcp_tool(
    name="synthesize_character_image_real",
    description="Generate a real cinematic character portrait using a discovered image backend and reject weak outputs.",
    tags=["image", "generation", "real", "preferred"],
    input_model=SynthesizeCharacterImageInput,
    output_model=ImageAsset,
)
def synthesize_character_image_real_tool(data: SynthesizeCharacterImageInput, context: dict) -> ImageAsset:
    return _generate_with_quality_control(data, context)
