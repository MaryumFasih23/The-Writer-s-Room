from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from uuid import uuid4

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from pydantic import BaseModel, Field

from writer_room.mcp.base import mcp_tool


class GetTaskGraphInput(BaseModel):
    scene_number: int
    heading: str
    summary: str
    action: list[str] = Field(default_factory=list)
    visual_cues: list[str] = Field(default_factory=list)
    dialogue_characters: list[str] = Field(default_factory=list)
    output_path: str


class GetTaskGraphOutput(BaseModel):
    task_id: str
    task_breakdown: list[str] = Field(default_factory=list)
    log_path: str


class DialogueSynthesisLine(BaseModel):
    line_index: int = 0
    character: str
    text: str
    emotion: str = "neutral"


class DialogueCueOutput(BaseModel):
    line_index: int
    character: str
    text: str
    emotion: str = "neutral"
    start_s: float
    end_s: float
    duration_s: float
    voice_name: str | None = None
    backend: str = "unknown"


class VoiceCloningSynthesizerInput(BaseModel):
    scene_number: int
    characters: list[str] = Field(default_factory=list)
    transcript: str
    dialogue: list[DialogueSynthesisLine] = Field(default_factory=list)
    output_path: str


class VoiceCloningSynthesizerOutput(BaseModel):
    audio_path: str
    transcript: str
    cue_path: str = ""
    cues: list[DialogueCueOutput] = Field(default_factory=list)


class QueryStockFootageInput(BaseModel):
    character_name: str
    appearance_description: str
    scene_number: int | None = None
    scene_heading: str | None = None
    scene_summary: str | None = None
    visual_cues: list[str] = Field(default_factory=list)
    environment_description: str | None = None
    character_images: list[str] = Field(default_factory=list)
    output_path: str | None = None


class QueryStockFootageOutput(BaseModel):
    reference_style: str
    footage_prompt: str = ""
    shot_plan: list[str] = Field(default_factory=list)
    video_path: str = ""
    frame_dir: str = ""


class IdentityValidatorInput(BaseModel):
    character_names: list[str] = Field(default_factory=list)
    character_images: list[str] = Field(default_factory=list)


class IdentityValidatorOutput(BaseModel):
    valid: bool
    notes: str


class FaceSwapperInput(BaseModel):
    scene_number: int
    source_video_path: str
    character_names: list[str] = Field(default_factory=list)
    character_images: list[str] = Field(default_factory=list)
    scene_text: str = ""
    output_path: str


class FaceSwapperOutput(BaseModel):
    video_path: str
    mapped_characters: list[str] = Field(default_factory=list)
    frame_dir: str = ""


class LipSyncAlignerInput(BaseModel):
    scene_number: int
    source_video_path: str
    audio_path: str
    cue_path: str = ""
    cues: list[DialogueCueOutput] = Field(default_factory=list)
    character_names: list[str] = Field(default_factory=list)
    character_images: list[str] = Field(default_factory=list)
    output_path: str


class LipSyncAlignerOutput(BaseModel):
    video_path: str
    audio_path: str
    frame_dir: str = ""
    alignment_path: str = ""


FORMANTS = {
    "a": (800, 1200),
    "e": (500, 1900),
    "i": (350, 2300),
    "o": (450, 900),
    "u": (325, 700),
}

DEFAULT_MOUTH_ANCHORS = {
    "mira": {"x": 0.58, "y": 0.51, "scale": 0.76, "source": "calibrated-character-reference"},
    "rowan": {"x": 0.50, "y": 0.27, "scale": 0.58, "source": "calibrated-character-reference"},
}

DEFAULT_EDGE_VOICES = {
    "mira": "en-US-JennyNeural",
    "rowan": "en-US-GuyNeural",
}

EDGE_VOICE_FALLBACKS = [
    "en-US-JennyNeural",
    "en-US-GuyNeural",
]

EDGE_TTS_SAMPLE_RATE = 24000


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _round_time(value: float) -> float:
    return round(value, 3)


def _frame_dir_for(output_path: Path) -> Path:
    return output_path.parent / "frames" / output_path.stem


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _speaker_seed(name: str) -> int:
    return sum(ord(ch) for ch in name.lower())


def _character_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _emotion_rate(emotion: str) -> int:
    text = emotion.lower()
    if any(word in text for word in ["urgent", "excited", "challenging", "angry", "resentful"]):
        return 2
    if any(word in text for word in ["sad", "guarded", "resolved", "quiet", "focused"]):
        return -1
    if any(word in text for word in ["determined", "confident"]):
        return 1
    return 0


def _edge_rate(emotion: str) -> str:
    rate = _emotion_rate(emotion)
    if rate >= 2:
        return "+8%"
    if rate == 1:
        return "+4%"
    if rate <= -1:
        return "-6%"
    return "+0%"


def _edge_pitch(emotion: str) -> str:
    text = emotion.lower()
    if any(word in text for word in ["challenging", "angry", "resentful", "urgent"]):
        return "+2Hz"
    if any(word in text for word in ["guarded", "sad", "resolved", "focused"]):
        return "-2Hz"
    return "+0Hz"


def _load_voice_map(output_path: Path) -> dict[str, str]:
    voice_map = dict(DEFAULT_EDGE_VOICES)
    candidate_paths = [
        output_path.parent / "voice_map.json",
        Path("outputs") / "voice_map.json",
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for name, voice in payload.items():
            if isinstance(name, str) and isinstance(voice, str) and voice.strip():
                voice_map[_character_slug(name)] = voice.strip()
    return voice_map


def _select_edge_voice(character: str, voice_map: dict[str, str]) -> str:
    slug = _character_slug(character)
    if slug in voice_map:
        return voice_map[slug]
    return EDGE_VOICE_FALLBACKS[_speaker_seed(character) % len(EDGE_VOICE_FALLBACKS)]


def _clean_dialogue_for_tts(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2014": ", ",
        "\u2013": ", ",
        "&": " and ",
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\.{3,}", ", ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])(?=[^\s])", r"\1 ", text)
    text = re.sub(r"\bAI\b", "A I", text)
    text = re.sub(r"\bTTS\b", "text to speech", text)
    if text and text[-1] not in ".!?":
        text += "."
    return text


async def _edge_tts_save(text: str, voice: str, rate: str, pitch: str, output_path: Path) -> None:
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("edge-tts is not installed. Run pip install -r requirements.txt.") from exc

    communicate = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch)
    await communicate.save(str(output_path))


def _run_edge_tts(text: str, voice: str, rate: str, pitch: str, output_path: Path) -> None:
    asyncio.run(_edge_tts_save(text, voice, rate, pitch, output_path))


def _convert_audio_to_wav(input_path: Path, output_path: Path, sample_rate: int = EDGE_TTS_SAMPLE_RATE) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        str(output_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0 or not output_path.exists():
        message = result.stderr.strip() or result.stdout.strip() or "FFmpeg failed to convert Edge-TTS audio."
        raise RuntimeError(message)


def _system_speech_synthesize_line(text: str, output_path: Path, speaker: str, scene_number: int, emotion: str) -> str:
    powershell = shutil.which("powershell") or shutil.which("powershell.exe")
    if not powershell:
        raise RuntimeError("Windows PowerShell was not found for local SAPI speech synthesis.")

    voice_index = (_speaker_seed(speaker) + scene_number) % 32
    rate = _emotion_rate(emotion)
    script = f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = {rate}
$synth.Volume = 100
$voices = @($synth.GetInstalledVoices() | Where-Object {{ $_.Enabled }})
$selectedVoice = $null
if ($voices.Count -gt 0) {{
    $selectedVoice = $voices[{voice_index} % $voices.Count].VoiceInfo.Name
    $synth.SelectVoice($selectedVoice)
}}
$synth.SetOutputToWaveFile({_ps_quote(str(output_path))})
$synth.Speak({_ps_quote(text)})
$synth.Dispose()
Write-Output $selectedVoice
"""
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    result = subprocess.run(
        [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    if result.returncode != 0 or not output_path.exists():
        message = result.stderr.strip() or result.stdout.strip() or "Unknown SAPI synthesis failure."
        raise RuntimeError(message)
    return result.stdout.strip() or "Windows SAPI"


def _vbs_quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _vbs_sapi_synthesize_line(text: str, output_path: Path, speaker: str, scene_number: int, emotion: str) -> str:
    cscript = shutil.which("cscript") or shutil.which("cscript.exe")
    if not cscript:
        raise RuntimeError("cscript.exe was not found for local SAPI COM speech synthesis.")

    voice_index = (_speaker_seed(speaker) + scene_number) % 32
    rate = _emotion_rate(emotion)
    script_path = output_path.with_suffix(".tts.vbs")
    script = f"""
Dim voice, voices, stream, format, voiceIndex
Set voice = CreateObject("SAPI.SpVoice")
Set voices = voice.GetVoices
voiceIndex = {voice_index}
If voices.Count > 0 Then
    Set voice.Voice = voices.Item(voiceIndex Mod voices.Count)
End If
voice.Rate = {rate}
voice.Volume = 100
Set stream = CreateObject("SAPI.SpFileStream")
Set format = CreateObject("SAPI.SpAudioFormat")
format.Type = 22
Set stream.Format = format
stream.Open {_vbs_quote(str(output_path))}, 3, False
Set voice.AudioOutputStream = stream
voice.Speak {_vbs_quote(text)}, 0
voice.WaitUntilDone 30000
Set voice.AudioOutputStream = Nothing
stream.Close
WScript.Echo voice.Voice.GetDescription
"""
    script_path.write_text(script, encoding="utf-16")
    try:
        result = subprocess.run(
            [cscript, "//NoLogo", str(script_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
    finally:
        if script_path.exists():
            script_path.unlink()
    if result.returncode != 0 or not output_path.exists():
        message = result.stderr.strip() or result.stdout.strip() or "Unknown SAPI COM synthesis failure."
        raise RuntimeError(message)
    return result.stdout.strip() or "Windows SAPI COM"


def _sapi_synthesize_line(text: str, output_path: Path, speaker: str, scene_number: int, emotion: str) -> str:
    errors: list[str] = []
    for synthesizer in (_system_speech_synthesize_line, _vbs_sapi_synthesize_line):
        try:
            voice_name = synthesizer(text, output_path, speaker, scene_number, emotion)
            _, samples = _read_wave_samples(output_path)
            if samples:
                return voice_name
            errors.append(f"{synthesizer.__name__}: produced an empty WAV file")
            if output_path.exists():
                output_path.unlink()
        except Exception as exc:
            errors.append(f"{synthesizer.__name__}: {exc}")
    raise RuntimeError(" | ".join(errors))


def _read_wave_samples(path: Path) -> tuple[int, list[int]]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        raw = np.frombuffer(frames, dtype=np.uint8).astype(np.int16)
        samples = (raw - 128) * 256
    elif sample_width == 2:
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.int16)
    elif sample_width == 4:
        raw = np.frombuffer(frames, dtype=np.int32).astype(np.float32)
        samples = np.clip(raw / 65536.0, -32768, 32767).astype(np.int16)
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1 and len(samples):
        samples = samples.reshape((-1, channels)).mean(axis=1).astype(np.int16)

    return sample_rate, samples.astype(np.int16).tolist()


def _resample_samples(samples: list[int], source_rate: int, target_rate: int) -> list[int]:
    if source_rate == target_rate or not samples:
        return samples
    duration = len(samples) / source_rate
    target_count = max(1, int(duration * target_rate))
    source_x = np.linspace(0.0, duration, num=len(samples), endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_count, endpoint=False)
    resampled = np.interp(target_x, source_x, np.asarray(samples, dtype=np.float32))
    return np.clip(resampled, -32768, 32767).astype(np.int16).tolist()


def _audio_duration_s(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as wav_file:
        return wav_file.getnframes() / float(wav_file.getframerate())


def _scene_palette(scene_number: int, scene_text: str) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    text = scene_text.lower()
    if any(word in text for word in ["harbor", "coastal", "shipyard", "water", "dock"]):
        return (26, 55, 92), (242, 156, 88), (255, 230, 184)
    if any(word in text for word in ["control room", "machine shop", "chamber", "interior", "night"]):
        return (25, 28, 46), (132, 74, 128), (88, 190, 220)
    if any(word in text for word in ["lantern", "garden", "sky", "floating", "moon"]):
        return (72, 82, 136), (236, 168, 112), (255, 238, 190)
    palettes = {
        1: ((49, 72, 116), (232, 166, 92), (255, 226, 184)),
        2: ((39, 41, 70), (119, 82, 140), (116, 196, 224)),
        3: ((54, 78, 96), (244, 180, 132), (225, 235, 245)),
    }
    return palettes.get(scene_number, ((42, 60, 92), (188, 144, 86), (228, 233, 238)))


def _keyword_flags(scene_text: str) -> dict[str, bool]:
    text = scene_text.lower()
    return {
        "water": any(word in text for word in ["harbor", "coastal", "shipyard", "dock", "water", "surf", "breakwater"]),
        "interior": any(word in text for word in ["int.", "interior", "room", "shop", "control", "chamber"]),
        "fantasy": any(word in text for word in ["lantern", "floating", "sky", "moon", "garden", "crystal"]),
        "storm": any(word in text for word in ["storm", "surge", "rain", "fog", "mist", "warning"]),
    }


def _dominant_vowel(word: str) -> str:
    for ch in word.lower():
        if ch in FORMANTS:
            return ch
    return "a"


def _character_pitch(name: str, scene_number: int) -> int:
    return 105 + ((sum(ord(c) for c in name.lower()) + scene_number * 13) % 75)


def _append_segment(signal: list[int], sample_rate: int, frequency: float, duration_s: float, vowel: str, gain: float) -> None:
    frames = max(1, int(sample_rate * duration_s))
    f1, f2 = FORMANTS.get(vowel, FORMANTS["a"])
    for i in range(frames):
        t = i / sample_rate
        envelope = math.sin(math.pi * i / frames) ** 1.3
        voiced = (
            0.60 * math.sin(2 * math.pi * frequency * t)
            + 0.25 * math.sin(2 * math.pi * frequency * 2.0 * t)
            + 0.10 * math.sin(2 * math.pi * frequency * 3.0 * t)
        )
        formant = 0.16 * math.sin(2 * math.pi * f1 * t) + 0.08 * math.sin(2 * math.pi * f2 * t)
        breath = 0.02 * math.sin(2 * math.pi * 35 * t)
        sample = int(24000 * gain * envelope * (voiced + formant + breath))
        signal.append(max(-32767, min(32767, sample)))


def _append_noise(signal: list[int], sample_rate: int, duration_s: float, strength: float) -> None:
    frames = max(1, int(sample_rate * duration_s))
    for i in range(frames):
        envelope = math.sin(math.pi * i / frames) ** 1.1
        noise = math.sin(i * 12.9898) * 43758.5453
        noise = noise - math.floor(noise)
        sample = int((noise * 2 - 1) * 8000 * strength * envelope)
        signal.append(max(-32767, min(32767, sample)))


def _dialogue_blocks(transcript: str) -> list[tuple[str, str]]:
    matches = re.findall(r"([A-Za-z][A-Za-z0-9_ ]+):\s*(.*?)(?=(?:[A-Za-z][A-Za-z0-9_ ]+:)|$)", transcript)
    if matches:
        return [(name.strip(), text.strip()) for name, text in matches if text.strip()]
    return [("Narrator", transcript.strip())] if transcript.strip() else []


def _normalize_dialogue_lines(dialogue: list[DialogueSynthesisLine], transcript: str) -> list[DialogueSynthesisLine]:
    if dialogue:
        return dialogue
    return [
        DialogueSynthesisLine(line_index=index, character=speaker, text=text, emotion="neutral")
        for index, (speaker, text) in enumerate(_dialogue_blocks(transcript))
    ]


def _append_silence(signal: list[int], sample_rate: int, duration_s: float) -> None:
    signal.extend([0] * int(sample_rate * duration_s))


def _synthesize_formant_dialogue(
    dialogue: list[DialogueSynthesisLine],
    scene_number: int,
) -> tuple[int, list[int], list[DialogueCueOutput]]:
    sample_rate = 22050
    signal: list[int] = []
    cues: list[DialogueCueOutput] = []

    for line in dialogue:
        base_pitch = _character_pitch(line.character, scene_number)
        start_s = len(signal) / sample_rate
        words = re.findall(r"[A-Za-z']+|[.,!?;]", line.text)
        if not words:
            _append_silence(signal, sample_rate, 0.3)
            continue
        for token in words:
            if token in ".,!?;":
                _append_silence(signal, sample_rate, 0.12)
                continue
            vowel = _dominant_vowel(token)
            duration = min(0.22, 0.08 + len(token) * 0.018)
            _append_segment(signal, sample_rate, base_pitch, duration, vowel, gain=0.95)
            if token[-1].lower() in {"s", "f", "h", "x"}:
                _append_noise(signal, sample_rate, 0.025, strength=0.5)
            _append_silence(signal, sample_rate, 0.03)
        end_s = len(signal) / sample_rate
        cues.append(
            DialogueCueOutput(
                line_index=line.line_index,
                character=line.character,
                text=line.text,
                emotion=line.emotion,
                start_s=_round_time(start_s),
                end_s=_round_time(end_s),
                duration_s=_round_time(end_s - start_s),
                voice_name="procedural-formant",
                backend="offline-formant-fallback",
            )
        )
        _append_silence(signal, sample_rate, 0.22)

    if not signal:
        signal = [0] * int(sample_rate * 1.5)

    return sample_rate, signal, cues


def _synthesize_sapi_dialogue(
    dialogue: list[DialogueSynthesisLine],
    scene_number: int,
    output_path: Path,
) -> tuple[int, list[int], list[DialogueCueOutput]]:
    sample_rate: int | None = None
    signal: list[int] = []
    cues: list[DialogueCueOutput] = []

    with tempfile.TemporaryDirectory(prefix=f"scene_{scene_number:02d}_tts_", dir=str(output_path.parent)) as temp_dir:
        temp_root = Path(temp_dir)
        for index, line in enumerate(dialogue):
            line_path = temp_root / f"line_{index:03d}.wav"
            voice_name = _sapi_synthesize_line(line.text, line_path, line.character, scene_number, line.emotion)
            line_rate, line_samples = _read_wave_samples(line_path)
            if sample_rate is None:
                sample_rate = line_rate
            if line_rate != sample_rate:
                line_samples = _resample_samples(line_samples, line_rate, sample_rate)

            start_s = len(signal) / sample_rate
            signal.extend(line_samples)
            end_s = len(signal) / sample_rate
            cues.append(
                DialogueCueOutput(
                    line_index=line.line_index,
                    character=line.character,
                    text=line.text,
                    emotion=line.emotion,
                    start_s=_round_time(start_s),
                    end_s=_round_time(end_s),
                    duration_s=_round_time(end_s - start_s),
                    voice_name=voice_name,
                    backend="windows-sapi",
                )
            )
            _append_silence(signal, sample_rate, 0.28)

    if sample_rate is None:
        raise RuntimeError("No dialogue lines were available for SAPI synthesis.")
    return sample_rate, signal, cues


def _synthesize_edge_dialogue(
    dialogue: list[DialogueSynthesisLine],
    scene_number: int,
    output_path: Path,
) -> tuple[int, list[int], list[DialogueCueOutput]]:
    voice_map = _load_voice_map(output_path)
    sample_rate = EDGE_TTS_SAMPLE_RATE
    signal: list[int] = []
    cues: list[DialogueCueOutput] = []

    with tempfile.TemporaryDirectory(prefix=f"scene_{scene_number:02d}_edge_tts_", dir=str(output_path.parent)) as temp_dir:
        temp_root = Path(temp_dir)
        for index, line in enumerate(dialogue):
            cleaned_text = _clean_dialogue_for_tts(line.text)
            if not cleaned_text:
                continue

            voice_name = _select_edge_voice(line.character, voice_map)
            mp3_path = temp_root / f"line_{index:03d}.mp3"
            wav_path = temp_root / f"line_{index:03d}.wav"

            _run_edge_tts(
                cleaned_text,
                voice_name,
                _edge_rate(line.emotion),
                _edge_pitch(line.emotion),
                mp3_path,
            )
            _convert_audio_to_wav(mp3_path, wav_path, sample_rate=sample_rate)
            line_rate, line_samples = _read_wave_samples(wav_path)
            if line_rate != sample_rate:
                line_samples = _resample_samples(line_samples, line_rate, sample_rate)
            if not line_samples:
                raise RuntimeError(f"Edge-TTS generated empty audio for {line.character}.")

            start_s = len(signal) / sample_rate
            signal.extend(line_samples)
            end_s = len(signal) / sample_rate
            cues.append(
                DialogueCueOutput(
                    line_index=line.line_index,
                    character=line.character,
                    text=cleaned_text,
                    emotion=line.emotion,
                    start_s=_round_time(start_s),
                    end_s=_round_time(end_s),
                    duration_s=_round_time(end_s - start_s),
                    voice_name=voice_name,
                    backend="edge-tts",
                )
            )
            _append_silence(signal, sample_rate, 0.32)

    if not signal:
        raise RuntimeError("No dialogue lines were available for Edge-TTS synthesis.")
    return sample_rate, signal, cues


def _synthesize_dialogue_audio(
    dialogue: list[DialogueSynthesisLine],
    transcript: str,
    scene_number: int,
    output_path: Path,
) -> tuple[int, list[int], list[DialogueCueOutput]]:
    normalized = _normalize_dialogue_lines(dialogue, transcript)
    if not normalized:
        normalized = [DialogueSynthesisLine(line_index=0, character="Narrator", text="No dialogue.", emotion="neutral")]

    try:
        return _synthesize_edge_dialogue(normalized, scene_number, output_path)
    except Exception:
        pass

    try:
        return _synthesize_sapi_dialogue(normalized, scene_number, output_path)
    except Exception:
        return _synthesize_formant_dialogue(normalized, scene_number)


def _write_wave(path: Path, sample_rate: int, signal: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(struct.pack("<h", sample) for sample in signal))


def _fit_portrait(image_path: str, size: tuple[int, int], frame_index: int, total_frames: int) -> Image.Image:
    source = Image.open(image_path).convert("RGBA")
    pan_x = 0.48 + 0.05 * math.sin((frame_index / max(total_frames, 1)) * math.tau)
    pan_y = 0.30 + 0.03 * math.cos((frame_index / max(total_frames, 1)) * math.tau)
    portrait = ImageOps.fit(source, size, method=Image.Resampling.LANCZOS, centering=(pan_x, pan_y))
    mask = Image.new("L", size, 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.ellipse((18, 8, size[0] - 18, size[1] - 8), fill=220)
    mdraw.rectangle((size[0] * 0.18, size[1] * 0.48, size[0] * 0.82, size[1] - 22), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(26))
    portrait.putalpha(mask)
    return portrait


def _draw_background(scene_number: int, scene_text: str, frame_index: int, total_frames: int, size: tuple[int, int]) -> Image.Image:
    width, height = size
    top, bottom, accent = _scene_palette(scene_number, scene_text)
    flags = _keyword_flags(scene_text)

    image = Image.new("RGBA", size, top + (255,))
    draw = ImageDraw.Draw(image, "RGBA")

    for y in range(height):
        ratio = y / max(height - 1, 1)
        color = tuple(int(top[i] * (1 - ratio) + bottom[i] * ratio) for i in range(3))
        draw.line((0, y, width, y), fill=color + (255,))

    glow_x = int(width * (0.16 + 0.1 * math.sin(frame_index / max(total_frames, 1) * math.tau)))
    glow_y = int(height * 0.22)
    glow_r = int(120 + 18 * math.sin(frame_index / max(total_frames, 1) * math.tau))
    draw.ellipse((glow_x - glow_r, glow_y - glow_r, glow_x + glow_r, glow_y + glow_r), fill=accent + (110,))

    if flags["water"]:
        horizon = int(height * 0.68)
        draw.rectangle((0, horizon, width, height), fill=(18, 34, 58, 110))
        for idx in range(10):
            y = horizon + idx * 16 + int(4 * math.sin(frame_index * 0.22 + idx))
            draw.line((0, y, width, y), fill=(220, 236, 255, 30), width=2)

    if flags["interior"]:
        for idx in range(7):
            x = 120 + idx * 160
            draw.rounded_rectangle((x, 90, x + 74, height - 120), radius=18, fill=(24, 30, 52, 68))
        for idx in range(6):
            y = 140 + idx * 84
            draw.line((100, y, width - 100, y), fill=(120, 180, 210, 36), width=2)

    if flags["fantasy"]:
        for idx in range(16):
            phase = frame_index * 0.04 + idx
            x = int((idx * 89 + 70) % width)
            y = int(height * (0.18 + 0.45 * ((math.sin(phase) + 1) / 2)))
            r = 10 + (idx % 4) * 4
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 224, 162, 90))

    if flags["storm"]:
        fog = Image.new("RGBA", size, (0, 0, 0, 0))
        fog_draw = ImageDraw.Draw(fog, "RGBA")
        for idx in range(5):
            y = int(height * (0.25 + idx * 0.13))
            offset = int(30 * math.sin(frame_index * 0.15 + idx))
            fog_draw.ellipse((-120 + offset, y, width + 120 + offset, y + 95), fill=(215, 225, 235, 34))
        image = Image.alpha_composite(image, fog.filter(ImageFilter.GaussianBlur(22)))

    vignette = Image.new("RGBA", size, (0, 0, 0, 0))
    vdraw = ImageDraw.Draw(vignette, "RGBA")
    vdraw.ellipse((-120, -60, width + 120, height + 60), fill=(0, 0, 0, 0))
    vdraw.rectangle((0, 0, width, 90), fill=(0, 0, 0, 26))
    vdraw.rectangle((0, height - 120, width, height), fill=(0, 0, 0, 54))
    vignette = vignette.filter(ImageFilter.GaussianBlur(34))
    return Image.alpha_composite(image, vignette)


def _portrait_layout(count: int, mode: str, size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    width, height = size
    if mode == "face_swap":
        box_w, box_h, y = 340, 470, int(height * 0.20)
    else:
        box_w, box_h, y = 300, 420, int(height * 0.22)

    if count <= 1:
        return [(width // 2 - box_w // 2, y, width // 2 + box_w // 2, y + box_h)]
    if count == 2:
        return [
            (int(width * 0.12), y + 30, int(width * 0.12) + box_w, y + 30 + box_h),
            (int(width * 0.56), y - 10, int(width * 0.56) + box_w, y - 10 + box_h),
        ]
    return [
        (70, y + 55, 70 + int(box_w * 0.82), y + 55 + int(box_h * 0.82)),
        (width // 2 - box_w // 2, y - 5, width // 2 + box_w // 2, y - 5 + box_h),
        (width - 70 - int(box_w * 0.82), y + 55, width - 70, y + 55 + int(box_h * 0.82)),
    ]


def _compose_scene_frame(
    scene_number: int,
    scene_text: str,
    character_images: list[str],
    frame_index: int,
    total_frames: int,
    mode: str,
) -> Image.Image:
    size = (1280, 720)
    frame = _draw_background(scene_number, scene_text, frame_index, total_frames, size)

    if not character_images:
        return frame.convert("RGB")

    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    positions = _portrait_layout(min(len(character_images), 3), mode, size)
    selected = character_images[: len(positions)]

    for idx, (image_path, box) in enumerate(zip(selected, positions)):
        if not Path(image_path).exists():
            continue
        x1, y1, x2, y2 = box
        bob = int(6 * math.sin((frame_index / max(total_frames, 1)) * math.tau + idx))
        portrait = _fit_portrait(image_path, (x2 - x1, y2 - y1), frame_index + idx * 6, total_frames)
        glow = Image.new("RGBA", portrait.size, (0, 0, 0, 0))
        glow.putalpha(portrait.getchannel("A").point(lambda p: int(p * 0.25)))
        glow = glow.filter(ImageFilter.GaussianBlur(28))
        overlay.alpha_composite(glow, (x1 + 4, y1 + 10 + bob))
        overlay.alpha_composite(portrait, (x1, y1 + bob))

    light = Image.new("RGBA", size, (0, 0, 0, 0))
    ldraw = ImageDraw.Draw(light, "RGBA")
    ldraw.ellipse((820, 40, 1220, 390), fill=(255, 214, 164, 34))
    ldraw.ellipse((20, 70, 320, 350), fill=(150, 200, 255, 20))
    frame = Image.alpha_composite(frame, overlay)
    frame = Image.alpha_composite(frame, light.filter(ImageFilter.GaussianBlur(30)))
    return frame.convert("RGB")


def _write_video_from_frames(
    output_path: Path,
    frames: list[Image.Image],
    fps: int = 12,
    frame_dir: Path | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if frame_dir:
        frame_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in frame_dir.glob("*.png"):
            old_frame.unlink()
    with imageio.get_writer(str(output_path), fps=fps, codec="libx264", format="FFMPEG") as writer:
        for index, frame in enumerate(frames):
            rgb = frame.convert("RGB")
            if frame_dir:
                rgb.save(frame_dir / f"frame_{index:04d}.png")
            writer.append_data(np.array(rgb))


def _render_scene_video(
    output_path: Path,
    scene_number: int,
    scene_text: str,
    character_images: list[str],
    mode: str,
) -> Path:
    total_frames = 72
    frames = [
        _compose_scene_frame(scene_number, scene_text, character_images, idx, total_frames, mode)
        for idx in range(total_frames)
    ]
    frame_dir = _frame_dir_for(output_path)
    _write_video_from_frames(output_path, frames, fps=12, frame_dir=frame_dir)
    return frame_dir


def _read_video_frames(path: Path) -> tuple[list[Image.Image], float]:
    reader = imageio.get_reader(str(path), format="ffmpeg")
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 12))
    frames = [Image.fromarray(frame).convert("RGB") for frame in reader]
    reader.close()
    return frames, fps


def _audio_envelope_by_frame(
    audio_path: Path,
    fps: float,
    frame_count: int,
    cues: list[DialogueCueOutput] | None = None,
) -> tuple[list[float], dict]:
    sample_rate, raw_samples = _read_wave_samples(audio_path)
    samples = np.asarray(raw_samples, dtype=np.float32)
    if len(samples) == 0 or frame_count <= 0:
        return [0.0] * max(frame_count, 1), {
            "sample_rate_hz": sample_rate,
            "samples_per_video_frame": 0.0,
            "analysis": "empty-audio",
        }

    fps = max(fps, 1.0)
    samples_per_frame = sample_rate / fps
    rms_values: list[float] = []
    frame_windows: list[dict[str, int]] = []
    active_speakers: list[str] = []
    speaker_frame_indices: dict[str, list[int]] = {}
    cue_list = cues or []

    for frame_index in range(frame_count):
        start = min(len(samples), int(round(frame_index * samples_per_frame)))
        end = min(len(samples), int(round((frame_index + 1) * samples_per_frame)))
        window = samples[start:end]
        frame_windows.append({"frame": frame_index, "sample_start": start, "sample_end": end})
        active = _active_cue(cue_list, (frame_index + 0.5) / fps)
        speaker_key = _character_slug(active.character) if active else ""
        active_speakers.append(speaker_key)
        if speaker_key:
            speaker_frame_indices.setdefault(speaker_key, []).append(frame_index)
        if len(window) == 0:
            rms_values.append(0.0)
            continue
        rms_values.append(float(np.sqrt(np.mean(np.square(window))) / 32767.0))

    rms = np.asarray(rms_values, dtype=np.float32)
    global_min = float(np.min(rms))
    global_max = float(np.max(rms))
    speaker_ranges: dict[str, dict[str, float | int]] = {}
    for speaker_key, indices in speaker_frame_indices.items():
        speaker_rms = rms[indices]
        speaker_ranges[speaker_key] = {
            "min_rms": round(float(np.min(speaker_rms)), 6),
            "max_rms": round(float(np.max(speaker_rms)), 6),
            "frame_count": len(indices),
        }

    mouth_values: list[float] = []
    for frame_index, rms_value in enumerate(rms):
        speaker_key = active_speakers[frame_index]
        if speaker_key and speaker_key in speaker_ranges:
            min_rms = float(speaker_ranges[speaker_key]["min_rms"])
            max_rms = float(speaker_ranges[speaker_key]["max_rms"])
        else:
            min_rms = global_min
            max_rms = global_max

        if max_rms <= min_rms + 1e-8:
            normalized = 0.0
        else:
            normalized = (float(rms_value) - min_rms) / (max_rms - min_rms)
        normalized = min(1.0, max(0.0, normalized))

        mouth_open = math.pow(normalized, 0.5)
        mouth_values.append(0.1 + 0.8 * mouth_open)

    envelope: list[float] = []
    previous = mouth_values[0] if mouth_values else 0.1
    for current in mouth_values:
        smoothed = 0.6 * previous + 0.4 * current
        envelope.append(min(1.0, max(0.0, smoothed)))
        previous = smoothed

    stats = {
        "sample_rate_hz": sample_rate,
        "video_fps": round(fps, 3),
        "samples_per_video_frame": round(samples_per_frame, 3),
        "audio_sample_count": int(len(samples)),
        "frame_count": int(frame_count),
        "min_rms": round(global_min, 6),
        "max_rms": round(global_max, 6),
        "speaker_ranges": speaker_ranges,
        "normalization": "min-max-rms-per-frame-with-sqrt-scaling-and-speaker-ranges",
        "smoothing": "smooth = 0.6 * prev + 0.4 * current",
        "analysis": "one-rms-audio-window-per-video-frame",
        "frame_windows_preview": frame_windows[:5],
    }
    return [float(value) for value in envelope], stats


def _fit_frames_to_duration(frames: list[Image.Image], fps: float, duration_s: float) -> list[Image.Image]:
    target_count = max(1, int(math.ceil(duration_s * fps)))
    if not frames:
        return [Image.new("RGB", (1280, 720), (10, 10, 10)) for _ in range(target_count)]
    if len(frames) == target_count:
        return [frame.copy() for frame in frames]
    adjusted: list[Image.Image] = []
    for index in range(target_count):
        source_index = min(len(frames) - 1, int(index * len(frames) / target_count))
        adjusted.append(frames[source_index].copy())
    return adjusted


def _load_cues(cues: list[DialogueCueOutput], cue_path: str) -> list[DialogueCueOutput]:
    if cues:
        return cues
    path = Path(cue_path) if cue_path else None
    if path and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [DialogueCueOutput(**item) for item in payload.get("cues", [])]
    return []


def _load_mouth_anchors(context: dict) -> dict[str, dict]:
    anchors = dict(DEFAULT_MOUTH_ANCHORS)
    output_dir = Path(context.get("output_dir", "outputs"))
    anchor_path = output_dir / "mouth_anchors.json"
    if anchor_path.exists():
        try:
            payload = json.loads(anchor_path.read_text(encoding="utf-8"))
            for name, values in payload.items():
                if isinstance(values, dict) and "x" in values and "y" in values:
                    anchors[_character_slug(name)] = values
        except json.JSONDecodeError:
            pass
    return anchors


def _active_cue(cues: list[DialogueCueOutput], time_s: float) -> DialogueCueOutput | None:
    for cue in cues:
        if cue.start_s <= time_s <= cue.end_s:
            return cue
    return None


def _cue_vowel(cue: DialogueCueOutput, time_s: float) -> str:
    words = re.findall(r"[A-Za-z']+", cue.text)
    if not words or cue.duration_s <= 0:
        return "a"
    progress = min(0.999, max(0.0, (time_s - cue.start_s) / cue.duration_s))
    word = words[min(len(words) - 1, int(progress * len(words)))]
    return _dominant_vowel(word)


def _draw_subtitle(frame: Image.Image, cue: DialogueCueOutput | None) -> None:
    if not cue:
        return
    draw = ImageDraw.Draw(frame, "RGBA")
    width, height = frame.size
    text = f"{cue.character}: {cue.text}"
    max_chars = 88
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    bbox = draw.textbbox((0, 0), text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad_x, pad_y = 18, 12
    x1 = max(32, (width - text_w) // 2 - pad_x)
    y1 = height - 78
    x2 = min(width - 32, (width + text_w) // 2 + pad_x)
    y2 = y1 + text_h + pad_y * 2
    draw.rounded_rectangle((x1, y1, x2, y2), radius=8, fill=(8, 10, 14, 185), outline=(255, 255, 255, 40))
    draw.text((x1 + pad_x, y1 + pad_y - 2), text, fill=(250, 246, 235, 255))
def _mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True)


def _video_info(path: Path) -> tuple[float, tuple[int, int], float]:
    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        meta = reader.get_meta_data()
        fps = float(meta.get("fps", 12))
        first_frame = reader.get_data(0)
        frame_count = 0
        for _ in reader:
            frame_count += 1
    finally:
        reader.close()
    height, width = first_frame.shape[:2]
    duration = frame_count / max(fps, 1.0)
    return fps, (width, height), duration


OUTER_LIP_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]

INNER_LIP_LANDMARKS = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
]

UPPER_LIP_INNER_CURVE = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_LIP_INNER_CURVE = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
UPPER_LIP_OUTER_CURVE = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP_OUTER_CURVE = [61, 146, 91, 181, 84, 17, 18, 314, 405, 321, 375, 291]
UPPER_LIP_MEASURE_POINTS = [13, 82, 312]
LOWER_LIP_MEASURE_POINTS = [14, 17, 18, 87, 317]
MOUTH_CORNER_LEFT = 61
MOUTH_CORNER_RIGHT = 291


def _create_face_mesh():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is required for lip sync. Install it with "
            "`pip install mediapipe` or `pip install -r requirements.txt`."
        ) from exc

    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=3,
        refine_landmarks=True,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    )


def _landmarks_to_pixels(landmarks, width: int, height: int) -> list[tuple[float, float]]:
    return [
        (
            min(width - 1, max(0, landmark.x * width)),
            min(height - 1, max(0, landmark.y * height)),
        )
        for landmark in landmarks
    ]


def _points_bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _expected_speaker_mouth(
    character_names: list[str],
    speaker: str,
    frame_size: tuple[int, int],
    context: dict,
) -> tuple[float, float] | None:
    if not speaker or not character_names:
        return None
    try:
        speaker_index = [name.lower() for name in character_names].index(speaker.lower())
    except ValueError:
        return None

    width, height = frame_size
    boxes = _portrait_layout(min(len(character_names), 3), "face_swap", (width, height))
    if speaker_index >= len(boxes):
        return None

    x1, y1, x2, y2 = boxes[speaker_index]
    anchor = _load_mouth_anchors(context).get(_character_slug(speaker), {})
    anchor_x = float(anchor.get("x", 0.5))
    anchor_y = float(anchor.get("y", 0.45))
    return x1 + (x2 - x1) * anchor_x, y1 + (y2 - y1) * anchor_y


def _select_face_landmarks(
    faces,
    frame_size: tuple[int, int],
    expected_mouth: tuple[float, float] | None,
):
    if not faces:
        return None

    width, height = frame_size
    best_face = None
    best_score = float("inf")
    for face in faces:
        pixels = _landmarks_to_pixels(face.landmark, width, height)
        lip_points = [pixels[index] for index in OUTER_LIP_LANDMARKS]
        face_bbox = _points_bbox(pixels)
        lip_center = _bbox_center(_points_bbox(lip_points))
        face_area = max(1.0, (face_bbox[2] - face_bbox[0]) * (face_bbox[3] - face_bbox[1]))

        if expected_mouth:
            dx = lip_center[0] - expected_mouth[0]
            dy = lip_center[1] - expected_mouth[1]
            score = dx * dx + dy * dy
        else:
            score = -face_area

        if score < best_score:
            best_score = score
            best_face = pixels

    return best_face


def _point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _mean_point(points: list[tuple[float, float]]) -> tuple[float, float]:
    return (
        sum(point[0] for point in points) / max(1, len(points)),
        sum(point[1] for point in points) / max(1, len(points)),
    )


def _smooth_number(current: float, previous: float | None, alpha: float = 0.70) -> float:
    if previous is None:
        return current
    return previous * alpha + current * (1.0 - alpha)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _ease_in_out(value: float) -> float:
    value = _clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _cue_edge_gate(cue: DialogueCueOutput, time_s: float) -> float:
    if cue.duration_s <= 0:
        return 1.0
    ramp_s = min(0.18, max(0.06, cue.duration_s * 0.10))
    opening = min(1.0, max(0.0, (time_s - cue.start_s) / ramp_s))
    closing = min(1.0, max(0.0, (cue.end_s - time_s) / ramp_s))
    return _ease_in_out(min(opening, closing))


def _speech_micro_offset(time_s: float, speaker_key: str, strength: float) -> float:
    if strength <= 0.02:
        return 0.0
    seed = (sum(ord(char) for char in speaker_key) % 113) / 113.0
    syllable = math.sin(math.tau * (6.4 * time_s + seed))
    flutter = math.sin(math.tau * (11.3 * time_s + seed * 0.41))
    return strength * (0.040 * syllable + 0.014 * flutter)


def _natural_speech_open(
    envelope_value: float,
    cue: DialogueCueOutput,
    time_s: float,
    speaker_key: str,
    smoothing_state: dict,
) -> tuple[float, float]:
    raw_open = _clamp(envelope_value, 0.0, 1.0)
    gate = _cue_edge_gate(cue, time_s)
    target = _clamp(raw_open * gate, 0.04 * gate, 0.92)

    target_key = f"{speaker_key}:openness_target"
    open_key = f"{speaker_key}:openness_interp"
    velocity_key = f"{speaker_key}:openness_velocity"
    target_smooth = _smooth_number(target, smoothing_state.get(target_key), alpha=0.18)
    previous_open = float(smoothing_state.get(open_key, target_smooth))
    previous_velocity = float(smoothing_state.get(velocity_key, 0.0))

    delta = target_smooth - previous_open
    velocity = previous_velocity * 0.34 + delta * 0.66
    interpolated = previous_open + delta * 0.72 + velocity * 0.22

    micro_strength = gate * (0.25 + interpolated * 0.55)
    micro = _speech_micro_offset(time_s, speaker_key, micro_strength)
    velocity_accent = _clamp(velocity * 0.34, -0.035, 0.045)
    final_open = _clamp(interpolated + micro + velocity_accent, 0.04 * gate, 0.92)

    previous_final = smoothing_state.get(f"{speaker_key}:openness")
    if gate > 0.20 and previous_final is not None and abs(final_open - float(previous_final)) < 0.003:
        phase = math.sin(math.tau * (7.1 * time_s + len(speaker_key) * 0.13))
        final_open = _clamp(final_open + phase * 0.006 * gate, 0.018 * gate, 0.84)

    smoothing_state[target_key] = target_smooth
    smoothing_state[open_key] = interpolated
    smoothing_state[velocity_key] = velocity
    smoothing_state[f"{speaker_key}:openness"] = final_open
    return final_open, velocity


def _mouth_axes(
    left_corner: tuple[float, float],
    right_corner: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float], float]:
    width = max(1.0, _point_distance(left_corner, right_corner))
    axis = ((right_corner[0] - left_corner[0]) / width, (right_corner[1] - left_corner[1]) / width)
    normal = (-axis[1], axis[0])
    return axis, normal, width


def _project_to_mouth_axes(
    point: tuple[float, float],
    origin: tuple[float, float],
    axis: tuple[float, float],
    normal: tuple[float, float],
) -> tuple[float, float]:
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]
    return dx * axis[0] + dy * axis[1], dx * normal[0] + dy * normal[1]


def _from_mouth_axes(
    origin: tuple[float, float],
    axis: tuple[float, float],
    normal: tuple[float, float],
    along: float,
    vertical: float,
) -> tuple[float, float]:
    return (
        origin[0] + axis[0] * along + normal[0] * vertical,
        origin[1] + axis[1] * along + normal[1] * vertical,
    )


def _lip_viseme_targets(
    vowel: str,
    speech_open: float,
    speech_velocity: float,
    corner_width: float,
    measured_gap: float,
) -> tuple[float, float]:
    speech_open = _clamp(speech_open, 0.0, 1.0)
    if vowel in {"o", "u"}:
        vertical_amount = 0.16
        horizontal_gain = 1.0 - speech_open * 0.10
    elif vowel in {"e", "i"}:
        vertical_amount = 0.10
        horizontal_gain = 1.0 + speech_open * 0.18
    else:
        vertical_amount = 0.20
        horizontal_gain = 1.0 + speech_open * 0.10

    velocity_lift = _clamp(speech_velocity, -0.10, 0.12) * corner_width * 0.18
    target_gap = max(measured_gap * 0.55, measured_gap + corner_width * vertical_amount * speech_open + velocity_lift)
    target_gap = min(measured_gap + corner_width * 0.21, max(measured_gap * 0.72, target_gap))
    horizontal_gain = _clamp(horizontal_gain + speech_velocity * 0.18, 0.90, 1.18)
    return target_gap, horizontal_gain


def _sample_lip_color(frame: Image.Image, points: list[tuple[float, float]]) -> tuple[int, int, int]:
    rgb = frame.convert("RGB")
    width, height = rgb.size
    samples = []
    for x, y in points:
        px = min(width - 1, max(0, int(round(x))))
        py = min(height - 1, max(0, int(round(y))))
        samples.append(rgb.getpixel((px, py)))
    if not samples:
        return 132, 52, 63
    return tuple(int(sum(sample[channel] for sample in samples) / len(samples)) for channel in range(3))


def _transform_lip_curve(
    points: list[tuple[float, float]],
    origin: tuple[float, float],
    axis: tuple[float, float],
    normal: tuple[float, float],
    half_width: float,
    horizontal_gain: float,
    open_delta: float,
    direction: float,
) -> list[tuple[float, float]]:
    transformed = []
    for point in points:
        along, vertical = _project_to_mouth_axes(point, origin, axis, normal)
        center_weight = max(0.0, 1.0 - min(1.0, abs(along) / max(1.0, half_width)) ** 1.7)
        transformed.append(
            _from_mouth_axes(
                origin,
                axis,
                normal,
                along * horizontal_gain,
                vertical + direction * open_delta * center_weight,
            )
        )
    return transformed


def _draw_landmark_lips(
    frame: Image.Image,
    face_pixels: list[tuple[float, float]],
    target_gap: float,
    measured_gap: float,
    baseline_gap: float,
    horizontal_gain: float,
) -> Image.Image:
    left_corner = face_pixels[MOUTH_CORNER_LEFT]
    right_corner = face_pixels[MOUTH_CORNER_RIGHT]
    axis, normal, corner_width = _mouth_axes(left_corner, right_corner)
    upper_center = _mean_point([face_pixels[index] for index in UPPER_LIP_MEASURE_POINTS])
    lower_center = _mean_point([face_pixels[index] for index in LOWER_LIP_MEASURE_POINTS])
    origin = _mean_point([upper_center, lower_center, left_corner, right_corner])
    half_width = corner_width / 2.0
    relative_measured_gap = max(0.0, measured_gap - baseline_gap)
    relative_target_gap = max(0.0, target_gap - baseline_gap)
    open_delta = max(-max(1.0, relative_measured_gap) * 0.65, relative_target_gap - relative_measured_gap)

    upper_inner = _transform_lip_curve(
        [face_pixels[index] for index in UPPER_LIP_INNER_CURVE],
        origin,
        axis,
        normal,
        half_width,
        horizontal_gain,
        open_delta * 0.46,
        -1.0,
    )
    lower_inner = _transform_lip_curve(
        [face_pixels[index] for index in LOWER_LIP_INNER_CURVE],
        origin,
        axis,
        normal,
        half_width,
        horizontal_gain,
        open_delta * 0.74,
        1.0,
    )
    upper_outer = _transform_lip_curve(
        [face_pixels[index] for index in UPPER_LIP_OUTER_CURVE],
        origin,
        axis,
        normal,
        half_width,
        horizontal_gain,
        open_delta * 0.34,
        -1.0,
    )
    lower_outer = _transform_lip_curve(
        [face_pixels[index] for index in LOWER_LIP_OUTER_CURVE],
        origin,
        axis,
        normal,
        half_width,
        horizontal_gain,
        open_delta * 0.62,
        1.0,
    )

    lip_color = _sample_lip_color(frame, [face_pixels[index] for index in OUTER_LIP_LANDMARKS])
    lip_shadow = tuple(max(0, int(channel * 0.64)) for channel in lip_color)
    lip_highlight = tuple(min(255, int(channel * 1.22 + 16)) for channel in lip_color)
    open_ratio = min(1.0, max(0.0, relative_target_gap / max(1.0, corner_width * 0.18)))

    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    mouth_polygon = upper_inner + list(reversed(lower_inner))
    if len(mouth_polygon) >= 3:
        draw.polygon(mouth_polygon, fill=(20, 4, 8, int(92 + open_ratio * 130)))

    line_width = max(2, int(corner_width * 0.026))
    inner_width = max(1, int(corner_width * 0.014))
    draw.line(upper_outer, fill=(*lip_shadow, 165), width=line_width)
    draw.line(lower_outer, fill=(*lip_highlight, 150), width=line_width)
    draw.line(upper_inner, fill=(*lip_shadow, 190), width=inner_width)
    draw.line(lower_inner, fill=(*lip_highlight, 130), width=inner_width)

    if open_ratio > 0.35:
        tongue_center = _from_mouth_axes(origin, axis, normal, 0.0, target_gap * 0.28)
        tongue_w = corner_width * (0.16 + open_ratio * 0.10)
        tongue_h = target_gap * 0.22
        draw.ellipse(
            (
                tongue_center[0] - tongue_w,
                tongue_center[1] - tongue_h,
                tongue_center[0] + tongue_w,
                tongue_center[1] + tongue_h,
            ),
            fill=(118, 42, 54, int(60 + open_ratio * 55)),
        )

    overlay = overlay.filter(ImageFilter.GaussianBlur(0.35))
    return Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")


def _apply_landmark_mouth_motion(
    frame: Image.Image,
    face_pixels: list[tuple[float, float]],
    speech_open: float,
    speech_velocity: float,
    vowel: str,
    smoothing_state: dict,
    speaker_key: str,
) -> Image.Image:
    upper_lip = _mean_point([face_pixels[index] for index in UPPER_LIP_MEASURE_POINTS])
    lower_lip = _mean_point([face_pixels[index] for index in LOWER_LIP_MEASURE_POINTS])
    measured_gap = _point_distance(upper_lip, lower_lip)
    corner_width = _point_distance(face_pixels[MOUTH_CORNER_LEFT], face_pixels[MOUTH_CORNER_RIGHT])
    baseline_key = f"{speaker_key}:baseline_gap"
    baseline_gap = float(smoothing_state.get(baseline_key, measured_gap))
    if baseline_key not in smoothing_state:
        smoothing_state[baseline_key] = baseline_gap

    relative_open = max(0.0, measured_gap - baseline_gap)
    relative_open_key = f"{speaker_key}:relative_open"
    relative_open = _smooth_number(relative_open, smoothing_state.get(relative_open_key), alpha=0.55)
    smoothing_state[relative_open_key] = relative_open
    normalized_relative_open = _clamp(relative_open / max(1.0, corner_width * 0.12), 0.0, 1.0)
    audio_relative_open = _clamp((speech_open - 0.1) / 0.8, 0.0, 1.0)
    effective_open = max(audio_relative_open, normalized_relative_open)

    target_gap, horizontal_gain = _lip_viseme_targets(vowel, effective_open, speech_velocity, corner_width, baseline_gap)

    gap_key = f"{speaker_key}:lip_gap"
    stretch_key = f"{speaker_key}:lip_stretch"
    gap_alpha = 0.32 if speech_velocity >= 0 else 0.46
    target_gap = _smooth_number(target_gap, smoothing_state.get(gap_key), alpha=gap_alpha)
    horizontal_gain = _smooth_number(horizontal_gain, smoothing_state.get(stretch_key), alpha=0.44)
    smoothing_state[gap_key] = target_gap
    smoothing_state[stretch_key] = horizontal_gain

    return _draw_landmark_lips(frame, face_pixels, target_gap, measured_gap, baseline_gap, horizontal_gain)


def _mediapipe_lip_sync_frame(
    frame: Image.Image,
    face_mesh,
    time_s: float,
    cues: list[DialogueCueOutput],
    envelope_value: float,
    character_names: list[str],
    context: dict,
    smoothing_state: dict,
) -> Image.Image:
    active = _active_cue(cues, time_s)
    if not active:
        _draw_subtitle(frame, None)
        return frame

    width, height = frame.size
    result = face_mesh.process(np.asarray(frame.convert("RGB")))
    faces = result.multi_face_landmarks or []
    expected_mouth = _expected_speaker_mouth(character_names, active.character, (width, height), context)
    face_pixels = _select_face_landmarks(faces, (width, height), expected_mouth)
    if face_pixels is None:
        _draw_subtitle(frame, active)
        return frame

    speaker_key = _character_slug(active.character)
    speech_open, speech_velocity = _natural_speech_open(envelope_value, active, time_s, speaker_key, smoothing_state)

    synced = _apply_landmark_mouth_motion(
        frame,
        face_pixels,
        speech_open,
        speech_velocity,
        _cue_vowel(active, time_s),
        smoothing_state,
        speaker_key,
    )
    _draw_subtitle(synced, active)
    return synced


@mcp_tool(
    name="get_task_graph",
    description="Break a scene into executable multimodal tasks and persist a task graph log.",
    tags=["phase2", "task-graph", "scene-parser"],
    input_model=GetTaskGraphInput,
    output_model=GetTaskGraphOutput,
)
def get_task_graph_tool(data: GetTaskGraphInput, _: dict) -> GetTaskGraphOutput:
    task_id = f"scene-{data.scene_number:02d}-{uuid4().hex[:8]}"
    breakdown = [
        f"Parse scene heading: {data.heading}",
        "Synthesize natural Edge-TTS dialogue audio and cue timings",
        "Generate raw cinematic scene video and frame sequence",
        "Validate and map character identities onto frames",
        "Run MediaPipe Face Mesh lip synchronization and mux final scene audio",
    ]
    output_path = Path(data.output_path)
    _write_json(
        output_path,
        {
            "task_id": task_id,
            "scene_number": data.scene_number,
            "heading": data.heading,
            "summary": data.summary,
            "characters": data.dialogue_characters,
            "task_breakdown": breakdown,
            "action": data.action,
            "visual_cues": data.visual_cues,
        },
    )
    return GetTaskGraphOutput(task_id=task_id, task_breakdown=breakdown, log_path=str(output_path))


@mcp_tool(
    name="voice_cloning_synthesizer",
    description="Generate an audible synthetic dialogue waveform for a scene.",
    tags=["phase2", "audio", "tts"],
    input_model=VoiceCloningSynthesizerInput,
    output_model=VoiceCloningSynthesizerOutput,
)
def voice_cloning_synthesizer_tool(data: VoiceCloningSynthesizerInput, _: dict) -> VoiceCloningSynthesizerOutput:
    output_path = Path(data.output_path)
    sample_rate, signal, cues = _synthesize_dialogue_audio(data.dialogue, data.transcript, data.scene_number, output_path)
    _write_wave(output_path, sample_rate, signal)
    cue_path = output_path.with_suffix(".cues.json")
    _write_json(
        cue_path,
        {
            "scene_number": data.scene_number,
            "audio_path": str(output_path),
            "transcript": data.transcript,
            "cues": [cue.model_dump() for cue in cues],
        },
    )
    return VoiceCloningSynthesizerOutput(
        audio_path=str(output_path),
        transcript=data.transcript,
        cue_path=str(cue_path),
        cues=cues,
    )


@mcp_tool(
    name="query_stock_footage",
    description="Return a reference style for characters or generate a cinematic scene video from the scene description.",
    tags=["character", "reference", "phase2", "video", "stock-footage"],
    input_model=QueryStockFootageInput,
    output_model=QueryStockFootageOutput,
)
def query_stock_footage_tool(data: QueryStockFootageInput, _: dict) -> QueryStockFootageOutput:
    reference_style = (
        f"{data.character_name}: cinematic portrait, close-up to mid-shot framing, "
        f"{data.appearance_description}"
    )

    if not data.output_path:
        return QueryStockFootageOutput(reference_style=reference_style)

    footage_prompt = (
        f"{data.scene_heading or ''}. {data.scene_summary or ''}. "
        f"{data.environment_description or data.appearance_description}. "
        f"{' '.join(data.visual_cues)}"
    ).strip()
    shot_plan = [
        "Establishing environment pass",
        "Character-emphasis portrait motion",
        "Atmospheric close shot with parallax",
    ]
    output_path = Path(data.output_path)
    frame_dir = _render_scene_video(
        output_path,
        data.scene_number or 1,
        footage_prompt,
        data.character_images,
        mode="raw",
    )
    return QueryStockFootageOutput(
        reference_style=reference_style,
        footage_prompt=footage_prompt,
        shot_plan=shot_plan,
        video_path=str(output_path),
        frame_dir=str(frame_dir),
    )


@mcp_tool(
    name="identity_validator",
    description="Validate that referenced character identities are available before face mapping.",
    tags=["phase2", "video", "identity"],
    input_model=IdentityValidatorInput,
    output_model=IdentityValidatorOutput,
)
def identity_validator_tool(data: IdentityValidatorInput, _: dict) -> IdentityValidatorOutput:
    missing = [path for path in data.character_images if not Path(path).exists()]
    if missing:
        return IdentityValidatorOutput(valid=False, notes=f"Missing character images: {missing}")
    return IdentityValidatorOutput(valid=True, notes=f"Validated identities for {', '.join(data.character_names)}")


@mcp_tool(
    name="face_swapper",
    description="Map character portrait identities onto the scene video frames.",
    tags=["phase2", "video", "face-swap"],
    input_model=FaceSwapperInput,
    output_model=FaceSwapperOutput,
)
def face_swapper_tool(data: FaceSwapperInput, _: dict) -> FaceSwapperOutput:
    output_path = Path(data.output_path)
    scene_text = data.scene_text or Path(data.source_video_path).stem.replace("_", " ")
    frame_dir = _render_scene_video(
        output_path,
        data.scene_number,
        scene_text,
        data.character_images,
        mode="face_swap",
    )
    return FaceSwapperOutput(
        video_path=str(output_path),
        mapped_characters=data.character_names,
        frame_dir=str(frame_dir),
    )


@mcp_tool(
    name="lip_sync_aligner",
    description="Use MediaPipe Face Mesh landmarks to anchor and animate the mouth, then mux scene audio into the final output.",
    tags=["phase2", "video", "audio", "lip-sync"],
    input_model=LipSyncAlignerInput,
    output_model=LipSyncAlignerOutput,
)
def lip_sync_aligner_tool(data: LipSyncAlignerInput, context: dict) -> LipSyncAlignerOutput:
    source_video = Path(data.source_video_path)
    audio_path = Path(data.audio_path)
    output_path = Path(data.output_path)

    temp_root = Path(context["output_dir"]) / "_phase2_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)

    frames, fps = _read_video_frames(source_video)
    cues = _load_cues(data.cues, data.cue_path)
    duration_s = max(_audio_duration_s(audio_path), len(frames) / max(fps, 1.0))
    aligned_frames = _fit_frames_to_duration(frames, fps, duration_s)
    envelope, audio_frame_stats = _audio_envelope_by_frame(audio_path, fps, len(aligned_frames), cues)
    character_names = data.character_names or sorted({cue.character for cue in cues})
    smoothing_state: dict = {}

    with _create_face_mesh() as face_mesh:
        synced_frames = [
            _mediapipe_lip_sync_frame(
                frame,
                face_mesh,
                index / max(fps, 1.0),
                cues,
                envelope[index],
                character_names,
                context,
                smoothing_state,
            )
            for index, frame in enumerate(aligned_frames)
        ]

    temp_video = temp_root / f"{output_path.stem}_mediapipe_temp.mp4"
    frame_dir = _frame_dir_for(output_path)
    _write_video_from_frames(temp_video, synced_frames, fps=max(1, int(round(fps))), frame_dir=frame_dir)
    _mux_audio(temp_video, audio_path, output_path)
    if temp_video.exists():
        temp_video.unlink()

    alignment_path = output_path.with_suffix(".alignment.json")
    fps, frame_size, video_duration_s = _video_info(output_path)
    frame_count = int(round(video_duration_s * max(fps, 1.0)))
    _write_json(
        alignment_path,
        {
            "scene_number": data.scene_number,
            "source_video_path": str(source_video),
            "audio_path": str(audio_path),
            "output_path": str(output_path),
            "lip_sync_backend": "mediapipe-face-mesh",
            "mediapipe": {
                "landmarks": 468,
                "mouth_landmarks": {
                    "outer": OUTER_LIP_LANDMARKS,
                    "inner": INNER_LIP_LANDMARKS,
                    "upper_lip_measure": UPPER_LIP_MEASURE_POINTS,
                    "lower_lip_measure": LOWER_LIP_MEASURE_POINTS,
                    "mouth_corners": [MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT],
                },
                "motion": "baseline-normalized-rms-frame-energy-lip-landmark-gap",
                "smoothing": "0.6-prev-0.4-current-frame-energy-plus-relative-mouth-smoothing",
                "baseline": "first-detected-mouth-gap-per-character",
            },
            "fps": fps,
            "frame_size": {"width": frame_size[0], "height": frame_size[1]},
            "frame_count": frame_count,
            "duration_s": _round_time(video_duration_s),
            "audio_frame_alignment": audio_frame_stats,
            "characters": data.character_names,
            "cues": [cue.model_dump() for cue in _load_cues(data.cues, data.cue_path)],
        },
    )

    return LipSyncAlignerOutput(
        video_path=str(output_path),
        audio_path=str(audio_path),
        frame_dir=str(frame_dir),
        alignment_path=str(alignment_path),
    )
