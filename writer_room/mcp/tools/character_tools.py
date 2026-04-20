from __future__ import annotations

from collections import defaultdict
import re

from pydantic import BaseModel

from writer_room.mcp.base import mcp_tool
from writer_room.schemas.screenplay import CharacterProfile, SceneManifest


class ExtractCharacterProfilesInput(BaseModel):
    scene_manifest: SceneManifest


class ExtractCharacterProfilesOutput(BaseModel):
    characters: list[CharacterProfile]


class QueryStockFootageInput(BaseModel):
    character_name: str
    appearance_description: str


class QueryStockFootageOutput(BaseModel):
    reference_style: str


def _trait_seed(name: str) -> list[str]:
    trait_bank = {
        "m": ["observant", "resilient", "inventive"],
        "r": ["protective", "pragmatic", "loyal"],
        "a": ["curious", "driven", "empathetic"],
    }
    return trait_bank.get(name[:1].lower(), ["focused", "resourceful", "layered"])


def _infer_age_descriptor(name: str) -> str:
    if name.lower() in {"mira", "rowan", "zara", "leena", "idris"}:
        return "young adult"
    return "adult"


def _infer_gender_presentation(name: str) -> str:
    mapping = {
        "mira": "woman",
        "rowan": "man",
        "zara": "woman",
        "leena": "woman",
        "idris": "man",
    }
    return mapping.get(name.lower(), "person")


def _scene_text_for_character(manifest: SceneManifest, scene_ids: list[int]) -> str:
    relevant = [scene for scene in manifest.scenes if scene.scene_number in scene_ids]
    chunks: list[str] = []
    for scene in relevant:
        chunks.append(scene.heading)
        chunks.extend(scene.action)
        chunks.extend(scene.visual_cues)
    return " ".join(chunks)


def _infer_environment_hint(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ["floating", "lantern", "moonwork", "clocktower", "sky garden", "enchanted", "glass forest"]):
        return "whimsical floating city of lantern bridges, moonlit towers, and magical clockwork gardens"
    if any(word in lowered for word in ["orbital", "reactor", "station", "planet", "transit ring", "holographic", "plasma"]):
        return "futuristic orbital station with glowing reactor light, metallic structures, and vast cosmic backdrops"
    if "harbor" in lowered or "coastal" in lowered:
        return "industrial coastal harbor with wet steel docks and storm-heavy skies"
    if "machine shop" in lowered or "workbench" in lowered:
        return "rain-soaked machine shop filled with steel tools, cables, and tungsten light"
    if "tower" in lowered or "platform" in lowered:
        return "wind-lashed maintenance platform above crashing waves and warning beacons"
    return "grounded cinematic environment tied to the story world"


def _infer_wardrobe(name: str, text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ["floating", "lantern", "moonwork", "clocktower", "sky garden", "enchanted", "glass forest"]):
        return "tailored whimsical adventurer clothing with elegant layers, magical utility details, and storybook textures"
    if any(word in lowered for word in ["orbital", "reactor", "station", "planet", "transit ring", "plasma"]):
        return "sleek futuristic utility wear with structured layers and subtle technical detailing"
    if name.lower() == "mira":
        return "weatherproof dark technical coat over utilitarian layers with storm-ready details"
    if name.lower() == "rowan":
        return "rugged dark utility coat with layered knitwear suited to a harsh waterfront"
    if name.lower() == "zara":
        return "dark fitted utility jacket with technical layers and practical storm-weather clothing"
    if name.lower() == "leena":
        return "structured field coat over modern industrial workwear with clean functional lines"
    if name.lower() == "idris":
        return "dark work coat over layered industrial knitwear and practical dockside clothing"
    if "harbor" in lowered or "coastal" in lowered:
        return "weatherproof dark coat over utilitarian layers with wind-worn textures"
    if "machine shop" in lowered:
        return "practical work jacket, rolled sleeves, and industrial utility clothing"
    return "grounded contemporary wardrobe shaped by the scene context"


def _infer_body_language(traits: list[str]) -> str:
    lowered = {trait.lower() for trait in traits}
    if "protective" in lowered or "pragmatic" in lowered:
        return "steady stance, guarded shoulders, and controlled protective energy"
    if "observant" in lowered or "inventive" in lowered:
        return "intent gaze, thoughtful posture, and contained restless energy"
    return "expressive body language with subtle tension and emotional realism"


def _appearance_description(name: str, environment_hint: str, wardrobe: str, body_language: str) -> str:
    return (
        f"{name} has a cinematic live-action look, distinct facial detail, {wardrobe}, "
        f"set against {environment_hint}, with {body_language}."
    )


@mcp_tool(
    name="extract_character_profiles",
    description="Derive persistent character identities from scene manifests.",
    tags=["character", "extraction"],
    input_model=ExtractCharacterProfilesInput,
    output_model=ExtractCharacterProfilesOutput,
)
def extract_character_profiles_tool(data: ExtractCharacterProfilesInput, _: dict) -> ExtractCharacterProfilesOutput:
    appearances: dict[str, str] = {}
    source_scenes: dict[str, list[int]] = defaultdict(list)

    for scene in data.scene_manifest.scenes:
        for line in scene.dialogue:
            source_scenes[line.character].append(scene.scene_number)

    characters = []
    for name in sorted(source_scenes):
        traits = _trait_seed(name)
        scene_ids = sorted(set(source_scenes[name]))
        scene_text = _scene_text_for_character(data.scene_manifest, scene_ids)
        environment_hint = _infer_environment_hint(scene_text)
        wardrobe = _infer_wardrobe(name, scene_text)
        body_language = _infer_body_language(traits)
        appearances[name] = _appearance_description(name, environment_hint, wardrobe, body_language)
        characters.append(
            CharacterProfile(
                name=name,
                personality_traits=traits,
                age_descriptor=_infer_age_descriptor(name),
                gender_presentation=_infer_gender_presentation(name),
                appearance_description=appearances[name],
                wardrobe=wardrobe,
                environment_hint=environment_hint,
                body_language=body_language,
                reference_style="cinematic portrait, mid-shot, realistic lighting, film still detail",
                source_scenes=scene_ids,
            )
        )

    return ExtractCharacterProfilesOutput(characters=characters)


@mcp_tool(
    name="query_stock_footage",
    description="Return a visual reference style for consistent character treatment.",
    tags=["character", "reference"],
    input_model=QueryStockFootageInput,
    output_model=QueryStockFootageOutput,
)
def query_stock_footage_tool(data: QueryStockFootageInput, _: dict) -> QueryStockFootageOutput:
    environment_match = re.search(r"set against (.*?), with", data.appearance_description)
    environment_hint = environment_match.group(1) if environment_match else "story-relevant environment"
    style = (
        f"{data.character_name}: cinematic portrait, close-up to mid-shot framing, {environment_hint}, "
        f"soft key light, rim light, realistic skin detail, dramatic shadows, expressive body language"
    )
    return QueryStockFootageOutput(reference_style=style)
