from __future__ import annotations

import re

from pydantic import BaseModel, Field

from writer_room.mcp.base import mcp_tool
from writer_room.schemas.screenplay import DialogueLine, Scene, SceneManifest


SCENE_HEADER_PATTERN = re.compile(r"^(INT\.|EXT\.)\s+.+", re.IGNORECASE)


class GenerateScriptSegmentInput(BaseModel):
    prompt: str
    memory_context: list[str] = Field(default_factory=list)


class ValidateScriptInput(BaseModel):
    raw_script: str


class ValidateScriptOutput(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    scene_manifest: SceneManifest | None = None


def _infer_title(prompt: str) -> str:
    words = [word.strip(" ,.!?") for word in prompt.split() if word.strip()]
    return " ".join(words[:5]).title() or "Untitled Story"


def _extract_candidate_names(text: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    filtered: list[str] = []
    blocked = {"The", "Scene", "Night", "Day", "Interior", "Exterior"}
    for item in candidates:
        if item not in filtered and item not in blocked:
            filtered.append(item)
    return filtered[:4]


def _detect_story_theme(prompt: str) -> str:
    lowered = prompt.lower()
    if any(word in lowered for word in ["whimsical", "floating", "lantern", "moon", "magic", "glass forest", "clockmaker", "enchanted"]):
        return "whimsical_fantasy"
    if any(word in lowered for word in ["space", "starship", "android", "galaxy", "planet", "orbital", "sci-fi", "cyber"]):
        return "science_fiction"
    return "grounded_drama"


def _build_grounded_scenes(protagonist: str, partner: str, threat: str) -> list[Scene]:
    return [
        Scene(
            scene_number=1,
            heading="EXT. COASTAL HARBOR - SUNSET",
            summary=f"{protagonist} returns home and confronts the emotional weight of the harbor she left behind.",
            action=[
                f"{protagonist} steps off a weather-beaten ferry carrying a compact robotics case.",
                f"Fishing boats sway under a sky flashing with distant energy from the {threat}.",
            ],
            dialogue=[
                DialogueLine(
                    character=protagonist,
                    text="I told myself I came back for the town, not for the ghosts in it.",
                    emotion="guarded",
                ),
                DialogueLine(
                    character=partner,
                    text="Then stop staring at the tower and help me keep this harbor standing.",
                    emotion="challenging",
                ),
            ],
            visual_cues=[
                "Orange sunset reflecting on metallic water.",
                "Electrical arcs pulse around the distant tower.",
                "Wind catches loose harbor tarps and cables.",
            ],
        ),
        Scene(
            scene_number=2,
            heading="INT. MACHINE SHOP - NIGHT",
            summary=f"{protagonist} and {partner} rebuild trust while tracing the tower's design flaw.",
            action=[
                "Workbench lights cut sharp silhouettes across blueprints and dismantled drone parts.",
                f"{protagonist} overlays old schematics with current weather telemetry.",
                f"{partner} reveals he kept monitoring the {threat} after everyone else gave up.",
            ],
            dialogue=[
                DialogueLine(
                    character=partner,
                    text="You left when the funding vanished. I stayed when the danger got personal.",
                    emotion="resentful",
                ),
                DialogueLine(
                    character=protagonist,
                    text="I left because I knew the design would fail. I came back because now I know how to stop it.",
                    emotion="determined",
                ),
            ],
            visual_cues=[
                "Warm tungsten lamps against cold rain on the windows.",
                "Close-ups of grease-streaked hands moving across circuit boards.",
                "A wall map pinned with storm trajectories.",
            ],
        ),
        Scene(
            scene_number=3,
            heading="EXT. TOWER PLATFORM - PRE-DAWN",
            summary=f"{protagonist} and {partner} disable the tower before it fractures the harbor.",
            action=[
                "Waves crash beneath a narrow maintenance platform lashed by rain.",
                f"{protagonist} climbs into the control housing while {partner} reroutes emergency power.",
                "The tower surges once, then exhales into silence as dawn breaks over calmer water.",
            ],
            dialogue=[
                DialogueLine(
                    character=protagonist,
                    text="If this works, the town gets its morning back.",
                    emotion="focused",
                ),
                DialogueLine(
                    character=partner,
                    text="And if it doesn't, we fail together this time.",
                    emotion="resolved",
                ),
            ],
            visual_cues=[
                "Sheets of rain lit by warning beacons.",
                "A sudden stillness when the tower powers down.",
                "First light spreading across the reopened harbor.",
            ],
        ),
    ]


def _build_whimsical_scenes(protagonist: str, partner: str) -> list[Scene]:
    return [
        Scene(
            scene_number=1,
            heading="EXT. FLOATING LANTERN CITY - TWILIGHT",
            summary=f"{protagonist} crosses into a dreamlike city of lantern bridges and drifting towers to begin the journey.",
            action=[
                f"{protagonist} steps onto a swaying bridge woven from brass chains and glowing paper lanterns.",
                f"{partner} follows with a satchel of delicate tools while moonlit airships drift above the rooftops.",
            ],
            dialogue=[
                DialogueLine(
                    character=protagonist,
                    text="If the moon machine wakes before dawn, the whole sky will forget how to hold its light.",
                    emotion="awed",
                ),
                DialogueLine(
                    character=partner,
                    text="Then we move fast and gently. Magical things break harder than machines do.",
                    emotion="careful",
                ),
            ],
            visual_cues=[
                "Floating lanterns cast warm gold across clouds below the streets.",
                "Clockwork towers lean against a violet twilight sky.",
                "Tiny sparks drift from silver tram lines into the air like fireflies.",
            ],
        ),
        Scene(
            scene_number=2,
            heading="INT. MOONWORK CLOCKTOWER - NIGHT",
            summary=f"{protagonist} and {partner} discover the hidden heart of the moon machine inside a whimsical mechanical tower.",
            action=[
                "Glass gears revolve behind transparent walls filled with suspended stardust.",
                f"{protagonist} aligns a trembling brass key with a crescent-shaped control spindle.",
                f"{partner} reads shifting constellations reflected across the polished floor.",
            ],
            dialogue=[
                DialogueLine(
                    character=partner,
                    text="Someone taught this machine to dream in storms.",
                    emotion="uneasy",
                ),
                DialogueLine(
                    character=protagonist,
                    text="Then we teach it something kinder before sunrise.",
                    emotion="determined",
                ),
            ],
            visual_cues=[
                "Moonbeams fracture through crystal pendulums.",
                "Soft blue light ripples across brass machinery and painted clouds on the ceiling.",
                "A halo of glowing dust circles the central gear chamber.",
            ],
        ),
        Scene(
            scene_number=3,
            heading="EXT. SKY GARDEN ABOVE THE CITY - DAWN",
            summary=f"{protagonist} and {partner} calm the moon machine as dawn spreads across a whimsical skyline.",
            action=[
                "Petal-like metal leaves unfold around a floating garden suspended over the city.",
                f"{protagonist} places the final key into a luminous cradle while {partner} steadies the singing mechanism.",
                "The city exhales as lanterns dim into the first pearl-colored light of morning.",
            ],
            dialogue=[
                DialogueLine(
                    character=protagonist,
                    text="There. Let the morning arrive on its own terms.",
                    emotion="hopeful",
                ),
                DialogueLine(
                    character=partner,
                    text="And let this city keep a little wonder for itself.",
                    emotion="relieved",
                ),
            ],
            visual_cues=[
                "Rose-gold dawn slips across suspended gardens and floating balconies.",
                "Lanterns fade one by one into pale morning mist.",
                "The city below gleams like a music box waking up.",
            ],
        ),
    ]


def _build_scifi_scenes(protagonist: str, partner: str, threat: str) -> list[Scene]:
    return [
        Scene(
            scene_number=1,
            heading="EXT. ORBITAL TRANSIT RING - ECLIPSE",
            summary=f"{protagonist} arrives at a failing orbital city where the {threat} threatens the ring's stability.",
            action=[
                f"{protagonist} steps from a docking shuttle into a corridor lined with pulsing navigation glass.",
                f"{partner} studies flickering station telemetry as a fractured planet fills the viewport.",
            ],
            dialogue=[
                DialogueLine(character=protagonist, text="The ring is already drifting out of sync.", emotion="alert"),
                DialogueLine(character=partner, text="Then we repair the heart of it before the next eclipse surge.", emotion="focused"),
            ],
            visual_cues=[
                "Neon guidance lines spill across polished metal floors.",
                "A distant planet glows beneath the station's skeletal frame.",
                "Emergency beacons pulse through drifting vapor.",
            ],
        ),
        Scene(
            scene_number=2,
            heading="INT. REACTOR SPINE - NIGHT CYCLE",
            summary=f"{protagonist} and {partner} trace the flaw to the core systems of the {threat}.",
            action=[
                "Magnetic rails hum beside a reactor chamber filled with blue plasma light.",
                f"{protagonist} threads a diagnostic wand through a nest of fiber conduits.",
                f"{partner} isolates a corrupted feedback loop inside the station lattice.",
            ],
            dialogue=[
                DialogueLine(character=partner, text="This wasn't an accident. Someone tuned the system to overload.", emotion="grim"),
                DialogueLine(character=protagonist, text="Then we undo it before the whole ring tears open.", emotion="determined"),
            ],
            visual_cues=[
                "Cold blue reactor light cuts across steel and glass.",
                "Holographic readouts float over an exposed core chamber.",
                "Tiny arcs leap between damaged conduits.",
            ],
        ),
        Scene(
            scene_number=3,
            heading="EXT. EXTERNAL REPAIR ARRAY - DAWN GLOW",
            summary=f"{protagonist} and {partner} restore balance to the station as the system stabilizes.",
            action=[
                "They brace against the outer hull with the stars turning slowly behind them.",
                f"{protagonist} locks a replacement regulator into the exposed array while {partner} reroutes reserve power.",
                "The orbital ring steadies and city lights bloom back to life below the transparent dome.",
            ],
            dialogue=[
                DialogueLine(character=protagonist, text="There. The city gets another sunrise.", emotion="relieved"),
                DialogueLine(character=partner, text="And this time it earns it.", emotion="steady"),
            ],
            visual_cues=[
                "Soft dawn glow spills over the station hull.",
                "Thousands of city lights reappear across the ring habitat.",
                "A stabilizing pulse ripples through the structure like a heartbeat.",
            ],
        ),
    ]


def _build_generated_manifest(prompt: str, memory_context: list[str]) -> SceneManifest:
    names = _extract_candidate_names(prompt)
    protagonist = names[0] if names else "Mira"
    partner = names[1] if len(names) > 1 else "Rowan"
    threat = "storm-harvesting tower" if "storm" in prompt.lower() else "unstable prototype"
    theme = _detect_story_theme(prompt)
    if theme == "whimsical_fantasy":
        scenes = _build_whimsical_scenes(protagonist, partner)
    elif theme == "science_fiction":
        scenes = _build_scifi_scenes(protagonist, partner, threat)
    else:
        scenes = _build_grounded_scenes(protagonist, partner, threat)

    return SceneManifest(title=_infer_title(prompt), premise=prompt, mode="prompt", scenes=scenes)


def _parse_manual_script(raw_script: str) -> tuple[list[Scene], list[str]]:
    lines = [line.rstrip() for line in raw_script.splitlines()]
    scenes: list[Scene] = []
    errors: list[str] = []
    current_scene: Scene | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if SCENE_HEADER_PATTERN.match(line):
            if current_scene:
                scenes.append(current_scene)
            current_scene = Scene(
                scene_number=len(scenes) + 1,
                heading=line,
                summary="",
                action=[],
                dialogue=[],
                visual_cues=[],
            )
            continue

        if current_scene is None:
            errors.append("Content found before first scene heading.")
            continue

        if line.startswith("ACTION:"):
            current_scene.action.append(line.replace("ACTION:", "", 1).strip())
        elif line.startswith("VISUAL:"):
            current_scene.visual_cues.append(line.replace("VISUAL:", "", 1).strip())
        elif ":" in line:
            character, text = line.split(":", 1)
            character = character.strip()
            text = text.strip()
            if not character.isupper():
                errors.append(f"Dialogue speaker must be uppercase: {line}")
            else:
                current_scene.dialogue.append(DialogueLine(character=character.title(), text=text, emotion="neutral"))
        else:
            current_scene.action.append(line)

    if current_scene:
        scenes.append(current_scene)

    for scene in scenes:
        if not scene.dialogue:
            errors.append(f"Scene '{scene.heading}' is missing dialogue.")
        if not scene.action:
            errors.append(f"Scene '{scene.heading}' is missing action descriptions.")
        if not scene.summary:
            scene.summary = scene.action[0] if scene.action else "No action available."

    if not scenes:
        errors.append("No valid scenes found.")

    return scenes, errors


@mcp_tool(
    name="generate_script_segment",
    description="Expand a prompt into a structured multi-scene screenplay manifest.",
    tags=["scriptwriter", "generation"],
    input_model=GenerateScriptSegmentInput,
    output_model=SceneManifest,
)
def generate_script_segment_tool(data: GenerateScriptSegmentInput, _: dict) -> SceneManifest:
    return _build_generated_manifest(data.prompt, data.memory_context)


@mcp_tool(
    name="validate_script",
    description="Validate and standardize a manually supplied screenplay into scene JSON.",
    tags=["validator", "validation"],
    input_model=ValidateScriptInput,
    output_model=ValidateScriptOutput,
)
def validate_script_tool(data: ValidateScriptInput, _: dict) -> ValidateScriptOutput:
    scenes, errors = _parse_manual_script(data.raw_script)
    manifest = None
    if scenes:
        manifest = SceneManifest(
            title="Manual Script Submission",
            premise="User-supplied screenplay.",
            mode="manual",
            scenes=scenes,
        )
    return ValidateScriptOutput(valid=not errors, errors=errors, scene_manifest=manifest)
