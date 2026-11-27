"""Prepare reference-conditioned pet frames/strips and validate their fixed atlas layout.

These checks establish image structure, not character identity or pose quality.
Template-strip generation (the default) produces one canonical still followed by
11 horizontal animation/look strips conditioned on checked-in pose templates and
that still, then slices and packs them into the v2 1536x2288 WebP atlas.
"""

from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, model_validator

CELL = (192, 208)
SOURCE_CELL = (384, 416)
STILL_SIZE = (768, 832)
STRIP_SIZE = (1536, 512)
BACKGROUND = (255, 0, 255)
COLUMNS = 8
GROUNDED_ROWS = frozenset(
    {"idle", "waving", "failed", "waiting", "running", "review", "look-0", "look-1"}
)
TEMPLATES_DIR = Path(__file__).resolve().parent / "pet_templates"

LOOK_DIRECTIONS = (
    "straight up toward the top edge of the image, chin raised and pupils upward",
    "mostly up and slightly right, chin raised and pupils toward the upper right",
    "diagonally up and right toward the top-right corner of the image",
    "mostly right and slightly up, toward the upper part of the right image edge",
    "straight right toward the right edge of the image, nose and pupils turned right",
    "mostly right and slightly down, toward the lower part of the right image edge",
    "diagonally down and right toward the bottom-right corner of the image",
    "mostly down and slightly right, chin lowered and pupils toward the lower right",
    "straight down toward the bottom edge of the image, chin lowered and pupils downward",
    "mostly down and slightly left, chin lowered and pupils toward the lower left",
    "diagonally down and left toward the bottom-left corner of the image",
    "mostly left and slightly down, toward the lower part of the left image edge",
    "straight left toward the left edge of the image, nose and pupils turned left",
    "mostly left and slightly up, toward the upper part of the left image edge",
    "diagonally up and left toward the top-left corner of the image",
    "mostly up and slightly left, chin raised and pupils toward the upper left",
)


class PetRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    index: int
    durations: tuple[int, ...]
    poses: tuple[str, ...]
    angles: tuple[float, ...] = ()

    @model_validator(mode="after")
    def validate_frames(self) -> "PetRow":
        if not 1 <= len(self.poses) <= COLUMNS:
            raise ValueError("A pet row needs between one and eight poses")
        if len(self.durations) != len(self.poses) or any(
            d <= 0 for d in self.durations
        ):
            raise ValueError("Each pose needs a positive duration")
        if self.angles and len(self.angles) != len(self.poses):
            raise ValueError("Each look pose needs an angle")
        return self

    @property
    def frame_count(self) -> int:
        return len(self.poses)


ROWS = (
    PetRow(
        name="idle",
        index=0,
        durations=(280, 110, 110, 140, 140, 320),
        poses=(
            "relaxed neutral stance, eyes open",
            "eyelids halfway closed",
            "eyes closed in a blink",
            "eyes opening",
            "gentle breath in, chest lifted",
            "relaxed neutral stance",
        ),
    ),
    PetRow(
        name="running-right",
        index=1,
        durations=(120,) * 7 + (220,),
        poses=(
            "facing right, right foot forward contact",
            "right leg absorbing weight",
            "left leg passing right",
            "left leg reaching forward",
            "left foot forward contact",
            "left leg absorbing weight",
            "right leg passing left",
            "right leg reaching forward",
        ),
    ),
    PetRow(
        name="running-left",
        index=2,
        durations=(120,) * 7 + (220,),
        poses=(
            "facing left, left foot forward contact",
            "left leg absorbing weight",
            "right leg passing left",
            "right leg reaching forward",
            "right foot forward contact",
            "right leg absorbing weight",
            "left leg passing right",
            "left leg reaching forward",
        ),
    ),
    PetRow(
        name="waving",
        index=3,
        durations=(140, 140, 140, 280),
        poses=(
            "arm raised to greet",
            "hand waving outward",
            "hand waving inward",
            "arm relaxing after greeting",
        ),
    ),
    PetRow(
        name="jumping",
        index=4,
        durations=(140, 140, 140, 140, 280),
        poses=(
            "crouched ready to jump",
            "pushing off the floor",
            "airborne at jump apex, feet above baseline",
            "descending towards floor",
            "soft landing with knees bent",
        ),
    ),
    PetRow(
        name="failed",
        index=5,
        durations=(140,) * 7 + (240,),
        poses=(
            "noticing a mistake",
            "startled wide eyes",
            "shoulders sagging",
            "head tilted down",
            "small disappointed sigh",
            "looking back up",
            "recovering composure",
            "determined neutral stance",
        ),
    ),
    PetRow(
        name="waiting",
        index=6,
        durations=(150,) * 5 + (260,),
        poses=(
            "patient attentive stance",
            "head tilting slightly right",
            "curious upward glance",
            "gentle blink",
            "head returning to center",
            "patient attentive stance",
        ),
    ),
    PetRow(
        name="running",
        index=7,
        durations=(120,) * 5 + (220,),
        poses=(
            "focused working stance",
            "leaning in, hands preparing a task",
            "hands actively working",
            "hands moving to next step",
            "checking the work",
            "focused working stance",
        ),
    ),
    PetRow(
        name="review",
        index=8,
        durations=(150,) * 5 + (280,),
        poses=(
            "thoughtful inspecting stance",
            "head leaning closer",
            "eyes scanning left",
            "eyes scanning right",
            "small thoughtful nod",
            "thoughtful inspecting stance",
        ),
    ),
    *(
        PetRow(
            name=f"look-{half}",
            index=9 + half,
            durations=(150,) * 8,
            angles=tuple((half * 8 + step) * 22.5 for step in range(8)),
            poses=tuple(
                f"body facing front, turn the head and direct both eyes {direction}. "
                "The gaze must visibly follow that screen direction rather than look at the viewer"
                for direction in LOOK_DIRECTIONS[half * 8 : (half + 1) * 8]
            ),
        )
        for half in range(2)
    ),
)

ROWS_BY_NAME = {row.name: row for row in ROWS}
VALID_TEMPLATE_NAMES = frozenset({"still", *ROWS_BY_NAME})

STRIP_ACTIONS = {
    "idle": "6-frame idle blink cycle",
    "running-right": "8-frame running right side-profile cycle",
    "running-left": "8-frame running left side-profile cycle",
    "waving": "4-frame friendly waving cycle",
    "jumping": "5-frame crouch, airborne jump, and landing cycle",
    "failed": "8-frame startled error and recovery cycle",
    "waiting": "6-frame clasped-hands waiting cycle",
    "running": "6-frame working cycle holding and tending a small terracotta potted green sprout",
    "review": "6-frame hand-on-chin thoughtful reviewing cycle",
    "look-0": "8-frame head and eye look-around strip clockwise from straight up through right to down-right",
    "look-1": "8-frame head and eye look-around strip clockwise from straight down through left to up-left",
}


@lru_cache(maxsize=16)
def template_bytes(name: str) -> bytes:
    if name not in VALID_TEMPLATE_NAMES:
        raise ValueError(f"Unknown pet template: {name}")
    path = TEMPLATES_DIR / f"{name}.png"
    return path.read_bytes()


def strip_base_prompt(description: str) -> str:
    return (
        f"Redraw the character from the reference pose as: {description.strip()}. "
        "Keep the exact front-facing full-body standing pose, camera framing, centered placement, "
        "crisp dark outline, and clean margins of the reference image. "
        "Use a completely flat solid pure magenta (#ff00ff) background with no shadows, ground, "
        "text, borders, or extra props. Do not use magenta anywhere on the character."
    )


def strip_prompt(description: str, row: PetRow) -> str:
    action = STRIP_ACTIONS[row.name]
    extra_motif = (
        " Keep the small terracotta potted green sprout prop from the first reference in each frame."
        if row.name == "running"
        else " Do not keep any straw hat, ponytail, rose, or green overalls from the first reference."
    )
    return (
        f"Exact sprite-sheet pose transfer ({action}, {row.frame_count} figures evenly spaced in one horizontal row). "
        f"Copy the exact {row.frame_count} poses, frame count, horizontal spacing, scale, head turns, and eye directions "
        "from the first reference strip, but replace the character completely with the character from the second reference "
        f"image ({description.strip()}). Preserve every costume, color, hair, accessory, and art-style detail from the "
        f"second reference character.{extra_motif} "
        "Completely flat solid pure magenta (#ff00ff) background, no text, no grid lines, no shadows, no magenta on the character."
    )


def base_prompt(description: str) -> str:
    return (
        f"Create one full-body character: {description.strip()}. Preserve the requested art style; "
        "any style is allowed. Front-facing relaxed neutral stance, one character only, entire body "
        "and all extremities visible. Leave generous empty space on every side and above for jumping. "
        "Use a completely flat pure magenta (#ff00ff) background, with no ground, shadows, text, border, "
        "or extra objects. Do not use magenta or near-magenta anywhere on the character."
    )


def character_sheet_prompt(description: str) -> str:
    return (
        f"Create a character reference sheet for ONE character design: {description.strip()}. "
        "Show the SAME character in front, side and back full-body views plus a few expressive "
        "poses. All views must share identity, proportions, colors, markings, costume and art "
        "style; do not introduce different characters. Separate the views with generous empty "
        "space on a flat pure magenta (#ff00ff) background. Keep every view fully inside the "
        "image. No text, labels, borders, grid lines, shadows or scenery. No magenta on the character."
    )


def sheet_pose_prompt() -> str:
    return (
        "The reference is a character sheet showing several views of the SAME character. "
        "Use those views only to understand the design. Draw exactly ONE character in ONE pose; "
        "do not reproduce the sheet, its layout, panels, grid, labels or multiple views. "
        "Use a fixed camera with the neutral character about 260 pixels wide or less and "
        "292 pixels tall or less on this 384x416 canvas, centered at x=192 with feet at y=368. "
    )


def sheet_base_prompt(description: str) -> str:
    return sheet_pose_prompt() + base_prompt(description)


def frame_name(row: PetRow, index: int) -> str:
    if not 0 <= index < row.frame_count:
        raise ValueError(f"Invalid frame index for {row.name}: {index}")
    return f"{row.name}/{index}"


def frame_prompt(
    description: str, row: PetRow, index: int, *, character_sheet: bool = False
) -> str:
    frame_name(row, index)
    direction = ""
    if row.name in {"running-right", "running-left"}:
        direction = f"Entire body facing {row.name.removeprefix('running-')}, in a running stride. "
    standing_level = (
        "standing floor y=368"
        if character_sheet
        else "same standing level as the reference"
    )
    grounding = (
        f"Keep both legs straight and both feet flat at the {standing_level}. "
        "Do not bend, cross or lift the legs, stand on tiptoe, crouch or float; only the requested "
        "upper-body, arm, head and eye movement changes. "
        if row.name not in {"running-right", "running-left", "jumping"}
        else ""
    )
    camera = (
        "Keep the same body scale and fixed camera across animation poses. "
        if character_sheet
        else "Preserve the reference's body scale, fixed camera and horizontal center at x=192. "
    )
    prompt = (
        f"Use the reference to draw exactly ONE full-body character ({description.strip()}) "
        f"in exactly ONE animation pose: {direction}{row.poses[index]}. {grounding}"
        "This is a single 384x416 image, not a sprite sheet or sequence. Keep the reference "
        "character's identity, body proportions, colors, markings, costume and art style. "
        "Generate a genuine articulated pose, with the required limb, face and head changes; "
        f"do not merely translate, rotate or scale the reference picture. {camera}The standing floor is y=368; "
        "airborne jump poses must rise above that floor. Preserve body proportions when crouching "
        "or extending limbs; never enlarge the character to fill the image. Keep all extremities "
        "inside the image with generous empty margins on every side and above. "
        "Use only a completely flat pure magenta (#ff00ff) background extending to all four edges. "
        "No border, frame, vignette, gradient, shadow, ground, text, labels or extra objects. "
        "No magenta on the character. Up is screen top (0 degrees), right is 90 degrees, "
        "down is 180 degrees, left is 270 degrees."
    )
    return sheet_pose_prompt() + prompt if character_sheet else prompt


def _png(data: bytes, expected: tuple[int, int] | None = None) -> Image.Image:
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG":
                raise ValueError("Expected a PNG image")
            if expected is not None and image.size != expected:
                raise ValueError(
                    f"Expected image dimensions {expected}, got {image.size}"
                )
            if image.width * image.height > 4096 * 4096:
                raise ValueError("Pet source image exceeds 16 megapixels")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Expected a single PNG frame")
            return image.convert("RGBA")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Could not decode pet PNG") from exc


def _despill_edges(image: Image.Image) -> Image.Image:
    """Suppress magenta chroma fringe on 1-2px edges adjacent to transparency."""
    alpha = image.getchannel("A")
    transparent = alpha.point(lambda a: 255 if a == 0 else 0)
    near_transparent = transparent.filter(ImageFilter.MaxFilter(5))
    edge_mask = ImageChops.multiply(
        near_transparent, alpha.point(lambda a: 255 if a > 0 else 0)
    )
    bbox = edge_mask.getbbox()
    if bbox is None:
        return image
    result = image.copy()
    pixels = result.load()
    mask_px = edge_mask.load()
    left, top, right, bottom = bbox
    for y in range(top, bottom):
        for x in range(left, right):
            if not mask_px[x, y]:
                continue
            r, g, b, a = pixels[x, y]
            spill = min(r - g, b - g)
            if spill <= 18:
                continue
            # Clamp magenta excess back toward the green channel and soften halo.
            nr = max(0, r - spill)
            nb = max(0, b - spill)
            na = 0 if spill > 95 and a < 220 else (max(0, a - spill) if spill > 55 else a)
            if na == 0:
                pixels[x, y] = (0, 0, 0, 0)
            else:
                pixels[x, y] = (nr, g, nb, na)
    return result


def _cutout(image: Image.Image, *, strip_cleanup: bool = False) -> Image.Image:
    red, green, blue, alpha = image.split()
    # Flood edge-connected chroma background.
    pink = ImageChops.darker(
        ImageChops.subtract(red, green), ImageChops.subtract(blue, green)
    )
    pink_edge = pink.point(lambda value: 255 if value > 60 else 0)
    bright = ImageChops.darker(red, blue).point(lambda value: 255 if value > 130 else 0)
    candidate = ImageChops.lighter(
        ImageChops.multiply(pink_edge, bright),
        alpha.point(lambda a: 255 if a == 0 else 0),
    )
    connected = Image.new("L", (image.width + 2, image.height + 2), 255)
    connected.paste(candidate, (1, 1))
    ImageDraw.floodfill(connected, (0, 0), 128)
    keep = connected.crop((1, 1, image.width + 1, image.height + 1)).point(
        lambda p: 0 if p == 128 else 255
    )
    if strip_cleanup:
        # Remove enclosed pure-magenta pockets (e.g., triangles between boots/arms).
        pure_pink = pink.point(lambda value: 255 if value > 120 else 0)
        pure_bright = ImageChops.darker(red, blue).point(
            lambda value: 255 if value > 185 else 0
        )
        low_green = green.point(lambda value: 255 if value < 80 else 0)
        interior_chroma = ImageChops.multiply(
            ImageChops.multiply(pure_pink, pure_bright), low_green
        )
        keep = ImageChops.multiply(
            keep, interior_chroma.point(lambda v: 0 if v else 255)
        )
    result = image.copy()
    result.putalpha(ImageChops.multiply(alpha, keep))
    if strip_cleanup:
        result = _despill_edges(result)
    result.paste(
        (0, 0, 0, 0), mask=result.getchannel("A").point(lambda a: 255 if a == 0 else 0)
    )
    return result


def _bounds(image: Image.Image) -> tuple[int, int, int, int]:
    visible = image.getchannel("A").point(lambda a: 255 if a > 16 else 0)
    bounds = visible.getbbox()
    if bounds is None or visible.histogram()[255] < 64:
        raise ValueError("Pet frame is blank or contains too little visible artwork")
    left, top, right, bottom = bounds
    if left < 2 or top < 2 or right > image.width - 2 or bottom > image.height - 2:
        raise ValueError(
            "Pet artwork touches a cell boundary; regenerate with clear margins"
        )
    return bounds


def _encode(image: Image.Image, format: str = "PNG") -> bytes:
    output = BytesIO()
    kwargs = {"lossless": True, "exact": True} if format == "WEBP" else {}
    image.save(output, format=format, **kwargs)
    return output.getvalue()


def compose_on_magenta(png: bytes) -> bytes:
    """Composite a transparent cutout PNG onto flat #ff00ff for model reference input."""
    image = _png(png)
    backdrop = Image.new("RGBA", image.size, (*BACKGROUND, 255))
    backdrop.alpha_composite(image)
    return _encode(backdrop.convert("RGB"), "PNG")


def prepare_still(png: bytes) -> bytes:
    """Validate and cut out the 768x832 canonical still for template-strip pets."""
    image = _cutout(_png(png, STILL_SIZE), strip_cleanup=True)
    _bounds(image)
    return _encode(image)


def _find_strip_cuts(alpha: Image.Image, frame_count: int) -> list[int]:
    width, height = alpha.size
    px = alpha.load()
    col_ink = [
        sum(1 for y in range(height) if px[x, y] > 16) for x in range(width)
    ]
    cuts = [0]
    slot = width / frame_count
    radius = max(4, int(slot * 0.28))
    for i in range(1, frame_count):
        center = int(round(i * slot))
        lo = max(cuts[-1] + 8, center - radius)
        hi = min(width - 8 * (frame_count - i), center + radius)
        best_x = center
        best_val = col_ink[center]
        for x in range(lo, hi + 1):
            val = col_ink[x]
            if val < best_val or (
                val == best_val and abs(x - center) < abs(best_x - center)
            ):
                best_val = val
                best_x = x
        cuts.append(best_x)
    cuts.append(width)
    return cuts


def slice_strip(strip_image: Image.Image, row: PetRow) -> list[Image.Image]:
    """Slice a cutout 1536x512 strip into row.frame_count packed 192x208 cells."""
    if strip_image.size != STRIP_SIZE:
        raise ValueError(
            f"Expected strip dimensions {STRIP_SIZE}, got {strip_image.size}"
        )
    alpha = strip_image.getchannel("A")
    cuts = _find_strip_cuts(alpha, row.frame_count)
    segments: list[tuple[Image.Image, tuple[int, int, int, int]]] = []
    for index in range(row.frame_count):
        seg = strip_image.crop((cuts[index], 0, cuts[index + 1], strip_image.height))
        visible = seg.getchannel("A").point(lambda a: 255 if a > 16 else 0)
        bbox = visible.getbbox()
        if bbox is None or visible.histogram()[255] < 64:
            raise ValueError(
                f"Strip {row.name} frame {index} is blank or contains too little artwork"
            )
        segments.append((seg, bbox))

    max_w = max(b[2] - b[0] for _, b in segments)
    max_h = max(b[3] - b[1] for _, b in segments)
    if row.name in GROUNDED_ROWS:
        scale = min(176.0 / max_w, 192.0 / max_h)
    else:
        min_top = min(b[1] for _, b in segments)
        max_bottom = max(b[3] for _, b in segments)
        vertical_span = max(1, max_bottom - min_top)
        scale = min(176.0 / max_w, 194.0 / vertical_span)

    baseline_y = 202
    max_bottom = max(b[3] for _, b in segments)
    cells: list[Image.Image] = []
    for seg, (left, top, right, bottom) in segments:
        crop = seg.crop((left, top, right, bottom))
        rw = max(1, round(crop.width * scale))
        rh = max(1, round(crop.height * scale))
        scaled = crop.resize((rw, rh), Image.Resampling.LANCZOS)
        cell = Image.new("RGBA", CELL, (0, 0, 0, 0))
        px = (CELL[0] - rw) // 2
        if row.name in GROUNDED_ROWS:
            py = baseline_y - rh
        else:
            lift = round((max_bottom - bottom) * scale)
            py = baseline_y - rh - lift
        py = max(2, min(CELL[1] - rh - 2, py))
        px = max(2, min(CELL[0] - rw - 2, px))
        cell.alpha_composite(scaled, (px, py))
        cell.paste(
            (0, 0, 0, 0), mask=cell.getchannel("A").point(lambda a: 255 if a == 0 else 0)
        )
        _bounds(cell)
        cells.append(cell)

    if len({cell.tobytes() for cell in cells}) == 1:
        raise ValueError(
            f"Row {row.name} repeats one image; animated poses must differ"
        )
    return cells


def prepare_strip(png: bytes, row: PetRow) -> bytes:
    """Cut out and validate a 1536x512 generated strip before persisting."""
    strip = _cutout(_png(png, STRIP_SIZE), strip_cleanup=True)
    slice_strip(strip, row)
    return _encode(strip)


def assemble_atlas_from_strips(strips: dict[str, bytes]) -> bytes:
    """Slice all 11 prepared row strips and pack the v2 1536x2288 WebP atlas."""
    expected = set(ROWS_BY_NAME)
    if set(strips) != expected:
        raise ValueError("Atlas needs all 11 animation and look strips")
    atlas = Image.new("RGBA", (COLUMNS * CELL[0], len(ROWS) * CELL[1]), (0, 0, 0, 0))
    for row in ROWS:
        cells = slice_strip(_png(strips[row.name], STRIP_SIZE), row)
        for index, cell in enumerate(cells):
            atlas.paste(cell, (index * CELL[0], row.index * CELL[1]))
    atlas.paste(
        (0, 0, 0, 0), mask=atlas.getchannel("A").point(lambda a: 255 if a == 0 else 0)
    )
    return _encode(atlas, "WEBP")


def prepare_base(png: bytes) -> bytes:
    image = _cutout(_png(png))
    character = image.crop(_bounds(image))
    character.thumbnail((260, 292), Image.Resampling.LANCZOS)
    canonical = Image.new("RGBA", SOURCE_CELL)
    canonical.paste(
        character, ((SOURCE_CELL[0] - character.width) // 2, 368 - character.height)
    )
    return _encode(canonical)


def prepare_character_sheet(png: bytes) -> bytes:
    # Preserve the reference layout, never fit or pack its multiple silhouettes
    # into a single animation cell. This validates structure, not visual identity.
    return _encode(_png(png))


def prepare_frame(png: bytes, row: PetRow, index: int) -> bytes:
    frame_name(row, index)
    frame = _cutout(_png(png, SOURCE_CELL))
    _bounds(frame)
    # Validate the final display size before this source frame is persisted.
    _bounds(frame.resize(CELL, Image.Resampling.LANCZOS))
    # Keep a shared camera and canvas. Fitting each silhouette would change body
    # scale when limbs extend and erase the height of an airborne jump.
    return _encode(frame)


def _validated_row(row: PetRow, frames: list[bytes]) -> list[Image.Image]:
    if len(frames) != row.frame_count:
        raise ValueError(f"Row {row.name} needs exactly {row.frame_count} frames")
    cells = []
    for png in frames:
        frame = _png(png, SOURCE_CELL)
        _bounds(frame)
        cell = frame.resize(CELL, Image.Resampling.LANCZOS)
        _bounds(cell)
        cells.append(cell)
    if len({cell.tobytes() for cell in cells}) == 1:
        raise ValueError(
            f"Row {row.name} repeats one image; animated poses must differ"
        )
    return cells


def validate_row(row: PetRow, frames: list[bytes]) -> None:
    """Check a complete row before its final frame is persisted."""
    _validated_row(row, frames)


def assemble_atlas(frames: dict[str, bytes]) -> bytes:
    expected = {
        frame_name(row, index) for row in ROWS for index in range(row.frame_count)
    }
    if set(frames) != expected:
        raise ValueError(
            "Atlas needs exactly all 73 animation and directional look frames"
        )
    atlas = Image.new("RGBA", (COLUMNS * CELL[0], len(ROWS) * CELL[1]))
    for row in ROWS:
        row_frames = _validated_row(
            row,
            [frames[frame_name(row, index)] for index in range(row.frame_count)],
        )
        for index, frame in enumerate(row_frames):
            atlas.paste(frame, (index * CELL[0], row.index * CELL[1]))
    return _encode(atlas, "WEBP")
