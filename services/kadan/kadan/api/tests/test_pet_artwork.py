from io import BytesIO

import pytest
from PIL import Image, ImageDraw

from kadan.api.pet_artwork import (
    BACKGROUND,
    CELL,
    ROWS,
    SOURCE_CELL,
    assemble_atlas,
    extract_row,
    prepare_base,
)


def png(image):
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def character():
    image = Image.new("RGB", SOURCE_CELL, BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle((120, 100, 260, 360), fill="blue")
    # This enclosed mark shares the chroma color, but is part of the artwork.
    draw.rectangle((170, 140, 200, 170), fill=BACKGROUND)
    return image


def generated_strip(row, identical=False):
    strip = Image.new(
        "RGB", (SOURCE_CELL[0] * row.frame_count, SOURCE_CELL[1]), BACKGROUND
    )
    for column in range(row.frame_count):
        frame = character()
        draw = ImageDraw.Draw(frame)
        arm_y = 200 if identical else 180 + column * 10
        draw.rectangle((260, arm_y, 300, arm_y + 12), fill="blue")
        strip.paste(frame, (column * SOURCE_CELL[0], 0))
    return png(strip)


def test_canonical_reference_has_transparent_margins_and_shared_floor():
    base = Image.open(BytesIO(prepare_base(png(character()))))
    assert base.size == SOURCE_CELL
    assert base.getpixel((0, 0))[3] == 0
    assert base.getchannel("A").getbbox()[3] == 368
    assert base.getpixel((192, 164)) == (*BACKGROUND, 255)


def test_extraction_preserves_generated_cell_positions():
    row = ROWS[4]
    strip = Image.open(BytesIO(generated_strip(row)))
    # One airborne fixture is higher; extraction must not independently recenter it.
    airborne = Image.new("RGB", SOURCE_CELL, BACKGROUND)
    airborne.paste(character().crop((0, 50, 384, 416)), (0, 0))
    strip.paste(airborne, (384 * 2, 0))
    frames = extract_row(png(strip), row)
    assert all(frame.size == CELL for frame in frames)
    assert (
        frames[2].getchannel("A").getbbox()[3]
        < frames[0].getchannel("A").getbbox()[3] - 20
    )
    assert frames[0].getpixel((92, 75)) == (*BACKGROUND, 255)


@pytest.mark.parametrize("kind", ["blank", "cropped", "wrong-size", "repeated"])
def test_invalid_generated_rows_fail_before_use(kind):
    row = ROWS[3]
    strip = Image.open(BytesIO(generated_strip(row, identical=kind == "repeated")))
    if kind == "blank":
        strip.paste(Image.new("RGB", SOURCE_CELL, BACKGROUND), (0, 0))
    elif kind == "cropped":
        ImageDraw.Draw(strip).rectangle((0, 100, 150, 200), fill="blue")
    elif kind == "wrong-size":
        strip = strip.crop((0, 0, strip.width - 1, strip.height))
    with pytest.raises(ValueError):
        extract_row(png(strip), row)


def test_invalid_base_is_rejected_instead_of_hidden_by_resizing():
    with pytest.raises(ValueError, match="boundary"):
        prepare_base(png(Image.new("RGB", SOURCE_CELL, "blue")))
    with pytest.raises(ValueError, match="blank"):
        prepare_base(png(Image.new("RGB", SOURCE_CELL, BACKGROUND)))
    with pytest.raises(ValueError, match="decode"):
        prepare_base(b"not an image")


def test_atlas_contains_all_poses_and_transparent_unused_cells():
    atlas = Image.open(
        BytesIO(assemble_atlas({row.name: generated_strip(row) for row in ROWS}))
    )
    assert atlas.format == "WEBP"
    assert atlas.size == (1536, 2288)
    assert atlas.crop((192 * 6, 0, 1536, 208)).getchannel("A").getbbox() is None
    assert (
        atlas.crop((192 * 7, 208 * 10, 1536, 2288)).getchannel("A").getbbox()
        is not None
    )


def test_missing_direction_rows_reject_the_atlas():
    with pytest.raises(ValueError, match="exactly"):
        assemble_atlas({row.name: b"" for row in ROWS[:9]})
