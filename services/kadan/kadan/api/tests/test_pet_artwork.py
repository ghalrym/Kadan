from io import BytesIO

import pytest
from PIL import Image, ImageDraw

from kadan.api.pet_artwork import (
    BACKGROUND,
    CELL,
    ROWS,
    SOURCE_CELL,
    assemble_atlas,
    character_sheet_prompt,
    frame_name,
    frame_prompt,
    prepare_base,
    prepare_character_sheet,
    prepare_frame,
    sheet_base_prompt,
    validate_row,
)


def png(image):
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def character(background=BACKGROUND):
    image = Image.new("RGBA", SOURCE_CELL, background)
    draw = ImageDraw.Draw(image)
    draw.rectangle((120, 100, 260, 360), fill="blue")
    # This enclosed mark shares the chroma color, but is part of the artwork.
    draw.rectangle((170, 140, 200, 170), fill=BACKGROUND)
    return image


def generated_frame(index, background=BACKGROUND):
    frame = character(background)
    arm_y = 180 + index * 10
    ImageDraw.Draw(frame).rectangle((260, arm_y, 300, arm_y + 12), fill="blue")
    return png(frame)


def prepared_frames():
    return {
        frame_name(row, index): generated_frame(index, (0, 0, 0, 0))
        for row in ROWS
        for index in range(row.frame_count)
    }


def test_canonical_reference_has_transparent_margins_and_shared_floor():
    base = Image.open(BytesIO(prepare_base(png(character()))))
    assert base.size == SOURCE_CELL
    assert base.getpixel((0, 0))[3] == 0
    assert base.getchannel("A").getbbox()[3] == 368
    assert base.getpixel((192, 164)) == (*BACKGROUND, 255)


def test_character_sheet_stays_a_reference_and_is_not_fitted_into_one_cell():
    sheet = Image.new("RGB", (1024, 1024), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    for x in [100, 400, 700]:
        draw.rectangle((x, 100, x + 80, 400), fill="blue")
    prepared = Image.open(BytesIO(prepare_character_sheet(png(sheet))))
    assert prepared.size == (1024, 1024)
    assert prepared.convert("RGB").tobytes() == sheet.tobytes()
    with pytest.raises(ValueError, match="dimensions"):
        prepare_frame(png(prepared), ROWS[0], 0)


def test_sheet_prompts_distinguish_reference_views_from_individual_output_poses():
    assert "SAME character" in character_sheet_prompt("Ink owl")
    for prompt in [
        sheet_base_prompt("Ink owl"),
        *(
            frame_prompt("Ink owl", row, index, character_sheet=True)
            for row in ROWS
            for index in range(row.frame_count)
        ),
    ]:
        assert "exactly ONE character in ONE pose" in prompt
        assert "do not reproduce the sheet" in prompt
        assert "384x416" in prompt
        assert "Preserve the reference's body scale" not in prompt
        assert "same standing level as the reference" not in prompt


def test_frame_preserves_body_scale_position_and_airborne_height():
    row = ROWS[4]
    ground = Image.open(BytesIO(prepare_frame(generated_frame(0), row, 0)))
    airborne = Image.new("RGB", SOURCE_CELL, BACKGROUND)
    airborne.paste(character().crop((0, 50, 384, 416)), (0, 0))
    jump = Image.open(BytesIO(prepare_frame(png(airborne), row, 2)))
    assert ground.size == jump.size == SOURCE_CELL
    assert jump.getchannel("A").getbbox() == (120, 50, 261, 311)
    assert ground.getchannel("A").getbbox() == (120, 100, 301, 361)
    assert jump.getpixel((192, 100)) == (*BACKGROUND, 255)
    assert ground.getpixel((0, 0)) == jump.getpixel((0, 0)) == (0, 0, 0, 0)


@pytest.mark.parametrize("kind", ["blank", "cropped", "wrong-size", "border"])
def test_invalid_generated_frames_fail_before_use(kind):
    frame = character()
    if kind == "blank":
        frame = Image.new("RGB", SOURCE_CELL, BACKGROUND)
    elif kind == "cropped":
        ImageDraw.Draw(frame).rectangle((0, 100, 150, 200), fill="blue")
    elif kind == "wrong-size":
        frame = frame.crop((0, 0, frame.width - 1, frame.height))
    elif kind == "border":
        ImageDraw.Draw(frame).rectangle((0, 0, 383, 415), outline="black")
    with pytest.raises(ValueError):
        prepare_frame(png(frame), ROWS[3], 0)


def test_source_margin_must_also_survive_final_cell_downsampling():
    frame = Image.new("RGBA", SOURCE_CELL)
    ImageDraw.Draw(frame).rectangle((3, 100, 150, 360), fill="blue")
    assert frame.getchannel("A").getbbox()[0] == 3
    with pytest.raises(ValueError, match="boundary"):
        prepare_frame(png(frame), ROWS[3], 0)


def test_invalid_base_is_rejected_instead_of_hidden_by_resizing():
    with pytest.raises(ValueError, match="boundary"):
        prepare_base(png(Image.new("RGB", SOURCE_CELL, "blue")))
    with pytest.raises(ValueError, match="blank"):
        prepare_base(png(Image.new("RGB", SOURCE_CELL, BACKGROUND)))
    with pytest.raises(ValueError, match="decode"):
        prepare_base(b"not an image")


def test_atlas_contains_all_poses_and_transparent_unused_cells():
    frames = prepared_frames()
    assert len(frames) == 73
    encoded = assemble_atlas(frames)
    assert assemble_atlas(dict(reversed(list(frames.items())))) == encoded
    atlas = Image.open(BytesIO(encoded))
    assert atlas.format == "WEBP"
    assert atlas.size == (1536, 2288)
    assert atlas.crop((192 * 6, 0, 1536, 208)).getchannel("A").getbbox() is None
    assert (
        atlas.crop((192 * 7, 208 * 10, 1536, 2288)).getchannel("A").getbbox()
        is not None
    )
    expected = Image.open(BytesIO(frames["waving/2"])).resize(
        CELL, Image.Resampling.LANCZOS
    )
    actual = atlas.crop((192 * 2, 208 * 3, 192 * 3, 208 * 4))
    assert actual.getchannel("A").tobytes() == expected.getchannel("A").tobytes()
    assert actual.getpixel((90, 80)) == expected.getpixel((90, 80))


def test_repeated_pose_row_rejects_atlas_even_with_different_chroma():
    frames = prepared_frames()
    row = ROWS[0]
    for index in range(row.frame_count):
        frame = character()
        # Background variation must not count as animation.
        ImageDraw.Draw(frame).rectangle((0, 0, 50, 50), fill=(255, index, 255))
        frames[frame_name(row, index)] = prepare_frame(png(frame), row, index)
    with pytest.raises(ValueError, match="repeats one image"):
        assemble_atlas(frames)


@pytest.mark.parametrize("invalid", ["missing", "extra", "unknown"])
def test_incomplete_or_unrecognized_frame_sets_reject_atlas(invalid):
    frames = prepared_frames()
    if invalid == "missing":
        frames.pop("look-1/7")
    elif invalid == "extra":
        frames["idle/6"] = frames["idle/0"]
    else:
        frames["idle/00"] = frames.pop("idle/0")
    with pytest.raises(ValueError, match="exactly"):
        assemble_atlas(frames)


@pytest.mark.parametrize("index", [-1, 6])
def test_frame_index_must_belong_to_animation(index):
    with pytest.raises(ValueError, match="Invalid frame index"):
        prepare_frame(generated_frame(0), ROWS[0], index)


def test_complete_articulated_row_is_valid_before_final_frame_commit():
    row = ROWS[3]
    validate_row(
        row,
        [generated_frame(index, (0, 0, 0, 0)) for index in range(row.frame_count)],
    )


def test_repeated_row_fails_before_final_frame_commit():
    row = ROWS[3]
    with pytest.raises(ValueError, match="repeats one image"):
        validate_row(row, [generated_frame(0, (0, 0, 0, 0))] * row.frame_count)


@pytest.mark.parametrize("count", [3, 5])
def test_row_requires_exact_frame_count_before_final_frame_commit(count):
    with pytest.raises(ValueError, match="exactly 4 frames"):
        validate_row(
            ROWS[3], [generated_frame(index, (0, 0, 0, 0)) for index in range(count)]
        )


def test_row_validation_rechecks_final_cell_bounds():
    row = ROWS[3]
    frames = [generated_frame(index, (0, 0, 0, 0)) for index in range(row.frame_count)]
    narrow_margin = Image.new("RGBA", SOURCE_CELL)
    ImageDraw.Draw(narrow_margin).rectangle((3, 100, 150, 360), fill="blue")
    frames[-1] = png(narrow_margin)
    with pytest.raises(ValueError, match="boundary"):
        validate_row(row, frames)
