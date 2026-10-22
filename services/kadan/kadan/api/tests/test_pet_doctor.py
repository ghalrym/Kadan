import random

import pytest
from PIL import Image, ImageDraw

from kadan.api.pet_artwork import SOURCE_CELL, _bounds, _encode, _png
from kadan.api.pet_doctor import PetDoctor


def character(*, raised_arm=False, crouch=False, jump=False):
    image = Image.new("RGBA", SOURCE_CELL)
    draw = ImageDraw.Draw(image)
    draw.ellipse((142, 140, 242, 236), fill=(110, 170, 210))
    draw.rectangle((157, 226, 227, 320), fill=(180, 110, 60))
    draw.rectangle((157, 318, 177, 367), fill=(80, 110, 160))
    draw.rectangle((207, 318, 227, 367), fill=(80, 110, 160))
    if raised_arm:
        draw.line((225, 245, 267, 200, 267, 140), fill=(180, 110, 60), width=15)
    else:
        draw.line((225, 245, 248, 290), fill=(180, 110, 60), width=15)
    # Distinctive face/body markings, without relying on the silhouette.
    rng = random.Random(8)
    for _ in range(65):
        x, y = rng.randint(166, 215), rng.randint(155, 302)
        radius = rng.randint(2, 5)
        color = tuple(rng.randint(10, 240) for _ in range(3))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    if crouch:
        image.paste((0, 0, 0, 0), (0, 318, 384, 416))
        image = image.transform(
            SOURCE_CELL, Image.Transform.AFFINE, (1, 0, 0, 0, 1, -24)
        )
        draw = ImageDraw.Draw(image)
        draw.line((166, 341, 143, 353, 171, 367), fill=(80, 110, 160), width=14)
        draw.line((217, 341, 240, 353, 212, 367), fill=(80, 110, 160), width=14)
    if jump:
        image = image.transform(
            SOURCE_CELL, Image.Transform.AFFINE, (1, 0, 0, 0, 1, 40)
        )
    return image


def scaled(image, scale):
    bottom = _bounds(image)[3]
    return image.transform(
        SOURCE_CELL,
        Image.Transform.AFFINE,
        (1 / scale, 0, 192 * (1 - 1 / scale), 0, 1 / scale, bottom * (1 - 1 / scale)),
        resample=Image.Resampling.BICUBIC,
    )


@pytest.mark.parametrize(
    "pose", [{}, {"raised_arm": True}, {"crouch": True}, {"jump": True}]
)
@pytest.mark.parametrize("scale", [0.8, 1.2])
def test_corrects_body_scale_without_fitting_pose_silhouette(pose, scale):
    doctor = PetDoctor(_encode(character()))
    frame = scaled(character(**pose), scale)
    output, report = doctor.correct(_encode(frame))
    assert report.outcome == "corrected", report.reason
    assert report.scale == pytest.approx(1 / scale, abs=0.04)
    corrected = _png(output)
    assert corrected.size == SOURCE_CELL
    assert _bounds(corrected)[3] == pytest.approx(_bounds(frame)[3], abs=2)
    if pose.get("jump"):
        assert _bounds(corrected)[3] < 340


@pytest.mark.parametrize(
    "pose", [{}, {"raised_arm": True}, {"crouch": True}, {"jump": True}]
)
def test_articulation_alone_does_not_resize_character(pose):
    frame = _encode(character(**pose))
    output, report = PetDoctor(_encode(character())).correct(frame)
    assert report.outcome == "consistent", report.reason
    assert output == frame


def test_featureless_character_is_left_unchanged():
    reference = Image.new("RGBA", SOURCE_CELL)
    ImageDraw.Draw(reference).rectangle((150, 150, 230, 360), fill="white")
    frame = _encode(scaled(reference, 0.8))
    output, report = PetDoctor(_encode(reference)).correct(frame)
    assert report.outcome == "skipped"
    assert output == frame


def test_transparent_background_colors_do_not_affect_measurement():
    reference = character()
    frame = Image.new("RGBA", SOURCE_CELL, (255, 0, 255, 0))
    frame.paste(reference, mask=reference.getchannel("A"))
    png = _encode(frame)
    output, report = PetDoctor(_encode(reference)).correct(png)
    assert report.outcome == "consistent"
    assert output == png


def test_correction_that_would_clip_is_skipped():
    frame = scaled(character(), 0.8)
    frame = frame.transform(SOURCE_CELL, Image.Transform.AFFINE, (1, 0, 0, 0, 1, 150))
    png = _encode(frame)
    output, report = PetDoctor(_encode(character())).correct(png)
    assert report.outcome == "skipped"
    assert "clip" in report.reason
    assert output == png
