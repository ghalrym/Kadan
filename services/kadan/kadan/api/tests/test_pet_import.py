import asyncio
import hashlib
import json
import zipfile
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid5

import pytest
from PIL import Image

from kadan.api.pet_import import prepare_atlas, read_package
from kadan.api.pet_jobs import PetJobs

KEY = "andrew"
PET_ID = UUID("10000000-0000-4000-8000-000000000009")


def sheet(rows=11, size=None, format="WEBP", empty=False):
    image = Image.new("RGBA", size or (1536, rows * 208))
    if not empty:
        image.paste((200, 120, 40, 255), (60, 40, 130, 190))
    output = BytesIO()
    image.save(
        output, format=format, **({"lossless": True} if format == "WEBP" else {})
    )
    return output.getvalue()


def pet_json(**fields):
    return json.dumps(
        {
            "id": "copper-fox",
            "displayName": "Copper Fox",
            "description": "A small fox",
            "spriteVersionNumber": 2,
            "spritesheetPath": "spritesheet.webp",
            **fields,
        }
    ).encode()


def zipped(members):
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return output.getvalue()


@pytest.mark.parametrize(("rows", "version"), [(9, 1), (11, 2)])
def test_atlas_version_follows_dimensions_and_is_reencoded(rows, version):
    atlas, detected = prepare_atlas(sheet(rows, format="PNG"))
    assert detected == version
    with Image.open(BytesIO(atlas)) as image:
        assert (image.format, image.size, image.mode) == (
            "WEBP",
            (1536, rows * 208),
            "RGBA",
        )


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (sheet(size=(1536, 2000)), "1536x2000"),
        (sheet(empty=True), "idle frame"),
        (sheet(format="GIF"), "PNG or WebP"),
        (b"not an image", "could not be read"),
    ],
)
def test_invalid_atlas_is_rejected(data, message):
    with pytest.raises(ValueError, match=message):
        prepare_atlas(data)


def test_loose_folder_files_use_pet_json_metadata():
    pet = read_package([("pet.json", pet_json()), ("spritesheet.webp", sheet())])
    assert (pet.name, pet.description, pet.sprite_version) == (
        "Copper Fox",
        "A small fox",
        2,
    )
    assert pet.prompt == "Copper Fox\nA small fox"


def test_zip_of_pet_folder_resolves_custom_spritesheet_path():
    data = zipped(
        {
            "copper-fox/pet.json": pet_json(
                spritesheetPath="art/sheet.png", spriteVersionNumber=1
            ),
            "copper-fox/art/sheet.png": sheet(9, format="PNG"),
        }
    )
    pet = read_package([("copper-fox.zip", data)])
    assert (pet.name, pet.sprite_version) == ("Copper Fox", 1)


def test_spritesheet_alone_is_named_from_its_file():
    pet = read_package([("ember.png", sheet(format="PNG"))])
    assert (pet.name, pet.description, pet.prompt) == ("ember", "", "ember")
    assert read_package([("spritesheet.webp", sheet())]).name == "Imported pet"


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ([("pet.json", pet_json())], "Add the pet's spritesheet"),
        (
            [("pet.json", pet_json(spriteVersionNumber=1)), ("s.webp", sheet())],
            "declares sprite version 1",
        ),
        ([("pet.json", b"{")], "valid Codex pet"),
        ([("notes.txt", b"hi")], "not part of a Codex pet"),
        ([("a.webp", sheet()), ("b.webp", sheet())], "one spritesheet"),
        (
            [("pet.zip", zipped({"pet.json": pet_json(spritesheetPath="../x.webp")}))],
            "does not contain",
        ),
        (
            [("pet.zip", zipped({"a/pet.json": pet_json(), "b/pet.json": pet_json()}))],
            "more than one pet",
        ),
        ([("pet.zip", b"nope")], "could not be opened"),
        (
            [("pet.zip", zipped({"pet.json": pet_json()})), ("s.webp", sheet())],
            "zip on its own",
        ),
    ],
)
def test_invalid_packages_explain_what_is_wrong(files, message):
    with pytest.raises(ValueError, match=message):
        read_package(files)


def test_import_uploads_derived_atlas_then_records_ready_candidate():
    calls = []

    async def json_request(service, method, path, **kwargs):
        calls.append((method, path, kwargs))
        if path.endswith("/assets"):
            return {
                "id": kwargs["data"]["asset_id"],
                "agent_key": KEY,
                "media_type": "image/webp",
                "sha256": hashlib.sha256(kwargs["files"]["file"][1]).hexdigest(),
                "is_image": True,
            }
        body = kwargs["json"]
        return {
            **body,
            "agent_key": KEY,
            "origin": "imported",
            "status": "ready",
            "stage": "Imported",
            "version": 1,
            "created_at": "2026-10-17T00:00:00Z",
            "updated_at": "2026-10-17T00:00:00Z",
        }

    jobs = PetJobs(SimpleNamespace(json_request=AsyncMock(side_effect=json_request)))
    atlas, version = prepare_atlas(sheet(9))
    job = asyncio.run(jobs.import_atlas(KEY, PET_ID, "Copper Fox", atlas, version))
    assert [(method, path) for method, path, _ in calls] == [
        ("POST", f"/api/agents/{KEY}/assets"),
        ("POST", f"/api/agents/{KEY}/pets/imports"),
    ]
    assert calls[0][2]["data"]["asset_id"] == str(uuid5(PET_ID, "atlas"))
    manifest = calls[1][2]["json"]["manifest"]
    assert (
        manifest["sprite_version"],
        manifest["rows"],
        manifest["base_asset_id"],
    ) == (1, 9, None)
    assert manifest["animations"][0]["name"] == "idle"
    assert (job.origin, job.status, job.manifest.atlas_asset_id) == (
        "imported",
        "ready",
        uuid5(PET_ID, "atlas"),
    )
