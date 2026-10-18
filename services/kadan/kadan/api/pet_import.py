"""Read Codex (ChatGPT) pet packages: a pet.json plus an 8-column spritesheet.

Packages arrive as a zip of the pet folder, or as the folder's loose files.
Only the atlas is kept, re-encoded as lossless WebP, so imported artwork is
served from the same validated format as generated pets.
"""

import posixpath
import zipfile
from io import BytesIO
from typing import Literal

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .pet_artwork import CELL, COLUMNS, _encode

MAX_FILE_BYTES = 20 * 1024**2
MAX_ZIP_MEMBERS = 64
ATLAS_VERSIONS = {
    (COLUMNS * CELL[0], 9 * CELL[1]): 1,
    (COLUMNS * CELL[0], 11 * CELL[1]): 2,
}
IMAGE_SUFFIXES = (".webp", ".png")


class CodexPetJson(BaseModel):
    """The fields Kadan reads from a Codex pet.json; others are ignored."""

    model_config = ConfigDict(extra="ignore")
    id: str | None = Field(default=None, max_length=200)
    displayName: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    spriteVersionNumber: Literal[1, 2] | None = None
    spritesheetPath: str | None = Field(default=None, max_length=500)


class ImportedPet(BaseModel):
    name: str
    description: str
    sprite_version: Literal[1, 2]
    atlas: bytes

    @property
    def prompt(self) -> str:
        return f"{self.name}\n{self.description}" if self.description else self.name


def prepare_atlas(data: bytes) -> tuple[bytes, Literal[1, 2]]:
    """Validate a v1 or v2 Codex atlas and return it as lossless WebP."""
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in {"PNG", "WEBP"}:
                raise ValueError("The spritesheet must be a PNG or WebP image")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("The spritesheet must be a single still image")
            version = ATLAS_VERSIONS.get(image.size)
            if version is None:
                width, height = image.size
                raise ValueError(
                    f"The spritesheet is {width}x{height}; Codex pets are "
                    "1536x1872 (v1) or 1536x2288 (v2)"
                )
            atlas = image.convert("RGBA")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The spritesheet could not be read as an image") from exc
    if atlas.crop((0, 0, *CELL)).getchannel("A").getbbox() is None:
        raise ValueError("The first idle frame of the spritesheet is empty")
    return _encode(atlas, "WEBP"), version


def _pet_json(data: bytes) -> CodexPetJson:
    try:
        return CodexPetJson.model_validate_json(data)
    except ValidationError as exc:
        raise ValueError("pet.json is not a valid Codex pet description") from exc


def _from_zip(data: bytes) -> tuple[CodexPetJson | None, bytes, str]:
    try:
        archive = zipfile.ZipFile(BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("The zip file could not be opened") from exc
    with archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) > MAX_ZIP_MEMBERS:
            raise ValueError("The zip contains too many files to be a pet package")
        names = {posixpath.normpath(item.filename): item for item in members}

        def read(name: str) -> bytes:
            # Limit by declared and actual size; a crafted header cannot inflate past it.
            item = names[name]
            if item.file_size > MAX_FILE_BYTES:
                raise ValueError(f"{name} is larger than 20 MB")
            with archive.open(item) as file:
                content = file.read(MAX_FILE_BYTES + 1)
            if len(content) > MAX_FILE_BYTES:
                raise ValueError(f"{name} is larger than 20 MB")
            return content

        manifests = sorted(
            (name for name in names if posixpath.basename(name) == "pet.json"),
            key=lambda name: name.count("/"),
        )
        if manifests:
            if len(manifests) > 1 and manifests[0].count("/") == manifests[1].count(
                "/"
            ):
                raise ValueError(
                    "The zip contains more than one pet; import one at a time"
                )
            pet = _pet_json(read(manifests[0]))
            sheet = posixpath.normpath(
                posixpath.join(
                    posixpath.dirname(manifests[0]),
                    pet.spritesheetPath or "spritesheet.webp",
                )
            )
            if sheet.startswith(("../", "/")) or sheet not in names:
                raise ValueError(f"The zip does not contain the spritesheet {sheet}")
            return pet, read(sheet), sheet
        images = [name for name in names if name.lower().endswith(IMAGE_SUFFIXES)]
        if len(images) != 1:
            raise ValueError(
                "The zip needs a pet.json or exactly one spritesheet image"
            )
        return None, read(images[0]), images[0]


def read_package(files: list[tuple[str, bytes]]) -> ImportedPet:
    """Read uploaded (filename, content) pairs: one zip, or a spritesheet with an optional pet.json."""
    pet: CodexPetJson | None = None
    sheet: tuple[str, bytes] | None = None
    for filename, data in files:
        name = posixpath.basename(filename.replace("\\", "/")).lower()
        if len(data) > MAX_FILE_BYTES:
            raise ValueError(f"{filename} is larger than 20 MB")
        if name.endswith(".zip"):
            if len(files) != 1:
                raise ValueError("Upload a zip on its own")
            pet, content, sheet_name = _from_zip(data)
            sheet = (sheet_name, content)
        elif name == "pet.json":
            if pet is not None:
                raise ValueError("Upload one pet.json at a time")
            pet = _pet_json(data)
        elif name.endswith(IMAGE_SUFFIXES):
            if sheet is not None:
                raise ValueError("Upload one spritesheet at a time")
            sheet = (filename, data)
        else:
            raise ValueError(f"{filename} is not part of a Codex pet package")
    if sheet is None:
        raise ValueError(
            "Add the pet's spritesheet (.webp or .png) or a zip of its folder"
        )
    atlas, version = prepare_atlas(sheet[1])
    if pet and pet.spriteVersionNumber and pet.spriteVersionNumber != version:
        raise ValueError(
            f"pet.json declares sprite version {pet.spriteVersionNumber}, "
            f"but the spritesheet is a v{version} atlas"
        )
    stem = posixpath.splitext(posixpath.basename(sheet[0]))[0]
    name = next(
        (
            value.strip()
            for value in (
                pet and pet.displayName,
                pet and pet.id,
                None if stem.lower() == "spritesheet" else stem,
            )
            if value and value.strip()
        ),
        "Imported pet",
    )
    description = (pet.description or "").strip() if pet else ""
    return ImportedPet(
        name=name, description=description, sprite_version=version, atlas=atlas
    )
