import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, sentinel
from uuid import UUID

import pytest
from fastapi import Request

from kadan.api.config import ServiceError
from kadan.api.pet_import import ImportedPet
from kadan.api.routes.agents import pets

PET_ID = UUID("10000000-0000-4000-8000-00000000000a")


def request(import_atlas):
    return Request(
        {
            "type": "http",
            "app": SimpleNamespace(
                state=SimpleNamespace(
                    services=SimpleNamespace(
                        pets=SimpleNamespace(import_atlas=import_atlas)
                    )
                )
            ),
        }
    )


@pytest.mark.parametrize(
    ("name", "expected_prompt"),
    [(" Ember ", "Ember\nSmall"), ("  ", "Fox\nSmall")],
)
def test_import_route_names_candidate_and_passes_validated_atlas(
    monkeypatch, name, expected_prompt
):
    import_atlas = AsyncMock(return_value=sentinel.saved_pet)
    upload = SimpleNamespace(
        filename="pet.zip", read=AsyncMock(return_value=b"uploaded package")
    )
    parse_package = AsyncMock(
        return_value=ImportedPet(
            name="Fox", description="Small", sprite_version=1, atlas=b"validated atlas"
        )
    )
    monkeypatch.setattr(pets.asyncio, "to_thread", parse_package)

    result = asyncio.run(
        pets.import_pet("andrew", request(import_atlas), PET_ID, [upload], name)
    )

    assert result is sentinel.saved_pet
    upload.read.assert_awaited_once_with(pets.MAX_FILE_BYTES + 1)
    parse_package.assert_awaited_once_with(
        pets.read_package, [("pet.zip", b"uploaded package")]
    )
    import_atlas.assert_awaited_once_with(
        "andrew", PET_ID, expected_prompt, b"validated atlas", 1
    )


def test_import_route_rejects_invalid_package_before_storage(monkeypatch):
    import_atlas = AsyncMock()
    upload = SimpleNamespace(filename="pet.json", read=AsyncMock(return_value=b"{}"))
    error = ValueError("Add the pet's spritesheet")
    parse_package = AsyncMock(side_effect=error)
    monkeypatch.setattr(pets.asyncio, "to_thread", parse_package)

    with pytest.raises(ServiceError, match="Add the pet's spritesheet") as raised:
        asyncio.run(pets.import_pet("andrew", request(import_atlas), PET_ID, [upload]))

    assert raised.value.status == 422
    assert raised.value.code == "invalid_pet_package"
    assert raised.value.__cause__ is error
    parse_package.assert_awaited_once_with(pets.read_package, [("pet.json", b"{}")])
    import_atlas.assert_not_awaited()
