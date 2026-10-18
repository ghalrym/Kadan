from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

from fastapi.testclient import TestClient
from PIL import Image

from kadan.api.config import Settings
from kadan.api.server import create_app

PET_ID = UUID("10000000-0000-4000-8000-00000000000a")


def client(import_atlas):
    app = create_app(Settings(allowed_hosts=["testserver"]))
    # Without entering the lifespan, no queue worker or upstream client starts.
    app.state.services = SimpleNamespace(
        pets=SimpleNamespace(import_atlas=import_atlas)
    )
    return TestClient(app)


def webp(rows=11):
    output = BytesIO()
    image = Image.new("RGBA", (1536, rows * 208))
    image.paste((10, 200, 90, 255), (50, 50, 120, 160))
    image.save(output, format="WEBP", lossless=True)
    return output.getvalue()


def test_import_route_names_candidate_and_passes_validated_atlas():
    import_atlas = AsyncMock(
        return_value={
            "id": str(PET_ID),
            "agent_key": "andrew",
            "prompt": "Ember",
            "origin": "imported",
            "status": "ready",
            "version": 1,
            "created_at": "2026-10-17T00:00:00Z",
            "updated_at": "2026-10-17T00:00:00Z",
        }
    )
    response = client(import_atlas).post(
        "/api/agents/andrew/pets/imports",
        data={"id": str(PET_ID), "name": " Ember "},
        files=[
            (
                "files",
                (
                    "pet.json",
                    b'{"displayName": "Fox", "description": "Small"}',
                    "application/json",
                ),
            ),
            ("files", ("spritesheet.webp", webp(9), "image/webp")),
        ],
    )
    assert response.status_code == 201
    key, pet_id, prompt, atlas, version = import_atlas.await_args.args
    assert (key, pet_id, prompt, version) == ("andrew", PET_ID, "Ember\nSmall", 1)
    assert atlas.startswith(b"RIFF")


def test_import_route_rejects_invalid_package_before_storage():
    import_atlas = AsyncMock()
    response = client(import_atlas).post(
        "/api/agents/andrew/pets/imports",
        data={"id": str(PET_ID)},
        files=[("files", ("pet.json", b"{}", "application/json"))],
    )
    assert response.status_code == 422
    assert "spritesheet" in response.json()["detail"]
    import_atlas.assert_not_awaited()
