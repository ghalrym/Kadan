import asyncio
import base64
import json
from copy import deepcopy
from uuid import UUID

import httpx
import pytest

from kadan.api.config import ServiceError, Settings
from kadan.api.services import LocalServices

ASSET_ID = UUID("10000000-0000-4000-8000-000000000001")
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jhN0AAAAASUVORK5CYII="
)
AGENT = {
    "key": "ai",
    "name": "Ai",
    "created_at": "2026-10-17T00:00:00Z",
    "avatar_asset_id": None,
}
ASSET = {
    "id": str(ASSET_ID),
    "agent_key": "ai",
    "title": "Ai avatar",
    "filename": "avatar.png",
    "media_type": "image/png",
    "size_bytes": len(PNG),
    "sha256": "0" * 64,
    "is_image": True,
    "version": 1,
    "created_at": "2026-10-17T00:00:00Z",
    "updated_at": "2026-10-17T00:00:00Z",
}


class AvatarBoundary:
    def __init__(self):
        self.requests = []
        self.generation = {
            "image": {
                "mime_type": "image/png",
                "images_base64": [base64.b64encode(PNG).decode()],
            }
        }
        self.asset = deepcopy(ASSET)
        self.missing_agent = False
        self.hikari_error = None
        self.storage_error = False

    def __call__(self, request):
        self.requests.append(request)
        if request.url.path == "/api/agents/ai":
            if self.missing_agent:
                return httpx.Response(404, json={"detail": "Agent not found"})
            return httpx.Response(200, json=AGENT)
        if request.url.path == "/v1/images/generations":
            if self.hikari_error:
                raise self.hikari_error
            return httpx.Response(200, json=self.generation)
        if request.url.path == "/api/agents/ai/assets":
            if self.storage_error:
                raise httpx.ReadTimeout("Storage disconnected", request=request)
            return httpx.Response(201, json=self.asset)
        if request.url.path == "/api/agents/ai/avatar":
            body = json.loads(request.content)
            return httpx.Response(
                200, json=AGENT | {"avatar_asset_id": body["asset_id"]}
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")


def run(boundary, operation):
    async def invoke():
        async with httpx.AsyncClient(transport=httpx.MockTransport(boundary)) as client:
            return await operation(LocalServices(client, Settings(_env_file=None)))

    return asyncio.run(invoke())


def generate(service):
    return service.generate_avatar("ai", "A friendly fox portrait")


def test_generated_candidate_is_uploaded_to_its_agent_without_selecting_it():
    boundary = AvatarBoundary()
    asset = run(boundary, generate)
    assert asset.id == ASSET_ID
    assert [(request.method, request.url.path) for request in boundary.requests] == [
        ("GET", "/api/agents/ai"),
        ("POST", "/v1/images/generations"),
        ("POST", "/api/agents/ai/assets"),
    ]
    inference = boundary.requests[1]
    assert json.loads(inference.content) == {
        "prompt": "A friendly fox portrait",
        "aspect": "1:1",
        "count": 1,
    }
    assert inference.extensions["timeout"]["read"] == 1800
    upload = boundary.requests[2]
    assert upload.headers["content-type"].startswith("multipart/form-data;")
    assert PNG in upload.content
    assert b'filename="avatar.png"' in upload.content
    assert b"Ai avatar" in upload.content


def test_unknown_agent_is_rejected_before_generation():
    boundary = AvatarBoundary()
    boundary.missing_agent = True
    with pytest.raises(ServiceError) as raised:
        run(boundary, generate)
    assert raised.value.status == 404
    assert len(boundary.requests) == 1


@pytest.mark.parametrize(
    "images,mime",
    [
        ([], "image/png"),
        (["@@@@"], "image/png"),
        ([base64.b64encode(b"not an image").decode()], "image/png"),
        (["https://example.com/avatar.png"], "image/png"),
        ([base64.b64encode(PNG).decode()] * 2, "image/png"),
        ([base64.b64encode(PNG).decode()], "image/jpeg"),
    ],
)
def test_invalid_images_are_rejected_without_upload_or_remote_url_fetch(images, mime):
    boundary = AvatarBoundary()
    boundary.generation = {"image": {"mime_type": mime, "images_base64": images}}
    with pytest.raises(ServiceError) as raised:
        run(boundary, generate)
    assert raised.value.code == "invalid_image"
    assert len(boundary.requests) == 2


def test_oversize_image_is_rejected_before_upload(monkeypatch):
    monkeypatch.setattr("kadan.api.services.MAX_AVATAR_BYTES", len(PNG) - 1)
    boundary = AvatarBoundary()
    with pytest.raises(ServiceError) as raised:
        run(boundary, generate)
    assert raised.value.code == "invalid_image"
    assert len(boundary.requests) == 2


def test_generation_timeout_is_actionable_and_does_not_upload():
    boundary = AvatarBoundary()
    boundary.hikari_error = httpx.ReadTimeout("timeout")
    with pytest.raises(
        ServiceError, match="Check Hikari before trying again"
    ) as raised:
        run(boundary, generate)
    assert raised.value.code == "generation_unavailable"
    assert len(boundary.requests) == 2


def test_uncertain_upload_reports_possible_saved_image_before_retry():
    boundary = AvatarBoundary()
    boundary.storage_error = True
    with pytest.raises(
        ServiceError, match="image library before generating again"
    ) as raised:
        run(boundary, generate)
    assert raised.value.code == "avatar_storage_failed"


@pytest.mark.parametrize(
    "changes",
    [
        {"is_image": False},
        {"agent_key": "other"},
        {"deleted_at": "2026-10-17T00:00:00Z"},
    ],
)
def test_unusable_stored_asset_is_not_returned_as_candidate(changes):
    boundary = AvatarBoundary()
    boundary.asset.update(changes)
    with pytest.raises(ServiceError) as raised:
        run(boundary, generate)
    assert raised.value.code == "invalid_response"


@pytest.mark.parametrize("asset_id", [ASSET_ID, None])
def test_select_or_remove_avatar_uses_only_tsuchi(asset_id):
    boundary = AvatarBoundary()
    agent = run(boundary, lambda service: service.select_avatar("ai", asset_id))
    assert agent.avatar_asset_id == asset_id
    assert len(boundary.requests) == 1
    request = boundary.requests[0]
    assert request.method == "PUT"
    assert request.url.path == "/api/agents/ai/avatar"


def test_tsuchi_selection_validation_error_reaches_caller():
    def boundary(request):
        return httpx.Response(
            422, json={"detail": "Avatar must be an active image owned by this agent"}
        )

    with pytest.raises(ServiceError, match="active image") as raised:
        run(boundary, lambda service: service.select_avatar("ai", ASSET_ID))
    assert raised.value.status == 422
