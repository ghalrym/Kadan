import asyncio
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import UUID

import httpx
import pytest

from kadan.api.config import ServiceError, Settings
from kadan.api.pydantic_models.resources import Message, Resource
from kadan.api.routes.agents.conversations.notes import NoteInput
from kadan.api.routes.agents.resources import PersonaUpdate
from kadan.api.services import LocalServices

CONVERSATION_ID = UUID("10000000-0000-4000-8000-000000000001")
PERSONA_ID = UUID("10000000-0000-4000-8000-000000000002")
MESSAGE_ID = UUID("10000000-0000-4000-8000-000000000003")
NOW = datetime(2026, 10, 17, tzinfo=UTC)


@pytest.mark.parametrize(
    "value",
    [
        {
            "text": "Café 東京 🎤\nsecond line",
            "metadata": {"ready": True, "error": None},
        },
        [{"version": 9223372036854775808, "ratio": 0.125}, [], False],
        None,
        "A response",
    ],
)
def test_upstream_json_preserves_utf8_text_numbers_and_nested_values(value):
    result = run(
        lambda request: httpx.Response(
            200,
            content=json.dumps(value, ensure_ascii=False).encode(),
            headers={"content-type": "application/json"},
        ),
        lambda service: service.json_request("Tsuchi", "GET", "/example"),
    )
    assert result == value


@pytest.mark.parametrize("body", [b"", b'{"unfinished":', b'{"ok": true} extra'])
def test_malformed_upstream_json_remains_an_invalid_response_error(body):
    with pytest.raises(
        ServiceError, match="Tsuchi returned an invalid response"
    ) as error:
        run(
            lambda request: httpx.Response(200, content=body),
            lambda service: service.json_request("Tsuchi", "GET", "/example"),
        )
    assert error.value.status == 502
    assert error.value.code == "invalid_response"


def resource(kind="conversation", messages=None, **changes):
    fields = {
        "id": PERSONA_ID if kind == "persona" else CONVERSATION_ID,
        "agent_key": "test",
        "kind": kind,
        "title": "Test",
        "version": 1,
        "created_at": NOW,
        "updated_at": NOW,
        "messages": messages or [],
    }
    return Resource(**{**fields, **changes})


def ready_settings(status="complete", available=True):
    return {
        "tier": "go",
        "tiers": [
            {
                "id": "go",
                "models": [
                    {
                        "modality": "llm",
                        "model": {
                            "id": "test-model",
                            "status": status,
                            "inference_available": available,
                        },
                    }
                ],
            }
        ],
    }


class Boundary:
    def __init__(self, settings=None):
        self.current = resource()
        self.persona = resource("persona", markdown="Be precise.")
        self.settings = settings if settings is not None else ready_settings()
        self.requests = []
        self.appends = []
        self.inference_body = None
        self.assistant_conflict = False
        self.inference_status = 200
        self.note_conflict = False

    def __call__(self, request):
        self.requests.append((request.method, request.url.path))
        if request.url.path == "/v1/settings":
            return httpx.Response(200, json=self.settings)
        if request.url.path == "/api/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path.endswith(f"/{PERSONA_ID}") and request.method == "GET":
            return httpx.Response(200, json=self.persona.model_dump(mode="json"))
        if request.url.path.endswith(f"/{CONVERSATION_ID}") and request.method == "GET":
            return httpx.Response(200, json=self.current.model_dump(mode="json"))
        if request.url.path.endswith("/messages"):
            value = json.loads(request.content)
            message = Message.model_validate(value["message"])
            self.appends.append(value)
            if (
                (self.assistant_conflict and message.role == "assistant")
                or self.note_conflict
                or value["expected_version"] != self.current.version
            ):
                return httpx.Response(409, json={"detail": "Resource changed"})
            self.current = self.current.model_copy(
                update={
                    "version": self.current.version + 1,
                    "messages": [*self.current.messages, message],
                }
            )
            return httpx.Response(200, json=self.current.model_dump(mode="json"))
        raise AssertionError(
            f"Unexpected boundary call: {request.method} {request.url}"
        )


async def with_services(boundary, operation):
    async with httpx.AsyncClient(transport=httpx.MockTransport(boundary)) as client:
        return await operation(LocalServices(client, Settings(_env_file=None)))


def run(boundary, operation):
    return asyncio.run(with_services(boundary, operation))


@pytest.mark.parametrize(
    ("settings", "state"),
    [
        ({"tier": None, "tiers": []}, "tier_required"),
        (ready_settings(status="not_downloaded"), "model_required"),
        (ready_settings(status="downloading"), "model_required"),
        (ready_settings(status="failed"), "model_required"),
        (ready_settings(available=False), "unavailable"),
        (ready_settings(), "ready"),
        ({"tiers": []}, "unavailable"),
    ],
)
def test_readiness_uses_actual_selected_language_checkpoint(settings, state):
    boundary = Boundary(settings=settings)
    status = run(boundary, lambda service: service.inference_status())
    assert status.state == state
    assert status.settings_url == "http://127.0.0.1:8088/settings"
    assert boundary.requests == [("GET", "/v1/settings")]


def test_unavailable_services_return_actionable_status():
    def offline(request):
        raise httpx.ConnectError("offline", request=request)

    status = run(offline, lambda service: service.status())
    assert status.tsuchi.state == status.hikari.state == "unavailable"
    assert "local service is running" in status.hikari.message


def test_note_persists_without_contacting_hikari():
    boundary = Boundary()
    result = run(
        boundary,
        lambda service: service.save_note(
            "test",
            CONVERSATION_ID,
            NoteInput(
                content="A saved thought", expected_version=1, message_id=MESSAGE_ID
            ),
        ),
    )
    assert result.version == 2
    assert result.messages[0].content == "A saved thought"
    assert all(not path.startswith("/v1/") for _, path in boundary.requests)


class ChainBoundary:
    def __init__(self, rows):
        self.rows = rows
        self.created = []

    def __call__(self, request):
        assert request.url.path == "/api/agents/test/resources"
        if request.method == "GET":
            assert request.url.params["kind"] == "conversation"
            return httpx.Response(
                200,
                json=[
                    row.model_dump(mode="json")
                    for row in self.rows[
                        int(request.url.params["offset"]) : int(
                            request.url.params["offset"]
                        )
                        + 200
                    ]
                ],
            )
        body = json.loads(request.content)
        assert body == {"kind": "conversation", "title": "Chain"}
        created = resource(id=UUID(int=len(self.created) + 1), title=body["title"])
        self.created.append(created)
        self.rows.append(created)
        return httpx.Response(201, json=created.model_dump(mode="json"))


def test_chain_is_created_once_for_concurrent_first_use():
    boundary = ChainBoundary([])

    async def operation(service):
        return await asyncio.gather(*(service.chain("test") for _ in range(3)))

    chains = run(boundary, operation)
    assert len(boundary.created) == 1
    assert {chain.id for chain in chains} == {boundary.created[0].id}


def test_chain_reuses_the_oldest_conversation():
    older = resource(id=UUID(int=7), created_at=datetime(2026, 1, 1, tzinfo=UTC))
    newer = resource(id=UUID(int=8))
    boundary = ChainBoundary([newer, older])
    assert run(boundary, lambda service: service.chain("test")).id == older.id
    assert boundary.created == []


def test_chain_finds_oldest_beyond_first_page_and_after_reordering():
    older = resource(id=UUID(int=999), created_at=datetime(2025, 1, 1, tzinfo=UTC))
    rows = [resource(id=UUID(int=index + 1)) for index in range(400)]
    boundary = ChainBoundary([*rows, older])
    assert run(boundary, lambda service: service.chain("test")).id == older.id
    boundary.rows = [older, *reversed(rows)]
    assert run(boundary, lambda service: service.chain("test")).id == older.id
    assert boundary.created == []


def test_edit_persona_preserves_existing_metadata():
    persona = resource(
        "persona",
        markdown="Old",
        tags=["personal"],
        parent_id=CONVERSATION_ID,
        attachment_ids=[MESSAGE_ID],
        sources=[{"label": "Source", "url": "https://example.com"}],
    )
    calls = []

    def boundary(request):
        if request.method == "GET":
            return httpx.Response(200, json=persona.model_dump(mode="json"))
        body = json.loads(request.content)
        calls.append(body)
        result = persona.model_dump(mode="json") | body | {"version": 2}
        result.pop("expected_version")
        return httpx.Response(200, json=result)

    result = run(
        boundary,
        lambda service: service.update_persona(
            "test",
            PERSONA_ID,
            PersonaUpdate(title="Edited", markdown="New", expected_version=1),
        ),
    )
    assert result.markdown == "New"
    assert calls[0]["attachment_ids"] == [str(MESSAGE_ID)]
    assert calls[0]["tags"] == ["personal"]
    assert calls[0]["parent_id"] == str(CONVERSATION_ID)


@pytest.mark.parametrize("unavailable", [False, True])
def test_docker_status_requests_internal_origins_and_returns_browser_links(unavailable):
    requests = []

    def boundary(request):
        requests.append(str(request.url))
        if unavailable:
            raise httpx.ConnectError("offline", request=request)
        if request.url.host == "tsuchi-api":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"tier": None, "tiers": []})

    async def status():
        settings = Settings(
            tsuchi_url="http://tsuchi-api:8000",
            hikari_url="http://hikari-api:8000",
            tsuchi_browser_url="http://localhost:18620",
            hikari_browser_url="http://localhost:18088",
            _env_file=None,
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(boundary)) as client:
            return await LocalServices(client, settings).status()

    result = asyncio.run(status())
    assert set(requests) == {
        "http://tsuchi-api:8000/api/health",
        "http://hikari-api:8000/v1/settings",
    }
    assert result.tsuchi.state == ("unavailable" if unavailable else "ready")
    assert result.hikari.state == ("unavailable" if unavailable else "tier_required")
    assert result.tsuchi.browser_url == "http://localhost:18620"
    assert result.hikari.settings_url == "http://localhost:18088/settings"


@pytest.mark.parametrize(
    ("lifecycle", "state"),
    [
        (
            httpx.Response(200, json={"state": "loading", "model_id": "test-model"}),
            "loading",
        ),
        (
            httpx.Response(200, json={"state": "ready", "model_id": "test-model"}),
            "ready",
        ),
        (httpx.Response(200, json={"state": "ready", "model_id": "other"}), "unloaded"),
        (httpx.Response(200, json={"state": "unloaded", "model_id": None}), "unloaded"),
        (httpx.Response(503, json={"detail": "Busy"}), None),
        (httpx.Response(200, json={"state": "warming"}), None),
    ],
)
def test_status_reports_whether_the_selected_model_is_loaded(lifecycle, state):
    boundary = Boundary()

    def transport(request):
        if request.url.path == "/model-lifecycle":
            return lifecycle
        return boundary(request)

    status = run(transport, lambda service: service.status())
    assert status.hikari.state == "ready"
    assert status.hikari.model_state == state


def test_assignment_proxy_accepts_storage_no_content_response():
    async def check():
        def respond(request):
            assert request.method == "POST"
            assert request.url.path == "/api/agents/test/runner"
            return httpx.Response(204)

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            services = LocalServices(client, Settings(_env_file=None))
            assert (
                await services.json_request(
                    "Tsuchi",
                    "POST",
                    "/api/agents/test/runner",
                    json={"runner_id": "runner-2", "expected_runner_id": "runner-1"},
                )
                is None
            )

    asyncio.run(check())


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        ({"context_limit": 65536, "architecture_context_limit": 262144}, 65536),
        ({"context_limit": None, "architecture_context_limit": 262144}, 262144),
        ({}, None),
    ],
)
def test_ready_status_reports_the_selected_models_context_window(limits, expected):
    settings = ready_settings()
    settings["tiers"][0]["models"][0]["model"].update(limits)
    status = run(
        Boundary(settings=settings), lambda service: service.inference_status()
    )
    assert status.context_limit == expected


@pytest.mark.parametrize("present", [True, False])
def test_agent_persona_read_is_scoped_and_allows_absent_persona(present):
    persona = resource("persona", markdown="Be precise.")

    def response(request):
        assert request.method == "GET"
        assert request.url.path == "/api/agents/test/persona"
        return httpx.Response(
            200,
            content=json.dumps(persona.model_dump(mode="json") if present else None),
        )

    result = run(response, lambda service: service.agent_persona("test"))
    assert result == (persona if present else None)


@pytest.mark.parametrize(
    "changes",
    [
        {"agent_key": "other"},
        {"kind": "memory"},
        {"deleted_at": NOW},
    ],
)
def test_agent_persona_rejects_wrong_owner_kind_or_trashed_resource(changes):
    persona = resource(**{"kind": "persona", **changes})
    with pytest.raises(ServiceError):
        run(
            lambda request: httpx.Response(200, json=persona.model_dump(mode="json")),
            lambda service: service.agent_persona("test"),
        )


@pytest.mark.parametrize(
    "owned_present, selected",
    [
        (False, None),
        (True, None),
        (False, "other"),
        (True, "same"),
        (True, "other"),
    ],
)
def test_call_instruction_snapshots_owned_and_selected_personas(
    owned_present, selected
):
    async def check():
        service = LocalServices(None, Settings(_env_file=None))
        owned = (
            resource("persona", markdown="Owned instruction.")
            if owned_present
            else None
        )
        selected_id = (
            UUID("10000000-0000-4000-8000-000000000004")
            if selected == "other"
            else PERSONA_ID
            if selected
            else None
        )
        service.agent_persona = AsyncMock(return_value=owned)
        service.resource = AsyncMock(
            return_value=resource(
                "persona",
                id=selected_id or PERSONA_ID,
                markdown="Selected instruction.",
            )
        )
        instruction = await service.call_instruction("test", selected_id)
        assert "Have a spoken conversation." in instruction
        assert "You have no tools." in instruction
        assert instruction.count("Owned instruction.") == int(owned_present)
        assert instruction.count("Selected instruction.") == int(selected == "other")
        service.agent_persona.assert_awaited_once_with("test")
        if selected == "other":
            service.resource.assert_awaited_once_with("test", selected_id)
        else:
            service.resource.assert_not_awaited()
        if owned:
            owned.markdown = "Later edit."
            assert "Later edit." not in instruction

    asyncio.run(check())


@pytest.mark.parametrize(
    "changes",
    [
        {"agent_key": "other"},
        {"kind": "memory"},
        {"deleted_at": NOW},
    ],
)
def test_call_instruction_rejects_unavailable_selected_persona(changes):
    async def check():
        service = LocalServices(None, Settings(_env_file=None))
        service.agent_persona = AsyncMock(return_value=None)
        service.resource = AsyncMock(
            return_value=resource(**{"kind": "persona", **changes})
        )
        with pytest.raises(
            ServiceError, match="selected persona is no longer available"
        ):
            await service.call_instruction("test", PERSONA_ID)

    asyncio.run(check())


@pytest.mark.parametrize("with_selected", [False, True])
def test_call_instruction_preserves_full_text_at_limit_and_rejects_overflow(
    with_selected,
):
    async def check():
        service = LocalServices(None, Settings(_env_file=None))
        owned = resource("persona", markdown="")
        service.agent_persona = AsyncMock(return_value=owned)
        selected_id = (
            UUID("10000000-0000-4000-8000-000000000004") if with_selected else None
        )
        service.resource = AsyncMock(
            return_value=resource(
                "persona",
                id=selected_id or PERSONA_ID,
                markdown="Selected instruction.",
            )
        )
        base = await service.call_instruction("test", selected_id)
        owned.markdown = "x" * (8000 - len(base))
        instruction = await service.call_instruction("test", selected_id)
        assert len(instruction) == 8000
        assert owned.markdown in instruction
        owned.markdown += "x"
        with pytest.raises(ServiceError, match="8,000-character limit"):
            await service.call_instruction("test", selected_id)

    asyncio.run(check())
