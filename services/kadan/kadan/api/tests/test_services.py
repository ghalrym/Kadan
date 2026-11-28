import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID

import httpx
import pytest

from kadan.api.config import ServiceError, Settings
from kadan.api.pydantic_models.resources import Agent, Message, Resource
from kadan.api.routes.agents.conversations.notes import NoteInput
from kadan.api.services import (
    DEFAULT_AGENT_KEY,
    DEFAULT_AGENT_NAME,
    ONBOARDING_GREETING,
    ONBOARDING_GREETING_MESSAGE_ID,
    STARTER_PERSONA_MARKDOWN,
    LocalServices,
    character_persona_section,
)

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


def ready_models(status="complete", available=True, model_id="inkling-2-mini-iq4-nl"):
    return {
        "selected_model_id": model_id,
        "models": [
            {
                "id": model_id,
                "kind": "llm",
                "status": status,
                "inference_available": available,
            }
        ],
    }


class Boundary:
    def __init__(self, settings=None):
        self.current = resource()
        self.persona = resource("persona", markdown="Be precise.")
        self.settings = settings if settings is not None else ready_models()
        self.requests = []
        self.appends = []
        self.inference_body = None
        self.assistant_conflict = False
        self.inference_status = 200
        self.language = "local"
        self.note_conflict = False

    def __call__(self, request):
        self.requests.append((request.method, request.url.path))
        if request.url.path == "/v1/subscriptions":
            return httpx.Response(
                200, json={"capabilities": {"language": self.language}}
            )
        if request.url.path == "/v1/models":
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
        ({"models": []}, "unavailable"),
        (ready_models(status="not_downloaded"), "model_required"),
        (ready_models(status="downloading"), "model_required"),
        (ready_models(status="failed"), "model_required"),
        (ready_models(available=False), "unavailable"),
        (ready_models(), "ready"),
        (ready_models(model_id="inkling-2-mini-q8-0"), "ready"),
        ({}, "unavailable"),
    ],
)
def test_readiness_uses_actual_active_language_checkpoint(settings, state):
    boundary = Boundary(settings=settings)
    status = run(boundary, lambda service: service.inference_status())
    assert status.state == state
    assert boundary.requests == [("GET", "/v1/subscriptions"), ("GET", "/v1/models")]


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_subscription_language_is_ready_without_local_download(provider):
    boundary = Boundary(settings={"models": []})
    boundary.language = provider
    status = run(boundary, lambda service: service.inference_status())
    assert status.state == "ready"
    assert status.model_id is None
    assert {"claude": "Claude", "codex": "ChatGPT"}[provider] in status.message
    assert boundary.requests == [("GET", "/v1/subscriptions")]


@pytest.mark.parametrize(
    ("models", "selected", "expected", "expected_model"),
    [
        ([], None, "unavailable", None),
        (
            [
                {
                    "id": "flux.2-klein-4b",
                    "kind": "image",
                    "status": "complete",
                    "inference_available": True,
                }
            ],
            None,
            "unavailable",
            None,
        ),
        (
            [
                {
                    "id": "inkling-2-mini-iq4-nl",
                    "kind": "llm",
                    "status": "not_downloaded",
                    "inference_available": True,
                }
            ],
            "inkling-2-mini-iq4-nl",
            "model_required",
            "inkling-2-mini-iq4-nl",
        ),
        (
            [
                {
                    "id": "inkling-2-mini-iq4-nl",
                    "kind": "llm",
                    "status": "complete",
                    "inference_available": False,
                }
            ],
            "inkling-2-mini-iq4-nl",
            "unavailable",
            "inkling-2-mini-iq4-nl",
        ),
        (
            [
                {
                    "id": "inkling-2-mini-q8-0",
                    "kind": "llm",
                    "status": "complete",
                    "inference_available": True,
                    "context_limit": 131072,
                }
            ],
            "inkling-2-mini-q8-0",
            "ready",
            "inkling-2-mini-q8-0",
        ),
    ],
)
def test_active_inkling_requires_its_own_downloaded_worker(
    models, selected, expected, expected_model
):
    async def check():
        def boundary(request):
            assert request.method == "GET"
            if request.url.path == "/v1/subscriptions":
                return httpx.Response(200, json={"capabilities": {"language": "local"}})
            assert request.url.path == "/v1/models"
            return httpx.Response(
                200, json={"models": models, "selected_model_id": selected}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(boundary)) as client:
            service = LocalServices(client, Settings(_env_file=None))
            status = await service.inference_status()
            assert status.state == expected
            assert status.model_id == expected_model
            if expected == "ready":
                assert status.context_limit == 131072

    asyncio.run(check())


@pytest.mark.parametrize(
    ("image_provider", "models", "expected_state", "expected_download", "expected_text"),
    [
        (
            "codex",
            [],
            "ready",
            None,
            "ChatGPT",
        ),
        (
            "local",
            [
                {
                    "id": "flux.2-klein-4b",
                    "kind": "image",
                    "status": "complete",
                    "inference_available": True,
                }
            ],
            "ready",
            "complete",
            "ready",
        ),
        (
            "local",
            [
                {
                    "id": "flux.2-klein-4b",
                    "kind": "image",
                    "status": "not_downloaded",
                    "inference_available": True,
                }
            ],
            "model_required",
            "not_downloaded",
            "Kadan Server",
        ),
        (
            "local",
            [
                {
                    "id": "flux.2-klein-4b",
                    "kind": "image",
                    "status": "downloading",
                    "inference_available": True,
                }
            ],
            "model_required",
            "downloading",
            "currently downloading",
        ),
        (
            "local",
            [
                {
                    "id": "flux.2-klein-4b",
                    "kind": "image",
                    "status": "complete",
                    "inference_available": False,
                }
            ],
            "unavailable",
            None,
            "AI models",
        ),
    ],
)
def test_image_inference_status_reports_readiness_and_setup_guidance(
    image_provider, models, expected_state, expected_download, expected_text
):
    async def check():
        def boundary(request):
            if request.url.path == "/v1/subscriptions":
                return httpx.Response(
                    200,
                    json={"capabilities": {"language": "local", "image": image_provider}},
                )
            assert request.url.path == "/v1/models"
            return httpx.Response(200, json={"models": models})

        async with httpx.AsyncClient(transport=httpx.MockTransport(boundary)) as client:
            service = LocalServices(client, Settings(_env_file=None))
            status = await service.image_inference_status()
            assert status.state == expected_state
            assert status.download_state == expected_download
            assert expected_text in status.message

    asyncio.run(check())


def test_unavailable_services_return_actionable_status():
    def offline(request):
        raise httpx.ConnectError("offline", request=request)

    status = run(offline, lambda service: service.status())
    assert status.tsuchi.state == status.hikari.state == "unavailable"
    assert "local service is running" in status.hikari.message


@pytest.mark.parametrize(
    "download",
    ["not_downloaded", "downloading", "cancelling", "cancelled", "failed", "complete"],
)
def test_status_preserves_the_selected_models_reported_download_state(download):
    boundary = Boundary(settings=ready_models(status=download))

    def transport(request):
        if request.url.path == "/model-lifecycle":
            return httpx.Response(200, json={"state": "unloaded"})
        return boundary(request)

    status = run(transport, lambda service: service.status())
    assert status.hikari.download_state == download
    assert status.hikari.state == (
        "ready" if download == "complete" else "model_required"
    )


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


def test_save_agent_persona_persists_owned_persona_and_defaults_title():
    persona = resource("persona", title="Test Agent", markdown="Old")
    calls = []

    def boundary(request):
        if request.method == "GET" and request.url.path.endswith("/persona"):
            return httpx.Response(200, json=persona.model_dump(mode="json"))
        body = json.loads(request.content)
        calls.append(body)
        result = persona.model_dump(mode="json") | body | {"version": 2}
        result.pop("expected_version")
        return httpx.Response(200, json=result)

    result = run(
        boundary,
        lambda service: service.save_agent_persona(
            "test",
            markdown="New instructions",
            expected_version=1,
        ),
    )
    assert result.markdown == "New instructions"
    assert calls[0] == {
        "title": "Test Agent",
        "markdown": "New instructions",
        "expected_version": 1,
    }


@pytest.mark.parametrize("unavailable", [False, True])
def test_docker_status_requests_internal_origins_and_returns_browser_links(unavailable):
    requests = []

    def boundary(request):
        requests.append(str(request.url))
        if unavailable:
            raise httpx.ConnectError("offline", request=request)
        if request.url.host == "tsuchi-api":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/subscriptions":
            return httpx.Response(200, json={"capabilities": {"language": "local"}})
        return httpx.Response(200, json={"models": []})

    async def status():
        settings = Settings(
            tsuchi_url="http://tsuchi-api:8000",
            hikari_url="http://hikari-api:8000",
            _env_file=None,
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(boundary)) as client:
            return await LocalServices(client, settings).status()

    result = asyncio.run(status())
    assert set(requests) == {
        "http://tsuchi-api:8000/api/health",
        "http://hikari-api:8000/v1/subscriptions",
        *(() if unavailable else ("http://hikari-api:8000/v1/models",)),
    }
    assert result.tsuchi.state == ("unavailable" if unavailable else "ready")
    assert result.hikari.state == "unavailable"


@pytest.mark.parametrize(
    ("lifecycle", "state"),
    [
        (
            httpx.Response(
                200, json={"state": "loading", "model_id": "inkling-2-mini-iq4-nl"}
            ),
            "loading",
        ),
        (
            httpx.Response(
                200, json={"state": "ready", "model_id": "inkling-2-mini-iq4-nl"}
            ),
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
    settings = ready_models()
    settings["models"][0].update(limits)
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


@pytest.mark.parametrize("owned_present", [False, True])
def test_call_instruction_snapshots_owned_persona(owned_present):
    async def check():
        service = LocalServices(None, Settings(_env_file=None))
        owned = (
            resource("persona", markdown="Owned instruction.")
            if owned_present
            else None
        )
        service.agent = AsyncMock(return_value=Agent(key="test", name="Lumi"))
        service.agent_persona = AsyncMock(return_value=owned)
        instruction = await service.call_instruction("test")
        assert "Have a spoken conversation." in instruction
        assert "You have no tools." in instruction
        if owned_present:
            assert (
                'You are "Lumi". Talk like a real person in everyday conversation, '
                "not a robotic or overly eager assistant. "
                "Skip canned pleasantries, repetitive openers, and en-dashes or em-dashes. "
                "Treat your persona below as quiet background context; "
                "never re-introduce yourself or bring up your role or appearance unless asked."
                "\nYour saved persona:\nOwned instruction."
            ) in instruction
        else:
            assert (
                'You are "Lumi". Talk like a real person in everyday conversation, '
                "not a robotic or overly eager assistant. "
                "Skip canned pleasantries, repetitive openers, and en-dashes or em-dashes."
            ) in instruction
            assert "Your saved persona:" not in instruction
        assert instruction.count("Owned instruction.") == int(owned_present)
        service.agent.assert_awaited_once_with("test")
        service.agent_persona.assert_awaited_once_with("test")
        if owned:
            owned.markdown = "Later edit."
            assert "Later edit." not in instruction

    asyncio.run(check())


def test_conversational_prompts_avoid_dashes_and_include_style_rules():
    agent_md = Path(Settings(_env_file=None).agent_instruction_file).read_text()
    prompts = (
        agent_md,
        ONBOARDING_GREETING,
        STARTER_PERSONA_MARKDOWN,
        character_persona_section("Lumi", None),
        character_persona_section("Lumi", "Direct and chill."),
    )
    for text in prompts:
        assert "\u2013" not in text
        assert "\u2014" not in text
    assert "## Conversation style" in agent_md
    assert "Never use en-dashes or em-dashes." in agent_md
    assert "I'm happy to be here" in agent_md
    assert "Never re-introduce yourself" in agent_md
    assert "never re-introduce yourself" in character_persona_section(
        "Lumi", "Direct and chill."
    )


def test_call_instruction_preserves_full_text_at_limit_and_rejects_overflow():
    async def check():
        service = LocalServices(None, Settings(_env_file=None))
        owned = resource("persona", markdown="")
        service.agent = AsyncMock(return_value=Agent(key="test", name="Lumi"))
        service.agent_persona = AsyncMock(return_value=owned)
        base = await service.call_instruction("test")
        owned.markdown = "x" * (8000 - len(base))
        instruction = await service.call_instruction("test")
        assert len(instruction) == 8000
        assert owned.markdown in instruction
        owned.markdown += "x"
        with pytest.raises(ServiceError, match="8,000-character limit"):
            await service.call_instruction("test")

    asyncio.run(check())


def test_list_agents_auto_bootstraps_default_agent_conversation_and_starter_persona():
    agents = []
    conversations = []
    persona = None
    requests = []

    def tsuchi(request):
        nonlocal persona
        requests.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == "/api/agents":
            return httpx.Response(200, json=agents)
        if request.method == "POST" and request.url.path == "/api/agents":
            body = json.loads(request.content)
            agent = {
                "key": body["key"],
                "name": body["name"],
                "created_at": NOW.isoformat(),
                "selected_pet_id": None,
                "persona_id": None,
            }
            agents.append(agent)
            return httpx.Response(201, json=agent)
        if (
            request.method == "GET"
            and request.url.path == f"/api/agents/{DEFAULT_AGENT_KEY}/resources"
        ):
            return httpx.Response(
                200, json=[item.model_dump(mode="json") for item in conversations]
            )
        if (
            request.method == "POST"
            and request.url.path == f"/api/agents/{DEFAULT_AGENT_KEY}/resources"
        ):
            body = json.loads(request.content)
            created = resource(
                "conversation",
                agent_key=DEFAULT_AGENT_KEY,
                title=body["title"],
                messages=[],
            )
            conversations.append(created)
            return httpx.Response(201, json=created.model_dump(mode="json"))
        if request.method == "POST" and request.url.path.endswith("/messages"):
            body = json.loads(request.content)
            message = Message.model_validate(body["message"])
            current = conversations[0]
            assert body["expected_version"] == current.version
            conversations[0] = current.model_copy(
                update={
                    "version": current.version + 1,
                    "messages": [*current.messages, message],
                }
            )
            return httpx.Response(200, json=conversations[0].model_dump(mode="json"))
        if (
            request.method == "GET"
            and request.url.path == f"/api/agents/{DEFAULT_AGENT_KEY}/persona"
        ):
            return httpx.Response(
                200,
                content=json.dumps(
                    persona.model_dump(mode="json") if persona is not None else None
                ),
            )
        if (
            request.method == "PUT"
            and request.url.path == f"/api/agents/{DEFAULT_AGENT_KEY}/persona"
        ):
            body = json.loads(request.content)
            assert body["expected_version"] is None
            persona = resource(
                "persona",
                agent_key=DEFAULT_AGENT_KEY,
                title=body["title"],
                markdown=body["markdown"],
            )
            agents[0]["persona_id"] = str(persona.id)
            return httpx.Response(200, json=persona.model_dump(mode="json"))
        raise AssertionError(f"Unexpected call: {request.method} {request.url.path}")

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(tsuchi)) as client:
            service = LocalServices(client, Settings(_env_file=None))
            first = await service.list_agents()
            second = await service.list_agents()
            return first, second

    first, second = asyncio.run(scenario())
    assert len(first) == 1
    assert first[0]["key"] == DEFAULT_AGENT_KEY
    assert first[0]["name"] == DEFAULT_AGENT_NAME
    assert second == first
    assert len(conversations) == 1
    assert len(conversations[0].messages) == 1
    greeting = conversations[0].messages[0]
    assert greeting.id == ONBOARDING_GREETING_MESSAGE_ID
    assert greeting.role == "assistant"
    assert greeting.content == ONBOARDING_GREETING
    assert persona is not None
    assert persona.markdown == STARTER_PERSONA_MARKDOWN
    assert persona.markdown.count("TODO") >= 5
    assert "# Onboarding (remove this section once all TODOs above are filled in)" in persona.markdown
    # Second list_agents call must not re-create or re-seed anything.
    assert requests.count(("POST", "/api/agents")) == 1
    assert requests.count(("PUT", f"/api/agents/{DEFAULT_AGENT_KEY}/persona")) == 1


def test_list_agents_handles_concurrent_bootstrap_race_without_failing():
    agent_record = {
        "key": DEFAULT_AGENT_KEY,
        "name": DEFAULT_AGENT_NAME,
        "created_at": NOW.isoformat(),
        "selected_pet_id": None,
        "persona_id": str(PERSONA_ID),
    }
    created_by_other = False
    existing_conversation = resource(
        "conversation",
        agent_key=DEFAULT_AGENT_KEY,
        title="New chat",
        version=2,
        messages=[
            Message(
                id=ONBOARDING_GREETING_MESSAGE_ID,
                role="assistant",
                content=ONBOARDING_GREETING,
            )
        ],
    )
    existing_persona = resource(
        "persona",
        agent_key=DEFAULT_AGENT_KEY,
        title=DEFAULT_AGENT_NAME,
        markdown=STARTER_PERSONA_MARKDOWN,
    )
    writes = []

    def tsuchi(request):
        nonlocal created_by_other
        if request.method == "GET" and request.url.path == "/api/agents":
            return httpx.Response(200, json=[agent_record] if created_by_other else [])
        if request.method == "POST" and request.url.path == "/api/agents":
            # Another instance created the default agent right before us.
            created_by_other = True
            return httpx.Response(409, json={"detail": "This agent key already exists"})
        if (
            request.method == "GET"
            and request.url.path == f"/api/agents/{DEFAULT_AGENT_KEY}/resources"
        ):
            return httpx.Response(
                200, json=[existing_conversation.model_dump(mode="json")]
            )
        if request.method == "POST" and request.url.path.endswith("/messages"):
            writes.append("message")
            return httpx.Response(409, json={"detail": "This message ID already exists"})
        if (
            request.method == "GET"
            and request.url.path == f"/api/agents/{DEFAULT_AGENT_KEY}/persona"
        ):
            return httpx.Response(200, json=existing_persona.model_dump(mode="json"))
        if (
            request.method == "PUT"
            and request.url.path == f"/api/agents/{DEFAULT_AGENT_KEY}/persona"
        ):
            writes.append("persona")
            return httpx.Response(409, json={"detail": "This agent already has a persona."})
        raise AssertionError(f"Unexpected call: {request.method} {request.url.path}")

    result = run(tsuchi, lambda service: service.list_agents())
    assert result == [agent_record]
    assert writes == []


def test_list_agents_does_not_overwrite_or_reseed_existing_agents():
    existing = [
        {
            "key": "ember",
            "name": "Ember",
            "created_at": NOW.isoformat(),
            "selected_pet_id": None,
            "persona_id": None,
        }
    ]
    requests = []

    def tsuchi(request):
        requests.append((request.method, request.url.path))
        assert request.method == "GET" and request.url.path == "/api/agents"
        return httpx.Response(200, json=existing)

    result = run(tsuchi, lambda service: service.list_agents())
    assert result == existing
    assert requests == [("GET", "/api/agents")]


def test_list_agents_handles_409_races_on_greeting_and_persona_seed():
    created = False
    empty_conversation = resource(
        "conversation",
        agent_key=DEFAULT_AGENT_KEY,
        title="New chat",
        version=1,
        messages=[],
    )
    agent_record = {
        "key": DEFAULT_AGENT_KEY,
        "name": DEFAULT_AGENT_NAME,
        "created_at": NOW.isoformat(),
        "selected_pet_id": None,
        "persona_id": str(PERSONA_ID),
    }

    def tsuchi(request):
        nonlocal created
        if request.method == "GET" and request.url.path == "/api/agents":
            return httpx.Response(200, json=[agent_record] if created else [])
        if request.method == "POST" and request.url.path == "/api/agents":
            created = True
            return httpx.Response(409, json={"detail": "This agent key already exists"})
        if (
            request.method == "GET"
            and request.url.path == f"/api/agents/{DEFAULT_AGENT_KEY}/resources"
        ):
            return httpx.Response(
                200, json=[empty_conversation.model_dump(mode="json")]
            )
        if request.method == "POST" and request.url.path.endswith("/messages"):
            return httpx.Response(
                409, json={"detail": "This message ID already exists"}
            )
        if (
            request.method == "GET"
            and request.url.path == f"/api/agents/{DEFAULT_AGENT_KEY}/persona"
        ):
            return httpx.Response(200, content=b"null")
        if (
            request.method == "PUT"
            and request.url.path == f"/api/agents/{DEFAULT_AGENT_KEY}/persona"
        ):
            return httpx.Response(
                409, json={"detail": "This agent already has a persona."}
            )
        raise AssertionError(f"Unexpected call: {request.method} {request.url.path}")

    result = run(tsuchi, lambda service: service.list_agents())
    assert result == [agent_record]


def test_delete_agent_followed_by_list_agents_bootstraps_fresh_default_agent():
    agents = {}
    conversations = {}
    personas = {}
    generation = 0

    def tsuchi(request):
        nonlocal generation
        if request.method == "GET" and request.url.path == "/api/agents":
            return httpx.Response(200, json=list(agents.values()))
        if request.method == "POST" and request.url.path == "/api/agents":
            body = json.loads(request.content)
            key = body["key"]
            if key in agents:
                return httpx.Response(
                    409, json={"detail": "This agent key already exists"}
                )
            generation += 1
            agent = {
                "key": key,
                "name": body["name"],
                "created_at": NOW.isoformat(),
                "selected_pet_id": None,
                "persona_id": None,
            }
            agents[key] = agent
            return httpx.Response(201, json=agent)
        if request.method == "DELETE" and request.url.path.startswith("/api/agents/"):
            key = request.url.path.removeprefix("/api/agents/")
            if key not in agents:
                return httpx.Response(404, json={"detail": "Agent namespace not found"})
            del agents[key]
            conversations.pop(key, None)
            personas.pop(key, None)
            return httpx.Response(204)
        if (
            request.method == "GET"
            and request.url.path == f"/api/agents/{DEFAULT_AGENT_KEY}/resources"
        ):
            items = conversations.get(DEFAULT_AGENT_KEY, [])
            return httpx.Response(
                200, json=[item.model_dump(mode="json") for item in items]
            )
        if (
            request.method == "POST"
            and request.url.path == f"/api/agents/{DEFAULT_AGENT_KEY}/resources"
        ):
            body = json.loads(request.content)
            conv_id = UUID(f"10000000-0000-4000-8000-{generation:012d}")
            created = resource(
                "conversation",
                id=conv_id,
                agent_key=DEFAULT_AGENT_KEY,
                title=body["title"],
                messages=[],
            )
            conversations.setdefault(DEFAULT_AGENT_KEY, []).append(created)
            return httpx.Response(201, json=created.model_dump(mode="json"))
        if request.method == "POST" and request.url.path.endswith("/messages"):
            body = json.loads(request.content)
            message = Message.model_validate(body["message"])
            current = conversations[DEFAULT_AGENT_KEY][0]
            if any(m.id == message.id for m in current.messages):
                return httpx.Response(
                    409, json={"detail": "This message ID already exists"}
                )
            assert body["expected_version"] == current.version
            updated = current.model_copy(
                update={
                    "version": current.version + 1,
                    "messages": [*current.messages, message],
                }
            )
            conversations[DEFAULT_AGENT_KEY][0] = updated
            return httpx.Response(200, json=updated.model_dump(mode="json"))
        if (
            request.method == "GET"
            and request.url.path == f"/api/agents/{DEFAULT_AGENT_KEY}/persona"
        ):
            current_persona = personas.get(DEFAULT_AGENT_KEY)
            return httpx.Response(
                200,
                content=json.dumps(
                    current_persona.model_dump(mode="json")
                    if current_persona is not None
                    else None
                ),
            )
        if (
            request.method == "PUT"
            and request.url.path == f"/api/agents/{DEFAULT_AGENT_KEY}/persona"
        ):
            body = json.loads(request.content)
            persona_id = UUID(f"20000000-0000-4000-8000-{generation:012d}")
            saved = resource(
                "persona",
                id=persona_id,
                agent_key=DEFAULT_AGENT_KEY,
                title=body["title"],
                markdown=body["markdown"],
            )
            personas[DEFAULT_AGENT_KEY] = saved
            agents[DEFAULT_AGENT_KEY]["persona_id"] = str(persona_id)
            return httpx.Response(200, json=saved.model_dump(mode="json"))
        raise AssertionError(f"Unexpected call: {request.method} {request.url.path}")

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(tsuchi)) as client:
            service = LocalServices(client, Settings(_env_file=None))
            initial = await service.list_agents()
            initial_conv_id = conversations[DEFAULT_AGENT_KEY][0].id
            initial_persona_id = personas[DEFAULT_AGENT_KEY].id

            # Delete the agent via proxy call.
            result = await service.json_request(
                "Tsuchi", "DELETE", f"/api/agents/{DEFAULT_AGENT_KEY}"
            )
            assert result is None
            assert DEFAULT_AGENT_KEY not in agents

            # Deleting a missing agent propagates 404.
            with pytest.raises(ServiceError) as missing:
                await service.json_request(
                    "Tsuchi", "DELETE", f"/api/agents/{DEFAULT_AGENT_KEY}"
                )
            assert missing.value.status == 404

            # Listing agents immediately re-bootstraps a brand-new default agent.
            fresh = await service.list_agents()
            return initial, initial_conv_id, initial_persona_id, fresh

    initial, initial_conv_id, initial_persona_id, fresh = asyncio.run(scenario())
    assert len(initial) == 1
    assert len(fresh) == 1
    assert fresh[0]["key"] == DEFAULT_AGENT_KEY
    assert fresh[0]["name"] == DEFAULT_AGENT_NAME
    assert fresh[0]["persona_id"] != str(initial_persona_id)
    fresh_conv = conversations[DEFAULT_AGENT_KEY][0]
    assert fresh_conv.id != initial_conv_id
    assert len(fresh_conv.messages) == 1
    assert fresh_conv.messages[0].id == ONBOARDING_GREETING_MESSAGE_ID
    assert fresh_conv.messages[0].content == ONBOARDING_GREETING
    assert personas[DEFAULT_AGENT_KEY].markdown == STARTER_PERSONA_MARKDOWN


def test_inference_status_detects_downloaded_abliterated_companion():
    settings = ready_models()
    settings["models"].append(
        {
            **settings["models"][0],
            "id": "inkling-2-mini-abliterated-iq4-nl",
            "status": "complete",
        }
    )
    status = run(
        Boundary(settings=settings), lambda service: service.inference_status()
    )
    assert status.state == "ready"
    assert status.model_id == "inkling-2-mini-iq4-nl"
    assert status.abliterated_available is True

    base_only = ready_models()
    status_base = run(
        Boundary(settings=base_only), lambda service: service.inference_status()
    )
    assert status_base.abliterated_available is False
