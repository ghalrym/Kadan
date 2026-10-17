import asyncio
import json
from copy import deepcopy
from datetime import UTC, datetime
from uuid import UUID

import httpx
import pytest

from kadan.api.config import ServiceError, Settings
from kadan.api.pydantic_models.resources import Message, Resource
from kadan.api.routes.agents.conversations.chat import ChatInput
from kadan.api.routes.agents.conversations.notes import NoteInput
from kadan.api.routes.agents.resources import PersonaUpdate
from kadan.api.services import LocalServices, completion_messages, sse_data

CONVERSATION_ID = UUID("10000000-0000-4000-8000-000000000001")
PERSONA_ID = UUID("10000000-0000-4000-8000-000000000002")
MESSAGE_ID = UUID("10000000-0000-4000-8000-000000000003")
NOW = datetime(2026, 10, 17, tzinfo=UTC)


def resource(kind="conversation", messages=None, **changes):
    return Resource(
        id=PERSONA_ID if kind == "persona" else CONVERSATION_ID,
        agent_key="test",
        kind=kind,
        title="Test",
        version=1,
        created_at=NOW,
        updated_at=NOW,
        messages=messages or [],
        **changes,
    )


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


def chunk(text=None, finish=None):
    delta = {"content": text} if text is not None else {}
    return (
        "data: "
        + json.dumps(
            {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        )
        + "\n\n"
    )


SUCCESS = (
    chunk("")
    + chunk("Hello ")
    + chunk("there")
    + chunk(finish="stop")
    + "data: [DONE]\n\n"
)


class Boundary:
    def __init__(self, stream=SUCCESS, settings=None):
        self.current = resource()
        self.persona = resource("persona", markdown="Be precise.")
        self.settings = settings if settings is not None else ready_settings()
        self.stream = stream
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
        if request.url.path == "/v1/chat/completions":
            self.inference_body = json.loads(request.content)
            if self.inference_status != 200:
                return httpx.Response(
                    self.inference_status, json={"detail": "Model became unavailable"}
                )
            if isinstance(self.stream, httpx.AsyncByteStream):
                return httpx.Response(
                    200,
                    stream=self.stream,
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(
                200, text=self.stream, headers={"content-type": "text/event-stream"}
            )
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
        return await operation(
            LocalServices(client, Settings(_env_file=None, plugins=[]))
        )


def run(boundary, operation):
    return asyncio.run(with_services(boundary, operation))


async def generate(service, persona_id=PERSONA_ID):
    value = ChatInput(
        content="Hi", expected_version=1, message_id=MESSAGE_ID, persona_id=persona_id
    )
    prepared = await service.prepare_chat("test", CONVERSATION_ID, value)
    saved = await service.save_chat_user("test", prepared)
    frames = [frame async for frame in service.chat_events("test", prepared, saved)]
    return [
        (
            frame.splitlines()[0].removeprefix("event: "),
            json.loads(frame.splitlines()[1].removeprefix("data: ")),
        )
        for frame in frames
    ]


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


def test_chat_preflight_does_not_save_or_infer_without_tier():
    boundary = Boundary(settings={"tier": None, "tiers": []})
    with pytest.raises(ServiceError, match="Choose a tier") as raised:
        run(boundary, generate)
    assert raised.value.status == 503
    assert boundary.appends == []
    assert boundary.inference_body is None


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


def test_stream_persists_user_then_successful_assistant_and_replays_persona():
    boundary = Boundary()
    frames = run(boundary, generate)
    assert [name for name, _ in frames] == ["user_saved", "delta", "delta", "done"]
    assert [entry["message"]["role"] for entry in boundary.appends] == [
        "user",
        "assistant",
    ]
    assert [entry["expected_version"] for entry in boundary.appends] == [1, 2]
    assert boundary.current.messages[-1].content == "Hello there"
    assert boundary.current.messages[0].resource_ids == [PERSONA_ID]
    assert boundary.inference_body == {
        "messages": [
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Hi"},
        ],
        "model": "test-model",
        "stream": True,
        "conversation_id": f"kadan:test:{CONVERSATION_ID}",
    }
    assert frames[-1][1]["conversation"]["version"] == 3


def test_length_terminal_event_is_success_and_exposes_finish_reason():
    boundary = Boundary(
        stream=chunk("Truncated answer") + chunk(finish="length") + "data: [DONE]\n\n"
    )
    frames = run(boundary, generate)
    assert frames[-1][0] == "done"
    assert frames[-1][1]["finish_reason"] == "length"
    assert boundary.current.messages[-1].content == "Truncated answer"


@pytest.mark.parametrize(
    "suffix",
    [
        "",
        "data: [DONE]\n\n",
        chunk(finish="stop"),
        'data: {"error":{"message":"GPU is unavailable"}}\n\ndata: [DONE]\n\n',
        "data: not-json\n\n",
        chunk(finish="tool_calls") + "data: [DONE]\n\n",
        chunk(finish="stop") + chunk("after finish") + "data: [DONE]\n\n",
    ],
)
def test_incomplete_or_failed_stream_keeps_partial_visible_and_never_saves_assistant(
    suffix,
):
    boundary = Boundary(stream=chunk("Partial") + suffix)
    frames = run(boundary, generate)
    assert frames[1] == ("delta", {"content": "Partial"})
    assert frames[-1][0] == "error"
    assert frames[-1][1]["user_saved"] is True
    assert [message.role for message in boundary.current.messages] == ["user"]
    assert all(name != "done" for name, _ in frames)


def test_save_conflict_after_inference_reports_reply_unsaved():
    boundary = Boundary()
    boundary.assistant_conflict = True
    frames = run(boundary, generate)
    assert frames[-1][0] == "error"
    assert frames[-1][1]["code"] == "version_conflict"
    assert "this reply was not saved" in frames[-1][1]["message"]
    assert len(boundary.current.messages) == 1


def test_inference_http_error_after_user_saved_does_not_save_assistant():
    boundary = Boundary()
    boundary.inference_status = 503
    frames = run(boundary, generate)
    assert frames[-1][1]["message"] == "Model became unavailable"
    assert len(boundary.appends) == 1


def test_stale_conversation_version_rejected_before_any_write_or_inference():
    boundary = Boundary()
    boundary.current = boundary.current.model_copy(update={"version": 2})
    with pytest.raises(ServiceError) as raised:
        run(boundary, generate)
    assert raised.value.code == "version_conflict"
    assert boundary.appends == []
    assert boundary.inference_body is None


def test_race_at_user_save_stops_before_generation():
    boundary = Boundary()
    boundary.note_conflict = True
    with pytest.raises(ServiceError) as raised:
        run(boundary, generate)
    assert raised.value.status == 409
    assert boundary.inference_body is None


def test_duplicate_user_id_cannot_start_second_generation():
    boundary = Boundary()
    boundary.current = resource(
        messages=[Message(id=MESSAGE_ID, role="user", content="Already saved")]
    )
    with pytest.raises(ServiceError) as raised:
        run(boundary, generate)
    assert raised.value.code == "duplicate_message"
    assert boundary.appends == []


def test_no_persona_is_an_explicit_request_and_replays_existing_text():
    boundary = Boundary()
    boundary.current = resource(
        messages=[
            Message(role="user", content="Earlier"),
            Message(role="assistant", content="Answer"),
        ]
    )
    run(boundary, lambda service: generate(service, persona_id=None))
    assert boundary.inference_body["messages"] == [
        {"role": "user", "content": "Earlier"},
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "Hi"},
    ]
    assert not any(path.endswith(str(PERSONA_ID)) for _, path in boundary.requests)


@pytest.mark.parametrize(
    "messages",
    [
        [Message(role="assistant", content="")],
        [Message(role="user", content="x" * 48000)],
        [Message(role="user", content="x") for _ in range(100)],
    ],
)
def test_unsupported_history_is_rejected_without_truncating(messages):
    current = resource(messages=messages)
    before = deepcopy(current)
    with pytest.raises(ServiceError):
        completion_messages(current, None, Message(role="user", content="Next"))
    assert current == before


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


def test_sse_parser_handles_multiline_data_comments_and_unterminated_frame():
    async def lines():
        for line in [
            ": keepalive",
            "event: message",
            "data: one",
            "data: two",
            "",
            "data: incomplete",
        ]:
            yield line

    async def collect():
        return [value async for value in sse_data(lines())]

    assert asyncio.run(collect()) == ["one\ntwo"]


class InterruptedStream(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        yield chunk("Partial").encode()
        raise httpx.ReadError("Connection lost")

    async def aclose(self):
        self.closed = True


def test_network_failure_mid_stream_is_visible_and_closes_upstream():
    stream = InterruptedStream()
    boundary = Boundary(stream=stream)
    frames = run(boundary, generate)
    assert frames[-1][1]["code"] == "inference_unavailable"
    assert len(boundary.current.messages) == 1
    assert stream.closed


def test_client_cancellation_closes_upstream_and_keeps_only_user_message():
    stream = InterruptedStream()
    boundary = Boundary(stream=stream)

    async def cancel(service):
        prepared = await service.prepare_chat(
            "test",
            CONVERSATION_ID,
            ChatInput(
                content="Hi", expected_version=1, message_id=MESSAGE_ID, persona_id=None
            ),
        )
        saved = await service.save_chat_user("test", prepared)
        events = service.chat_events("test", prepared, saved)
        assert (await anext(events)).startswith("event: user_saved")
        assert (await anext(events)).startswith("event: delta")
        await events.aclose()

    run(boundary, cancel)
    assert len(boundary.current.messages) == 1
    assert stream.closed


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
