import asyncio

import pytest
from pydantic import ValidationError

from kadan.api.routes.agents.conversations.chat import ChatInput, ChatResponse


@pytest.mark.parametrize(
    "body",
    [
        {"content": "Hi", "expected_version": 1},
        {"content": " ", "expected_version": 1, "persona_id": None},
        {
            "content": "Hi",
            "expected_version": 1,
            "persona_id": None,
            "tools": [{"type": "shell"}],
        },
    ],
)
def test_chat_requests_reject_unsupported_or_ambiguous_input(body):
    with pytest.raises(ValidationError):
        ChatInput.model_validate(body)


def test_disconnect_cancels_silent_tool_work_on_asgi_24():
    async def run():
        started = asyncio.Event()
        closed = asyncio.Event()

        async def events():
            try:
                yield "event: status\ndata: {}\n\n"
                started.set()
                await asyncio.Event().wait()
            finally:
                closed.set()

        async def receive():
            await started.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        response = ChatResponse(events(), media_type="text/event-stream")
        await asyncio.wait_for(
            response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send),
            timeout=1,
        )
        assert closed.is_set()

    asyncio.run(run())
