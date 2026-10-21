import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from kadan.api.config import ServiceError
from kadan.api.routes.transcriptions import TranscriptionInput, transcribe


async def connected():
    await asyncio.Future()


def request_for(services, receive=connected):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(services=services)),
        receive=AsyncMock(side_effect=receive),
    )


def test_transcription_uses_existing_hikari_endpoint_without_enqueuing_a_chat():
    services = SimpleNamespace(
        json_request=AsyncMock(return_value={"text": "Hello agent", "model": "whisper"})
    )
    request = request_for(services)
    result = asyncio.run(
        transcribe(TranscriptionInput(audio="data:audio/wav;base64,AAAA"), request)
    )
    assert result.text == "Hello agent"
    args, kwargs = services.json_request.call_args
    assert args == ("Hikari", "POST", "/v1/audio/transcriptions")
    assert kwargs["json"] == {
        "audio": "data:audio/wav;base64,AAAA",
        "formatting": False,
    }
    assert kwargs["timeout"] == 120


@pytest.mark.parametrize(
    "audio",
    ["", "https://example.com/audio.wav", "data:audio/wav;base64,", "x" * 900001],
)
def test_transcription_rejects_missing_wav_transport_and_oversized_input(audio):
    with pytest.raises(ValidationError):
        TranscriptionInput(audio=audio)


def test_transcription_preserves_upstream_unavailability():
    services = SimpleNamespace(
        json_request=AsyncMock(
            side_effect=ServiceError("Install a transcription model", 503)
        )
    )
    request = request_for(services)
    with pytest.raises(ServiceError, match="Install a transcription model") as error:
        asyncio.run(
            transcribe(TranscriptionInput(audio="data:audio/wav;base64,AAAA"), request)
        )
    assert error.value.status == 503


def test_browser_disconnect_cancels_upstream_and_waits_for_connection_cleanup():
    async def check():
        started = asyncio.Event()
        cleaned = asyncio.Event()
        incoming = asyncio.Queue()
        incoming.put_nowait({"type": "http.request", "body": b""})

        async def upstream(*args, **kwargs):
            started.set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                cleaned.set()

        request = request_for(SimpleNamespace(json_request=upstream), incoming.get)
        task = asyncio.create_task(
            transcribe(TranscriptionInput(audio="data:audio/wav;base64,AAAA"), request)
        )
        await started.wait()
        incoming.put_nowait({"type": "http.disconnect"})
        with pytest.raises(HTTPException) as error:
            await asyncio.wait_for(task, timeout=1)
        assert error.value.status_code == 499
        assert cleaned.is_set()

    asyncio.run(check())


def test_handler_cancellation_closes_upstream_and_its_disconnect_watcher():
    async def check():
        started = asyncio.Event()
        watching = asyncio.Event()
        upstream_cleaned = asyncio.Event()
        watcher_cleaned = asyncio.Event()

        async def upstream(*args, **kwargs):
            started.set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                upstream_cleaned.set()

        async def receive():
            watching.set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                watcher_cleaned.set()

        request = request_for(SimpleNamespace(json_request=upstream), receive)
        task = asyncio.create_task(
            transcribe(TranscriptionInput(audio="data:audio/wav;base64,AAAA"), request)
        )
        await started.wait()
        await asyncio.wait_for(watching.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert upstream_cleaned.is_set()
        assert watcher_cleaned.is_set()

    asyncio.run(check())
