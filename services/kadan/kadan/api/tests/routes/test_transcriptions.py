import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from kadan.api.config import ServiceError
from kadan.api.routes.transcriptions import TranscriptionInput, transcribe


def test_transcription_uses_existing_hikari_endpoint_without_enqueuing_a_chat():
    services = SimpleNamespace(
        json_request=AsyncMock(return_value={"text": "Hello agent", "model": "whisper"})
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(services=services))
    )
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
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(services=services))
    )
    with pytest.raises(ServiceError, match="Install a transcription model") as error:
        asyncio.run(
            transcribe(TranscriptionInput(audio="data:audio/wav;base64,AAAA"), request)
        )
    assert error.value.status == 503
