import unittest
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

from fastapi import WebSocketDisconnect
from pydantic import ValidationError

from api.routes.v1.audio.calls import CallStart, call_event, voice_call
from api.tests.services.test_voice_call import snapshot


class CallMessageTests(unittest.TestCase):
    def test_requires_bounded_audio_and_known_messages_without_tools(self):
        for value in (
            {"type": "utterance", "turn": 1, "audio": "https://audio"},
            {
                "type": "utterance",
                "turn": 1,
                "audio": "data:audio/wav;base64," + "x" * 900000,
            },
            {"type": "interrupt", "turn": 0},
            {"type": "tools", "tools": []},
        ):
            with (
                self.subTest(value_type=value["type"]),
                self.assertRaises(ValidationError),
            ):
                call_event.validate_python(value)
        with self.assertRaises(ValidationError):
            CallStart(type="start", tools=[])


class CallRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_releases_residency_after_call_cleanup(self):
        events = []

        @contextmanager
        def residency(models):
            self.assertEqual(models, frozenset(("lfm3-audio-2b-bf16",)))
            events.append("start")
            try:
                yield
            finally:
                events.append("released")

        async def run(receive):
            events.append("cleaned")
            raise WebSocketDisconnect()

        socket = AsyncMock()
        socket.receive_json.return_value = {"type": "start"}
        with (
            patch("api.routes.v1.audio.calls.inference") as service,
            patch("api.routes.v1.audio.calls.VoiceCall") as call,
        ):
            service.models_status.return_value = snapshot()
            service.worker.full_duplex.side_effect = residency
            call.return_value.run.side_effect = run
            await voice_call(socket)
        self.assertEqual(events, ["start", "cleaned", "released"])
        socket.close.assert_awaited_once()

    async def test_missing_models_report_error_without_acquiring_residency(self):
        socket = AsyncMock()
        socket.receive_json.return_value = {"type": "start"}
        status = snapshot()
        status.models[0].status = "not_downloaded"
        with patch("api.routes.v1.audio.calls.inference") as service:
            service.models_status.return_value = status
            await voice_call(socket)
        service.worker.full_duplex.assert_not_called()
        self.assertEqual(socket.send_json.call_args.args[0]["type"], "error")
        socket.close.assert_awaited_once()
