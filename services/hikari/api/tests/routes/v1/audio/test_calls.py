import unittest
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
        for extra in ({"tools": []}, {"backend": "native"}, {"backend": "lfm3"}):
            with self.subTest(extra=extra), self.assertRaises(ValidationError):
                CallStart(type="start", **extra)


class CallRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_call_needs_no_flag_and_closes_socket_after_cleanup(self):
        events = []

        async def run(receive):
            events.append("cleaned")
            raise WebSocketDisconnect()

        socket = AsyncMock()
        socket.receive_json.return_value = {"type": "start"}
        socket.close.side_effect = lambda: events.append("closed")
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("api.routes.v1.audio.calls.inference") as service,
            patch("api.routes.v1.audio.calls.VoiceCall") as call,
        ):
            service.models_status.return_value = snapshot()
            call.return_value.run.side_effect = run
            await voice_call(socket)
        self.assertEqual(call.call_args.args[1].audio, "lfm3-audio-2b-bf16")
        self.assertEqual(events, ["cleaned", "closed"])
        socket.close.assert_awaited_once()

    async def test_missing_models_report_error_without_starting_call(self):
        socket = AsyncMock()
        socket.receive_json.return_value = {"type": "start"}
        status = snapshot()
        status.models[0].status = "not_downloaded"
        with (
            patch("api.routes.v1.audio.calls.inference") as service,
            patch("api.routes.v1.audio.calls.VoiceCall") as call,
        ):
            service.models_status.return_value = status
            await voice_call(socket)
        call.assert_not_called()
        self.assertEqual(socket.send_json.call_args.args[0]["type"], "error")
        socket.close.assert_awaited_once()
