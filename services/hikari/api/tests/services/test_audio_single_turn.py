import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from experimental_worker.call_history import append_call_history
from pydantic import ValidationError

from api.pydantic_models.inference_task import AudioCallTask, InferenceTask
from api.services.inference import InferenceService
from api.services.inference_errors import InferenceFailure
from api.services.inference_worker import InferenceJob, WorkerEvent
from api.services.voice_call import CallModels, VoiceCall
from api.services.worker_hub import WorkerHub
from api.tests.services.test_experimental_audio_call import audio_snapshot


def task():
    return AudioCallTask(
        mode="turn",
        audio="wav",
        model_id="lfm3-audio-2b-bf16",
        messages=[{"role": "system", "text": "Persona"}],
    )


class TurnStream:
    def __init__(self, *, error=None, wait=False):
        self.error, self.wait = error, wait
        self.transcribed = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        yield {"type": "transcript", "text": "Hi"}
        self.transcribed.set()
        if self.wait:
            await asyncio.Future()
        if self.error:
            raise self.error
        yield {
            "type": "reply",
            "transcript": "Hi",
            "text": "Hello",
            "audio_base64": "AAAA",
        }

    async def aclose(self):
        await asyncio.sleep(0)
        self.closed = True


class SingleTurnCallTests(unittest.IsolatedAsyncioTestCase):
    def call(self, stream):
        service = SimpleNamespace(
            models_status=audio_snapshot,
            open_audio_turn=AsyncMock(return_value=stream),
            submit=AsyncMock(),
        )
        call = VoiceCall(
            service,
            CallModels.selected(audio_snapshot()),
            "Persona",
            AsyncMock(),
        )
        call.turn = 1
        return call

    async def test_single_turn_preserves_transcript_before_buffered_reply(self):
        stream = TurnStream()
        call = self.call(stream)
        with patch.dict("os.environ", {"HIKARI_LFM3_AUDIO_SINGLE_TURN": "true"}):
            await call.respond("wav", 1)
        submitted = call.inference.open_audio_turn.call_args.args[0]
        self.assertEqual(
            submitted.model_dump(exclude_unset=True)["model_id"], "lfm3-audio-2b-bf16"
        )
        self.assertEqual(submitted.mode, "turn")
        call.inference.submit.assert_not_awaited()
        self.assertEqual([message.text for message in call.history], ["Hi"])
        self.assertEqual(
            [entry.args[0]["type"] for entry in call.send.call_args_list],
            ["transcript", "text", "audio", "done"],
        )
        self.assertTrue(stream.closed)

    async def test_transcript_stays_in_history_when_reply_generation_fails(self):
        stream = TurnStream(error=InferenceFailure("incomplete EOS", 502))
        call = self.call(stream)
        with (
            patch.dict("os.environ", {"HIKARI_LFM3_AUDIO_SINGLE_TURN": "true"}),
            self.assertRaisesRegex(InferenceFailure, "incomplete EOS"),
        ):
            await call.respond("wav", 1)
        self.assertEqual([message.text for message in call.history], ["Hi"])
        self.assertEqual(call.segments, [])
        self.assertEqual(
            [entry.args[0]["type"] for entry in call.send.call_args_list],
            ["transcript"],
        )
        self.assertTrue(stream.closed)

    async def test_interrupt_retains_transcript_and_joins_cleanup_before_ack(self):
        stream = TurnStream(wait=True)
        call = self.call(stream)
        with patch.dict("os.environ", {"HIKARI_LFM3_AUDIO_SINGLE_TURN": "true"}):
            call.response = asyncio.create_task(call.respond("wav", 1))
            await stream.transcribed.wait()
            await call.interrupt(2)
        self.assertTrue(stream.closed)
        self.assertEqual([message.text for message in call.history], ["Hi"])
        self.assertEqual(
            [entry.args[0]["type"] for entry in call.send.call_args_list],
            ["transcript", "interrupted"],
        )
        self.assertEqual(call.segments, [])

    async def test_flag_off_uses_the_existing_two_job_path(self):
        call = self.call(TurnStream())
        call.inference.submit.side_effect = [
            {"text": "Hi"},
            {"text": "Hello", "audio_base64": "AAAA"},
        ]
        with patch.dict("os.environ", {"HIKARI_LFM3_AUDIO_SINGLE_TURN": "false"}):
            await call.respond("wav", 1)
        call.inference.open_audio_turn.assert_not_awaited()
        self.assertEqual(
            [entry.args[0].mode for entry in call.inference.submit.call_args_list],
            ["transcribe", "reply"],
        )

    def test_shared_history_keeps_exact_existing_call_bounds(self):
        call = self.call(TurnStream())
        raw = []
        for text in [
            "a" * 4001,
            "b" * 4000,
            "c" * 4000,
            "d",
            *[str(i) for i in range(15)],
            "   ",
        ]:
            raw = append_call_history(raw, "user", text, model_bounded=True)
            call.remember("user", text)
            self.assertEqual(
                [message.model_dump(exclude_none=True) for message in call.history], raw
            )
            self.assertLessEqual(len(raw), 12)
        self.assertEqual(raw[-1]["text"], "14")


class AudioTurnTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_hub_routes_typed_transcript_without_chat_content_encoding(self):
        hub = WorkerHub()
        job = InferenceJob(InferenceTask(input=task()), streaming=True)
        job.machines.add("worker")
        hub.jobs[job.request.id] = job
        worker = SimpleNamespace(id="worker", name="audio", kind="lfm3_audio")
        hub._job_event(
            worker,
            WorkerEvent(id=job.request.id, state="audio_transcript", transcript="Hi"),
        )
        self.assertEqual(job.events.get_nowait(), {"type": "transcript", "text": "Hi"})
        self.assertEqual(job.state, "queued")

    async def test_service_stream_preserves_transcript_then_complete_result(self):
        service = InferenceService()
        job = InferenceJob(InferenceTask(input=task()), streaming=True)
        job.events.put_nowait({"type": "transcript", "text": "Hi"})
        completion = asyncio.get_running_loop().create_future()
        completion.set_result(
            {"transcript": "Hi", "text": "Hello", "audio_base64": "AAAA"}
        )
        with (
            patch.object(
                service, "_submit", new=AsyncMock(return_value=(job, completion))
            ) as submit,
            patch.object(service, "cancel", new=AsyncMock()) as cancel,
        ):
            stream = await service.open_audio_turn(task())
            events = [event async for event in stream]
            await stream.aclose()
        self.assertTrue(submit.call_args.kwargs["streaming"])
        self.assertEqual(events[0], {"type": "transcript", "text": "Hi"})
        self.assertEqual(events[1]["type"], "reply")
        cancel.assert_awaited_once_with(job.request.id)

    def test_worker_transcript_has_a_separate_bounded_typed_field(self):
        for fields in ({"state": "audio_transcript"},):
            with self.assertRaises(ValidationError):
                WorkerEvent(**fields)
        self.assertEqual(
            WorkerEvent(state="audio_transcript", transcript="").transcript, ""
        )
        self.assertEqual(
            len(
                WorkerEvent(state="audio_transcript", transcript="x" * 5000).transcript
            ),
            5000,
        )

    def test_complete_turn_requires_both_utterance_and_existing_text_context(self):
        for fields in (
            {"mode": "turn", "audio": "wav"},
            {"mode": "turn", "messages": [{"role": "system", "text": "Persona"}]},
        ):
            with self.assertRaises(ValidationError):
                AudioCallTask(**fields)


class CodecProbeSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_api_probe_flag_only_changes_the_complete_turn_task(self):
        for value, enabled in (("false", False), ("true", True)):
            stream = TurnStream()
            call = SingleTurnCallTests().call(stream)
            with patch.dict(
                "os.environ",
                {
                    "HIKARI_LFM3_AUDIO_SINGLE_TURN": "true",
                    "HIKARI_LFM3_AUDIO_STREAM_PROBE": value,
                },
            ):
                await call.respond("wav", 1)
            submitted = call.inference.open_audio_turn.call_args.args[0]
            self.assertIs(submitted.codec_streaming_probe, enabled)

    def test_probe_is_a_strict_boolean_and_only_enabled_for_complete_turns(self):
        for fields in (
            {"mode": "transcribe", "audio": "wav", "codec_streaming_probe": True},
            {
                "mode": "reply",
                "messages": [{"role": "system", "text": "Persona"}],
                "codec_streaming_probe": True,
            },
            {
                "mode": "turn",
                "audio": "wav",
                "messages": [{"role": "system", "text": "Persona"}],
                "codec_streaming_probe": "false",
            },
        ):
            with self.assertRaises(ValidationError):
                AudioCallTask(**fields)
