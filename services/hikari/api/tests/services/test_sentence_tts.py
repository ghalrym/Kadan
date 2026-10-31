import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from api.pydantic_models.inference_task import AudioCallTask, InferenceTask
from api.routes.v1.audio.calls import call_event
from api.services.inference_errors import InferenceFailure
from api.services.inference_worker import InferenceJob, WorkerEvent
from api.services.voice_call import CallModels, VoiceCall
from api.services.worker_hub import WorkerHub
from api.tests.services.test_experimental_audio_call import audio_snapshot

FLAGS = {
    "HIKARI_LFM3_AUDIO_SINGLE_TURN": "true",
    "HIKARI_LFM3_AUDIO_SENTENCE_TTS": "true",
    "HIKARI_LFM3_AUDIO_STREAM_PROBE": "false",
}


class SentenceStream:
    def __init__(
        self,
        *,
        wait=False,
        failure=False,
        receipt=None,
        before_first_failure=False,
        segment_override=None,
    ):
        self.wait, self.failure, self.receipt = wait, failure, receipt
        self.before_first_failure = before_first_failure
        self.segment_override = segment_override or {}
        self.first_segment = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        yield {"type": "transcript", "text": "Hi"}
        yield {"type": "answer", "text": "First. Second."}
        if self.before_first_failure:
            raise InferenceFailure("First TTS failed.", 502)
        yield {
            "type": "segment",
            "index": 0,
            "script": "First.",
            "audio_base64": "AAAA",
            **self.segment_override,
        }
        self.first_segment.set()
        if self.wait:
            await asyncio.Future()
        if self.failure:
            raise InferenceFailure("Later segment failed.", 502)
        yield {
            "type": "segment",
            "index": 1,
            "script": "Second.",
            "audio_base64": "BBBB",
        }
        yield {
            "type": "reply",
            "transcript": "Hi",
            "text": "First. Second.",
            "scripts": ["First.", "Second."],
            "segment_count": 2,
            **(self.receipt or {}),
        }

    async def aclose(self):
        await asyncio.sleep(0)
        self.closed = True


def call(stream):
    return VoiceCall(
        SimpleNamespace(
            models_status=audio_snapshot,
            open_audio_turn=AsyncMock(return_value=stream),
            submit=AsyncMock(),
        ),
        CallModels.selected(audio_snapshot()),
        "Saved persona",
        AsyncMock(),
    )


class SentenceCallTests(unittest.IsolatedAsyncioTestCase):
    async def test_segments_are_contiguous_and_done_waits_for_final_receipt(self):
        stream = SentenceStream()
        value = call(stream)
        value.turn = 1
        with patch.dict("os.environ", FLAGS):
            await value.respond("wav", 1)
        self.assertTrue(value.inference.open_audio_turn.call_args.args[0].sentence_tts)
        self.assertEqual(value.segments, ["First.", "Second."])
        self.assertEqual([message.text for message in value.history], ["Hi"])
        events = [entry.args[0] for entry in value.send.call_args_list]
        self.assertEqual(
            [event["segment"] for event in events if event["type"] == "audio"], [0, 1]
        )
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(
            "".join(event["text"] for event in events if event["type"] == "text"),
            "First. Second.",
        )
        self.assertEqual(
            [event["type"] for event in events[:3]], ["transcript", "text", "audio"]
        )
        self.assertTrue(stream.closed)

    async def test_validated_answer_is_not_displayed_before_first_completed_segment(
        self,
    ):
        value = call(SentenceStream(before_first_failure=True))
        value.turn = 1
        with (
            patch.dict("os.environ", FLAGS),
            self.assertRaisesRegex(InferenceFailure, "First TTS"),
        ):
            await value.respond("wav", 1)
        self.assertEqual(
            [entry.args[0]["type"] for entry in value.send.call_args_list],
            ["transcript"],
        )
        self.assertEqual(value.segments, [])

    async def test_wrong_script_or_noncontiguous_index_cannot_publish_first_segment(
        self,
    ):
        for override in ({"index": 1}, {"script": "Different."}):
            value = call(SentenceStream(segment_override=override))
            value.turn = 1
            with (
                self.subTest(override=override),
                patch.dict("os.environ", FLAGS),
                self.assertRaisesRegex(InferenceFailure, "known-script"),
            ):
                await value.respond("wav", 1)
            self.assertEqual(
                [entry.args[0]["type"] for entry in value.send.call_args_list],
                ["transcript"],
            )
            self.assertEqual(value.segments, [])

    async def test_later_failure_leaves_completed_prefix_without_done_or_unheard_history(
        self,
    ):
        stream = SentenceStream(failure=True)
        value = call(stream)
        value.turn = 1
        with (
            patch.dict("os.environ", FLAGS),
            self.assertRaisesRegex(InferenceFailure, "Later segment"),
        ):
            await value.respond("wav", 1)
        self.assertEqual(value.segments, ["First."])
        self.assertEqual([message.text for message in value.history], ["Hi"])
        self.assertNotIn(
            "done", [entry.args[0]["type"] for entry in value.send.call_args_list]
        )
        self.assertTrue(stream.closed)

    async def test_real_played_message_and_interrupt_retain_only_fully_played_script(
        self,
    ):
        stream = SentenceStream(wait=True)
        value = call(stream)
        inbox = asyncio.Queue()
        interrupted = asyncio.Event()

        async def send(event):
            if event["type"] == "interrupted" and event["turn"] == 2:
                self.assertTrue(stream.closed)
                interrupted.set()

        value.send = AsyncMock(side_effect=send)
        with patch.dict("os.environ", FLAGS):
            running = asyncio.create_task(value.run(inbox.get))
            inbox.put_nowait(
                call_event.validate_python({"type": "interrupt", "turn": 1})
            )
            inbox.put_nowait(
                call_event.validate_python(
                    {
                        "type": "utterance",
                        "turn": 1,
                        "audio": "data:audio/wav;base64,AAAA",
                    }
                )
            )
            await asyncio.wait_for(stream.first_segment.wait(), 1)
            inbox.put_nowait(
                call_event.validate_python({"type": "played", "turn": 1, "segment": 0})
            )
            inbox.put_nowait(
                call_event.validate_python({"type": "interrupt", "turn": 2})
            )
            await asyncio.wait_for(interrupted.wait(), 1)
            inbox.put_nowait(call_event.validate_python({"type": "end"}))
            await asyncio.wait_for(running, 1)
        self.assertEqual(
            [(message.role, message.text) for message in value.history],
            [("user", "Hi"), ("assistant", "First.")],
        )
        self.assertEqual(value.segments, [])

    async def test_final_receipt_mismatch_fails_without_done(self):
        for receipt in (
            {"scripts": ["Different."]},
            {"text": "Different answer"},
            {"segment_count": 1},
            {"segment_count": True},
        ):
            value = call(SentenceStream(receipt=receipt))
            value.turn = 1
            with (
                self.subTest(receipt=receipt),
                patch.dict("os.environ", FLAGS),
                self.assertRaisesRegex(InferenceFailure, "receipt"),
            ):
                await value.respond("wav", 1)
            self.assertNotIn(
                "done", [entry.args[0]["type"] for entry in value.send.call_args_list]
            )

    async def test_sentence_tts_requires_single_turn_and_excludes_private_probe(self):
        for override in (
            {"HIKARI_LFM3_AUDIO_SINGLE_TURN": "false"},
            {"HIKARI_LFM3_AUDIO_STREAM_PROBE": "true"},
        ):
            value = call(SentenceStream())
            with (
                patch.dict("os.environ", {**FLAGS, **override}),
                self.assertRaises(InferenceFailure),
            ):
                await value.respond("wav", 1)
            value.inference.open_audio_turn.assert_not_awaited()
            value.inference.submit.assert_not_awaited()


class SentenceEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_hub_routes_typed_events_only_for_opted_in_turn(self):
        hub = WorkerHub()
        body = AudioCallTask(
            mode="turn",
            audio="wav",
            sentence_tts=True,
            messages=[{"role": "system", "text": "Persona"}],
        )
        job = InferenceJob(InferenceTask(input=body), streaming=True)
        hub.jobs[job.request.id] = job
        job.machines.add("worker")
        worker = SimpleNamespace(id="worker", kind="lfm3_audio", name="audio")
        hub._job_event(
            worker,
            WorkerEvent(id=job.request.id, state="audio_answer", answer="First."),
        )
        hub._job_event(
            worker,
            WorkerEvent(
                id=job.request.id,
                state="audio_segment",
                audio_segment={"index": 0, "script": "First.", "audio_base64": "AAAA"},
            ),
        )
        self.assertEqual(job.events.get_nowait(), {"type": "answer", "text": "First."})
        self.assertEqual(job.events.get_nowait()["script"], "First.")
        job.request.input.sentence_tts = False
        with patch.object(hub, "cancel", new=AsyncMock()) as cancel:
            hub._job_event(
                worker,
                WorkerEvent(id=job.request.id, state="audio_answer", answer="First."),
            )
            await asyncio.sleep(0)
        cancel.assert_awaited_once_with(job.request.id)
        self.assertTrue(job.events.empty())

    def test_typed_fields_and_strict_flag_reject_incomplete_or_incompatible_inputs(
        self,
    ):
        for fields in (
            {"state": "audio_answer"},
            {"state": "audio_segment"},
            {
                "state": "audio_segment",
                "audio_segment": {
                    "index": 0,
                    "script": "x" * 241,
                    "audio_base64": "AAAA",
                },
            },
        ):
            with self.assertRaises(ValidationError):
                WorkerEvent(**fields)
        self.assertFalse(
            AudioCallTask(
                mode="turn",
                audio="wav",
                messages=[{"role": "system", "text": "Persona"}],
            ).sentence_tts
        )
        with self.assertRaises(ValidationError):
            AudioCallTask(mode="transcribe", audio="wav", sentence_tts=True)
        for flags in (
            {"sentence_tts": "true"},
            {"sentence_tts": True, "codec_streaming_probe": True},
        ):
            with self.assertRaises(ValidationError):
                AudioCallTask(
                    mode="turn",
                    audio="wav",
                    messages=[{"role": "system", "text": "Persona"}],
                    **flags,
                )
