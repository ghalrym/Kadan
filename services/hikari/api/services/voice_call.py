"""A call owns only queue requests and turn history, never an inference engine."""

import asyncio
import os
from contextlib import suppress
from typing import Literal

import anyio
from api.pydantic_models.chat import ChatMessage
from api.pydantic_models.inference_task import (
    AudioCallTask,
    ChatTask,
    SpeechTask,
    SpeechVoice,
    TranscriptionTask,
)
from api.pydantic_models.model_store import ModelStoreStatus
from api.services.inference_cancellation import await_cleanup
from api.services.inference_errors import InferenceFailure
from experimental_worker.call_history import append_call_history
from experimental_worker.speech_segments import speech_scripts, speech_segment
from pydantic import BaseModel


class CallModels(BaseModel):
    backend: Literal["native", "lfm3"] = "native"
    llm: str
    transcription: str
    speech: str

    @classmethod
    def selected(cls, status: ModelStoreStatus, backend: str = "native"):
        if backend == "lfm3":
            model = next(
                (model for model in status.models if model.id == "lfm3-audio-2b-bf16"),
                None,
            )
            if (
                model is None
                or model.status != "complete"
                or not model.inference_available
            ):
                raise InferenceFailure(
                    "Review the LFM3-Audio license and download its checkpoint in Hikari before calling.",
                    422,
                )
            return cls(
                backend="lfm3",
                llm=model.id,
                transcription=model.id,
                speech=model.id,
            )
        assignments = {
            kind: getattr(status.defaults, kind)
            for kind in ("llm", "transcription", "speech")
        }
        for kind, model_id in assignments.items():
            model = next(
                (model for model in status.models if model.id == model_id), None
            )
            if (
                model is None
                or model.status != "complete"
                or not model.inference_available
            ):
                raise InferenceFailure(
                    f"Download Hikari's {kind} model before calling.", 422
                )
        return cls(**assignments)

    def participants(self) -> frozenset[str]:
        return frozenset((self.llm, self.transcription, self.speech))


class VoiceCall:
    def __init__(self, inference, models: CallModels, instruction: str, send):
        self.inference = inference
        self.models = models
        self.instruction = instruction
        self.send = send
        self.turn = 0
        self.accepted = False
        self.response: asyncio.Task | None = None
        self.history: list[ChatMessage] = []
        self.segments: list[str] = []
        self.played = 0

    def check_models(self):
        if (
            CallModels.selected(self.inference.models_status(), self.models.backend)
            != self.models
        ):
            raise InferenceFailure(
                "Call model availability changed. Start a new call.", 409
            )

    async def emit(self, turn: int, **event):
        self.check_models()
        if turn == self.turn:
            await self.send({"turn": turn, **event})

    def remember(self, role, text):
        self.history[:] = [
            ChatMessage.model_validate(message)
            for message in append_call_history(
                [message.model_dump() for message in self.history],
                role,
                text,
                model_bounded=self.models.backend == "lfm3",
            )
        ]

    async def stop_response(self):
        if self.response is not None:
            self.response.cancel()
            with suppress(asyncio.CancelledError):
                await await_cleanup(self.response)
            self.response = None

    async def interrupt(self, turn: int):
        if turn <= self.turn:
            raise InferenceFailure("A new utterance requires a newer turn.", 422)
        # Invalidate first, then join native cancellation before accepting new work.
        self.turn = turn
        await self.stop_response()
        self.remember("assistant", " ".join(self.segments[: self.played]))
        self.segments, self.played = [], 0
        self.accepted = False
        await self.emit(turn, type="interrupted")

    async def run(self, receive):
        incoming = asyncio.create_task(receive())
        try:
            await self.send({"type": "ready", "models": self.models.model_dump()})
            while True:
                self.check_models()
                waiting = [incoming] + ([self.response] if self.response else [])
                done, _ = await asyncio.wait(
                    waiting, timeout=1, return_when=asyncio.FIRST_COMPLETED
                )
                if self.response is not None and self.response in done:
                    await self.response
                    self.response = None
                if incoming not in done:
                    continue
                event = incoming.result()
                incoming = asyncio.create_task(receive())
                if event.type == "end":
                    return
                if event.type == "interrupt":
                    await self.interrupt(event.turn)
                elif event.turn != self.turn:
                    # Late playback acknowledgements cannot enter another turn.
                    continue
                elif event.type == "played":
                    if event.segment != self.played or self.played >= len(
                        self.segments
                    ):
                        raise InferenceFailure(
                            "Unexpected playback acknowledgement.", 422
                        )
                    self.played += 1
                elif event.type == "utterance":
                    if self.turn == 0 or self.accepted:
                        raise InferenceFailure(
                            "Interrupt before sending a new utterance.", 422
                        )
                    self.accepted = True
                    self.response = asyncio.create_task(
                        self.respond(event.audio, self.turn)
                    )
        finally:
            self.turn += 1
            with anyio.CancelScope(shield=True):
                incoming.cancel()
                await asyncio.gather(incoming, return_exceptions=True)
                await self.stop_response()

    async def respond_audio_turn(self, audio: str, turn: int):
        sentence_tts = os.getenv("HIKARI_LFM3_AUDIO_SENTENCE_TTS", "").lower() in (
            "1",
            "true",
            "yes",
        )
        codec_probe = os.getenv("HIKARI_LFM3_AUDIO_STREAM_PROBE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        if sentence_tts and codec_probe:
            raise InferenceFailure(
                "Sentence TTS cannot run with the private codec probe.", 422
            )
        stream = await self.inference.open_audio_turn(
            AudioCallTask(
                mode="turn",
                sentence_tts=sentence_tts,
                codec_streaming_probe=codec_probe,
                audio=audio,
                model_id=self.models.llm,
                messages=[
                    ChatMessage(role="system", text=self.instruction),
                    *self.history,
                ],
            )
        )
        transcript, answer = None, None
        expected_scripts = []
        received = False
        try:
            async for event in stream:
                if turn != self.turn:
                    return
                if event["type"] == "transcript":
                    if transcript is not None:
                        raise InferenceFailure(
                            "Audio turn returned more than one transcript.", 502
                        )
                    if turn != self.turn:
                        return
                    transcript = event["text"].strip()
                    self.remember("user", transcript)
                    if transcript:
                        await self.emit(turn, type="transcript", text=transcript)
                elif event["type"] == "answer":
                    if not sentence_tts or not transcript or answer is not None:
                        raise InferenceFailure("Unexpected public audio answer.", 502)
                    answer = event["text"].strip()
                    if not answer:
                        raise InferenceFailure("Invalid public audio answer.", 502)
                    expected_scripts = speech_scripts(answer)
                elif event["type"] == "segment":
                    index, script = event["index"], event["script"]
                    if (
                        not sentence_tts
                        or answer is None
                        or type(index) is not int
                        or index != len(self.segments)
                        or index >= len(expected_scripts)
                        or script != expected_scripts[index]
                        or not event.get("audio_base64")
                    ):
                        raise InferenceFailure(
                            "Unexpected known-script audio segment.", 502
                        )
                    self.segments.append(script)
                    if index == 0:
                        # Preserve the complete answer's original spacing. The
                        # scripts identify played history, not display deltas.
                        await self.emit(turn, type="text", text=answer)
                    await self.emit(
                        turn, type="audio", segment=index, audio=event["audio_base64"]
                    )
                elif event["type"] == "reply":
                    if received:
                        raise InferenceFailure(
                            "Audio turn returned more than one final receipt.", 502
                        )
                    received = True
                    if (
                        transcript is None
                        or event.get("transcript", "").strip() != transcript
                    ):
                        raise InferenceFailure(
                            "Audio turn did not confirm its transcript.", 502
                        )
                    if transcript and sentence_tts:
                        if (
                            answer is None
                            or event.get("text") != answer
                            or event.get("scripts") != expected_scripts
                            or type(event.get("segment_count")) is not int
                            or event["segment_count"] != len(expected_scripts)
                            or self.segments != expected_scripts
                        ):
                            raise InferenceFailure(
                                "Sentence TTS final receipt did not match its completed scripts.",
                                502,
                            )
                        await self.emit(turn, type="done")
                    elif transcript:
                        await self.publish_audio_reply(event, turn)
                    else:
                        await self.emit(turn, type="done")
            if not received:
                raise InferenceFailure("Audio turn ended without a final receipt.", 502)
        finally:
            with anyio.CancelScope(shield=True):
                await await_cleanup(asyncio.create_task(stream.aclose()))

    async def publish_audio_reply(self, result: dict, turn: int):
        reply, audio_base64 = result.get("text", "").strip(), result.get("audio_base64")
        if not reply or not audio_base64:
            raise InferenceFailure(
                "Experimental audio worker returned an incomplete reply.", 502
            )
        self.segments.append(reply)
        await self.emit(turn, type="text", text=reply)
        await self.emit(turn, type="audio", segment=0, audio=audio_base64)
        await self.emit(turn, type="done")

    async def respond_audio(self, audio: str, turn: int):
        single_turn = os.getenv("HIKARI_LFM3_AUDIO_SINGLE_TURN", "").lower() in (
            "1",
            "true",
            "yes",
        )
        if (
            os.getenv("HIKARI_LFM3_AUDIO_SENTENCE_TTS", "").lower()
            in ("1", "true", "yes")
            and not single_turn
        ):
            raise InferenceFailure(
                "Sentence TTS requires the single-turn audio worker path.", 422
            )
        if single_turn:
            await self.respond_audio_turn(audio, turn)
            return
        transcript = await self.inference.submit(
            AudioCallTask(mode="transcribe", audio=audio, model_id=self.models.llm),
            feature="audio_call",
        )
        text = transcript["text"].strip()
        if not text:
            await self.emit(turn, type="done")
            return
        self.remember("user", text)
        await self.emit(turn, type="transcript", text=text)
        result = await self.inference.submit(
            AudioCallTask(
                mode="reply",
                model_id=self.models.llm,
                messages=[
                    ChatMessage(role="system", text=self.instruction),
                    *self.history,
                ],
            ),
            feature="audio_call",
        )
        # One buffered reply is one playback acknowledgement. Unheard replies
        # never enter history when this turn is interrupted.
        await self.publish_audio_reply(result, turn)

    async def respond(self, audio: str, turn: int):
        if self.models.backend == "lfm3":
            await self.respond_audio(audio, turn)
            return
        transcript = await self.inference.submit(
            TranscriptionTask(
                audio=audio,
                model=self.models.transcription.removeprefix("whisper-"),
                formatting=False,
            ),
            feature="stt",
        )
        text = transcript["text"].strip()
        if not text:
            await self.emit(turn, type="done")
            return
        self.remember("user", text)
        await self.emit(turn, type="transcript", text=text)
        stream, _ = await self.inference.open_chat_stream(
            ChatTask(
                messages=[
                    ChatMessage(role="system", text=self.instruction),
                    *self.history,
                ],
                model=self.models.llm,
                stream=True,
            )
        )
        # The 8000-character reply limit bounds this queue. Do not block chat
        # completion when synthesis is waiting for the same physical memory.
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def generate():
            pending = ""
            total = 0
            finish = None
            async for event in stream:
                finish = event.get("finish_reason", finish)
                content = event.get("content", "")
                total += len(content)
                if total > 8000:
                    raise InferenceFailure("Spoken reply exceeded the call limit.", 502)
                if content:
                    await self.emit(turn, type="text", text=content)
                    pending += content
                    while True:
                        segment, pending = speech_segment(pending)
                        if not segment:
                            break
                        await queue.put(segment)
            if finish not in ("stop", "length"):
                raise InferenceFailure(
                    "Spoken reply ended without a terminal event.", 502
                )
            if pending.strip():
                await queue.put(pending.strip())
            await queue.put(None)

        async def speak():
            while (segment := await queue.get()) is not None:
                result = await self.inference.submit(
                    SpeechTask(
                        script=segment,
                        model_id=self.models.speech,
                        language="English",
                        voice=SpeechVoice(mode="custom", speaker="Ryan"),
                    ),
                    feature="tts",
                )
                if not result.get("audio_base64"):
                    raise InferenceFailure("Speech worker returned no audio.", 502)
                index = len(self.segments)
                self.segments.append(segment)
                await self.emit(
                    turn, type="audio", segment=index, audio=result["audio_base64"]
                )

        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(generate())
                group.create_task(speak())
        except ExceptionGroup as error:
            # Preserve actionable admission/model failures from either stage.
            if isinstance(error.exceptions[0], InferenceFailure):
                raise error.exceptions[0] from error
            raise
        finally:
            with anyio.CancelScope(shield=True):
                await await_cleanup(asyncio.create_task(stream.aclose()))
        await self.emit(turn, type="done")
