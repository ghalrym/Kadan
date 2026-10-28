"""A call owns only queue requests and turn history, never an inference engine."""

import asyncio
import re
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
from api.pydantic_models.model_store import ModelStoreStatus, TierId
from api.services.inference_cancellation import await_cleanup
from api.services.inference_errors import InferenceFailure
from pydantic import BaseModel


class CallModels(BaseModel):
    tier: TierId | None
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
                tier=status.selected_tier,
                backend="lfm3",
                llm=model.id,
                transcription=model.id,
                speech=model.id,
            )
        tier = next(
            (tier for tier in status.tiers if tier.id == status.selected_tier), None
        )
        if tier is None:
            raise InferenceFailure(
                "Choose a tier in Hikari Settings before calling.", 422
            )
        assignments = {
            kind: getattr(tier.models, kind)
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
                    f"Install the selected tier's {kind} model before calling.", 422
                )
        return cls(tier=tier.id, **assignments)

    def participants(self) -> frozenset[str]:
        return frozenset((self.llm, self.transcription, self.speech))


def speech_segment(text: str, final=False) -> tuple[str, str]:
    """Bound Qwen's buffered synthesis to a sentence or a short text segment."""
    boundary = re.search(r"[.!?\n](?:\s|$)", text)
    if boundary and boundary.end() <= 240:
        end = boundary.end()
    elif len(text) >= 240:
        end = text.rfind(" ", 0, 240)
        if end < 1:
            end = 240
    elif final:
        end = len(text)
    else:
        return "", text
    return text[:end].strip(), text[end:].lstrip()


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
            raise InferenceFailure("The selected tier changed. Start a new call.", 409)

    async def emit(self, turn: int, **event):
        self.check_models()
        if turn == self.turn:
            await self.send({"turn": turn, **event})

    def remember(self, role, text):
        if text.strip():
            self.history.append(ChatMessage(role=role, text=text[:4000]))
        while (
            len(self.history) > 12
            or sum(len(message.text) for message in self.history) > 12000
        ):
            self.history.pop(0)

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

    async def respond_audio(self, audio: str, turn: int):
        transcript = await self.inference.submit(
            AudioCallTask(mode="transcribe", audio=audio, model_id=self.models.llm),
            feature="audio_call",
        )
        text = transcript["text"].strip()
        if not text:
            await self.emit(turn, type="done")
            return
        if len(text) > 4000:
            raise InferenceFailure("Recognized utterance exceeded the call limit.", 502)
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
        reply, audio_base64 = result.get("text", "").strip(), result.get("audio_base64")
        if not reply or len(reply) > 8000 or not audio_base64:
            raise InferenceFailure(
                "Experimental audio worker returned an incomplete reply.", 502
            )
        # One buffered reply is one playback acknowledgement. Unheard replies
        # never enter history when this turn is interrupted.
        self.segments.append(reply)
        await self.emit(turn, type="text", text=reply)
        await self.emit(turn, type="audio", segment=0, audio=audio_base64)
        await self.emit(turn, type="done")

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
