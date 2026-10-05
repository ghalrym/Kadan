"""Provider contracts and process-local, Kadan-owned speech residency.

Budgets are admission estimates, not measured allocations or hard memory limits.
One API process shares its ResourceManager with other workloads. Multiple API
processes do not coordinate ownership; that requires a separate coordination layer.
"""
from dataclasses import dataclass, field
import io
import threading
import traceback
from typing import Callable, Protocol
import uuid
import wave

from api.inference.resources import ResourceBusy, ResourceCancelled, ResourceManager


class SpeechUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SpeechInput:
    script: str
    voice: dict
    language: str = 'Auto'
    model_id: str | None = None


@dataclass(frozen=True)
class SpeechResult:
    wav: bytes
    voice: str


def wav_duration(result: SpeechResult) -> float:
    """Validate the provider-neutral PCM WAV contract before transport."""
    try:
        with wave.open(io.BytesIO(result.wav)) as audio:
            frames, rate = audio.getnframes(), audio.getframerate()
            expected = frames * audio.getnchannels() * audio.getsampwidth()
            if frames <= 0 or rate <= 0 or len(audio.readframes(frames)) != expected:
                raise ValueError('Empty or truncated audio')
            return frames / rate
    except (AttributeError, TypeError, OSError, EOFError, wave.Error, ValueError) as exc:
        raise SpeechUnavailable('Speech provider returned invalid PCM WAV audio.') from exc


@dataclass(frozen=True)
class SpeechModel:
    id: str
    name: str
    mode: str
    speakers: tuple[str, ...] = ()
    supports_instruction: bool = False
    default_speaker: str | None = None


class SpeechSession(Protocol):
    """Construct without allocations; load/generate only under the owner's lease.

    unload must finish freeing allocations and pending device operations before
    returning. If cleanup fails, raise so the owner retains its accounting.
    """
    def load(self, cancel: threading.Event) -> None: ...
    def generate(self, request: SpeechInput, cancel: threading.Event) -> SpeechResult: ...
    def unload(self) -> None: ...


@dataclass(frozen=True)
class SpeechPlan:
    identity: tuple[str, ...]
    host_bytes: int
    create: Callable[[], SpeechSession]
    device_bytes: dict[int, int] = field(default_factory=dict)


class SpeechProvider(Protocol):
    def models(self) -> tuple[SpeechModel, ...]: ...
    def enabled(self, model_id: str) -> bool: ...
    def validate(self, request: SpeechInput) -> None: ...
    def prepare(self, request: SpeechInput) -> SpeechPlan: ...


class SpeechRegistry:
    """Register adapters at composition time, without route or scheduler edits."""
    def __init__(self):
        self._models: dict[str, tuple[SpeechModel, SpeechProvider]] = {}

    def register(self, provider: SpeechProvider) -> None:
        models = provider.models()
        ids = [model.id for model in models]
        if len(set(ids)) != len(ids) or any(key in self._models for key in ids):
            raise ValueError('Speech model IDs must have exactly one provider')
        self._models.update((model.id, (model, provider)) for model in models)

    def models(self) -> list[SpeechModel]:
        return [model for model, provider in self._models.values() if provider.enabled(model.id)]

    def resolve(self, request: SpeechInput) -> tuple[SpeechInput, SpeechProvider]:
        model_id = request.model_id
        if not model_id:
            choices = [model for model, _ in self._models.values() if model.mode == request.voice['mode']]
            enabled = [model for model in choices if self._models[model.id][1].enabled(model.id)]
            if not choices:
                raise ValueError('No speech provider is registered for this voice mode')
            model_id = (enabled or choices)[0].id
        if model_id not in self._models:
            raise ValueError('Unknown speech model')
        model, provider = self._models[model_id]
        if request.voice['mode'] != model.mode:
            raise ValueError('Choose a speech model supporting this voice mode')
        normalized = SpeechInput(request.script, dict(request.voice), request.language, model_id)
        provider.validate(normalized)
        return normalized, provider


class SpeechRuntime:
    """One evictable resident, explicit load/unload, no durable queue or batching.

    The gate serializes transitions and generation. Eviction never waits for this
    gate while holding ResourceManager's admission transaction: busy owners refuse
    eviction instead. An idle resident keeps its accounting but no active lease.
    """
    def __init__(self, registry: SpeechRegistry, resources: Callable[[], ResourceManager]):
        self.registry = registry
        self._resources = resources
        self._gate = threading.Lock()
        self._session: SpeechSession | None = None
        self._reservation = None
        self._identity = None
        self._cancel: threading.Event | None = None
        self._ready = False

    def _unload(self):
        if self._session is not None:
            self._ready = False
            self._session.unload()
        # Failed cleanup deliberately leaves both handle and reservation intact.
        if self._reservation is not None:
            self._reservation.release()
        self._session = self._reservation = self._identity = None

    def _evict(self):
        if not self._gate.acquire(blocking=False):
            raise ResourceBusy('Speech is active or changing residency')
        try:
            self._unload()
        finally:
            self._gate.release()

    def unload(self):
        """Cancel active work and await cleanup before releasing residency."""
        while not self._gate.acquire(timeout=.05):
            if self._cancel is not None:
                self._cancel.set()
        try:
            self._unload()
        finally:
            self._gate.release()

    def _run(self, request: SpeechInput, cancel: threading.Event, generate: bool):
        if not self._gate.acquire(blocking=False):
            raise ResourceBusy('Speech is already active; retry when it finishes')
        self._cancel = cancel
        try:
            if cancel.is_set():
                raise ResourceCancelled('Speech generation cancelled')
            request, provider = self.registry.resolve(request)
            if not provider.enabled(request.model_id):
                raise SpeechUnavailable('This speech checkpoint integration is not enabled.')
            plan = provider.prepare(request)
            identity = (id(provider), request.model_id, plan.identity)
            if not self._ready or self._identity != identity:
                self._unload()
                self._reservation = self._resources().reserve(
                    'speech-' + uuid.uuid4().hex, 'speech', host_bytes=plan.host_bytes,
                    device_bytes=plan.device_bytes, evict=self._evict, cancel_event=cancel)
                try:
                    with self._reservation.lease(cancel):
                        self._session = plan.create()
                        self._session.load(cancel)
                        if cancel.is_set():
                            raise ResourceCancelled('Speech loading cancelled')
                        self._identity, self._ready = identity, True
                except BaseException as exc:
                    traceback.clear_frames(exc.__traceback__)
                    self._unload()
                    raise
            if not generate:
                return None
            try:
                with self._reservation.lease(cancel):
                    result = self._session.generate(request, cancel)
                    wav_duration(result)
                    if cancel.is_set():
                        raise ResourceCancelled('Speech generation cancelled')
                    return result
            except BaseException as exc:
                traceback.clear_frames(exc.__traceback__)
                self._unload()
                raise
        finally:
            self._cancel = None
            self._gate.release()

    def load(self, request: SpeechInput, cancel: threading.Event):
        """Admit/load without generation; repeated matching loads retain residency."""
        self._run(request, cancel, False)

    def generate(self, request: SpeechInput, cancel: threading.Event) -> SpeechResult:
        return self._run(request, cancel, True)
