"""Speech wrapper sharing Kadan's queue and sole resource owner."""
import os

from api.inference.cancellation import run_cancellable_thread
from api.inference.resources import ResourceBusy, ResourceCancelled, ResourceExhausted
from api.inference.tts.speech_runtime import SpeechInput, SpeechUnavailable
from api.services import speech
from api.inference.errors import InferenceFailure


class SpeechRequests:
    operations = ('generate',)
    name, workload = 'tts', 'tts'
    def __init__(self, speech_runtime=None):
        self.speech_runtime = speech_runtime or speech.speech_runtime
    @property
    def adapter(self):
        return self.speech_runtime._session
    def check_execution_state(self):
        check = getattr(self.adapter, "check_execution_state", None)
        if check is not None:
            check()
    def select(self, request):
        return request.model_id
    def validate(self, payload, operation):
        # Route-owned models remain lazy to avoid the manager construction cycle.
        from api.routes.v1.audio.speech import SpeechRequest
        return SpeechRequest.model_validate(payload)
    async def load(self, model=None):
        options = self.speech_runtime.registry.models()
        selected = next((m for m in options if m.id == model), None) if model else next(iter(options), None)
        if selected is None:
            raise InferenceFailure('Select an enabled speech model.', 422)
        request = SpeechInput('', {'mode': selected.mode, 'speaker': selected.default_speaker or (selected.speakers[0] if selected.speakers else None)}, language='English' if os.environ.get('KADAN_NATIVE_TTS_WORKER') else 'Auto', model_id=selected.id)
        await run_cancellable_thread(self.speech_runtime.load, request)
        return self.adapter
    async def offload_to_ram(self):
        await run_cancellable_thread(self.speech_runtime.offload_to_ram)
    async def unload(self):
        await run_cancellable_thread(lambda cancel: self.speech_runtime.unload())
    async def __call__(self, request, *, model=None, operation='generate', job_id=None):
        if model is not None and model != request.model_id:
            raise InferenceFailure('Queued model selection does not match the request.', 422)
        try:
            return await run_cancellable_thread(speech.generate_speech, request.model_dump(), self.speech_runtime)
        except (SpeechUnavailable, ResourceExhausted) as exc:
            raise InferenceFailure(str(exc), 503) from exc
        except ResourceBusy as exc:
            raise InferenceFailure(str(exc), 409) from exc
        except ResourceCancelled as exc:
            raise InferenceFailure(str(exc), 499) from exc
