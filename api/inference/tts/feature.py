"""Native speech wrapper sharing Kadan's queue and sole resource owner."""
from api.inference.feature import native_call
from api.inference.resources import ResourceBusy, ResourceCancelled, ResourceExhausted
from api.inference.tts.runtime import SpeechInput, SpeechUnavailable
from api.services import speech
from api.services.runtime import RuntimeFailure


class TTSFeature:
    operations = ('generate',)
    name, workload = 'tts', 'tts'
    def __init__(self, service=None):
        self.service = service or speech.speech_runtime
    @property
    def adapter(self):
        return self.service._session
    def select(self, request):
        return request.model_id
    def validate(self, payload, operation):
        # Route-owned models remain lazy to avoid the manager construction cycle.
        from api.routes.v1.audio.speech import SpeechRequest
        return SpeechRequest.model_validate(payload)
    async def load(self, model=None):
        options = self.service.registry.models()
        selected = next((m for m in options if m.id == model), None) if model else next(iter(options), None)
        if selected is None:
            raise RuntimeFailure('Select an enabled speech model.', 422)
        request = SpeechInput('', {'mode': selected.mode, 'speaker': selected.default_speaker or (selected.speakers[0] if selected.speakers else None)}, model_id=selected.id)
        await native_call(self.service.load, request)
        return self.adapter
    async def offload_to_ram(self):
        await native_call(self.service.offload_to_ram)
    async def unload(self):
        await native_call(lambda cancel: self.service.unload())
    async def __call__(self, request, *, model=None, operation='generate', job_id=None):
        if model is not None and model != request.model_id:
            raise RuntimeFailure('Queued model selection does not match the request.', 422)
        try:
            return await native_call(speech.generate_speech, request.model_dump(), self.service)
        except (SpeechUnavailable, ResourceExhausted) as exc:
            raise RuntimeFailure(str(exc), 503) from exc
        except ResourceBusy as exc:
            raise RuntimeFailure(str(exc), 409) from exc
        except ResourceCancelled as exc:
            raise RuntimeFailure(str(exc), 499) from exc
