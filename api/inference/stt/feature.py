"""Whisper wrapper retaining one selected native checkpoint."""
from api.inference.feature import native_call


class STTFeature:
    operations = ('generate',)
    name, workload = 'stt', 'speech'
    def __init__(self, service):
        self.service = service
    @property
    def adapter(self):
        return self.service.native
    def select(self, request):
        return request.model or self.service.selected()
    def validate(self, payload, operation):
        # Route-owned models import lazily to avoid the manager construction cycle.
        from api.routes.v1.audio.transcriptions import TranscriptionRequest
        return TranscriptionRequest.model_validate(payload)
    async def load(self, model=None):
        await native_call(self.service.load, model)
        return self.adapter
    async def offload_to_ram(self):
        await native_call(self.service.offload_to_ram)
    async def unload(self):
        await native_call(lambda cancel: self.service.close())
    async def __call__(self, request, *, model=None, operation='generate', job_id=None):
        # Validation, loading and forward remain one leased native transaction.
        model = model or self.select(request)
        result = await native_call(self.service.transcribe, request.audio, model, request.language)
        result['formatting_status'] = 'unavailable' if request.formatting else 'disabled'
        return result
