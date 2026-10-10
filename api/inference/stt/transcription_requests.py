"""Whisper wrapper retaining one selected Whisper checkpoint."""
from api.inference.cancellation import run_cancellable_thread


class TranscriptionRequests:
    operations = ('generate',)
    name, workload = 'stt', 'speech'
    def __init__(self, transcriber):
        self.transcriber = transcriber
    @property
    def adapter(self):
        return getattr(self.transcriber, "_cpp", None) or self.transcriber.whisper_model
    def check_execution_state(self):
        check = getattr(self.adapter, "check_execution_state", None)
        if check is not None:
            check()
    def select(self, request):
        return request.model or self.transcriber.selected()
    def validate(self, payload, operation):
        # Route-owned models import lazily to avoid the manager construction cycle.
        from api.routes.v1.audio.transcriptions import TranscriptionRequest
        return TranscriptionRequest.model_validate(payload)
    async def load(self, model=None):
        await run_cancellable_thread(self.transcriber.load, model)
        return self.adapter
    async def offload_to_ram(self):
        await run_cancellable_thread(self.transcriber.offload_to_ram)
    async def unload(self):
        await run_cancellable_thread(lambda cancel: self.transcriber.close())
    async def __call__(self, request, *, model=None, operation='generate', job_id=None):
        # Validation, loading and forward remain one leased model transaction.
        model = model or self.select(request)
        result = await run_cancellable_thread(self.transcriber.transcribe, request.audio, model, request.language)
        result['formatting_status'] = 'unavailable' if request.formatting else 'disabled'
        return result
