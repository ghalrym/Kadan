"""Chat wrapper over the selected RuntimeManager/ReloadableAdapter."""
from api.inference.feature import native_call
from api.services.runtime import RuntimeFailure, finish_cleanup


class LLMFeature:
    operations = ('generate',)
    name, workload = 'llm', 'llm'
    def __init__(self, service):
        self.service = service
    @property
    def adapter(self):
        return self.service.adapter
    def select(self, request):
        return request.model or self.service.model_id
    def validate(self, payload, operation):
        # Route-owned models import lazily to avoid routes -> manager -> wrapper cycles.
        from api.routes.v1.chat.completions import CompletionRequest
        return CompletionRequest.model_validate(payload)
    async def load(self, model=None):
        model = model or self.service.model_id
        if self.service.state == 'ready' and self.service.model_id == model:
            return self.adapter
        await self.service.load(model)
        if self.service.task is not None:
            await finish_cleanup(self.service.task)
        if self.service.state != 'ready':
            raise RuntimeFailure(self.service.error or 'Chat model could not load.')
        return self.adapter
    async def offload_to_ram(self):
        # Existing per-expert/device callbacks preserve the ReloadableAdapter host bank.
        await native_call(self._offload)
    def _offload(self, cancel):
        self.service.ensure_resources().offload_workload_devices(self.workload, cancel)
    async def unload(self):
        await self.service.unload()
    async def __call__(self, request, *, model=None, operation='generate', job_id=None):
        model = model or self.select(request)
        if model is None:
            raise RuntimeFailure('No model is ready. Load a model in Settings.')
        return await self.service.complete(request.messages, model)
