"""Chat wrapper over the selected ChatRuntime/ReloadableAdapter."""
from api.inference.cancellation import run_cancellable_thread
from api.inference.errors import InferenceFailure
from api.inference.cancellation import await_cleanup


class ChatRequests:
    operations = ('generate', 'completion')
    name, workload = 'llm', 'llm'
    def __init__(self, chat_runtime):
        self.chat_runtime = chat_runtime
    @property
    def adapter(self):
        return self.chat_runtime.adapter
    def select(self, request):
        return request.model or self.chat_runtime.model_id
    def validate(self, payload, operation):
        # Route-owned models import lazily to avoid routes -> manager -> wrapper cycles.
        from api.routes.v1.chat.completions import CompletionRequest
        return CompletionRequest.model_validate(payload)
    async def load(self, model=None):
        model = model or self.chat_runtime.model_id
        if self.chat_runtime.state == 'ready' and self.chat_runtime.model_id == model:
            return self.adapter
        await self.chat_runtime.load(model)
        if self.chat_runtime.task is not None:
            await await_cleanup(self.chat_runtime.task)
        if self.chat_runtime.state != 'ready':
            raise InferenceFailure(self.chat_runtime.error or 'Chat model could not load.')
        return self.adapter
    async def offload_to_ram(self):
        # Existing per-expert/device callbacks preserve the ReloadableAdapter host bank.
        await run_cancellable_thread(self._offload)
    def _offload(self, cancel):
        self.chat_runtime.ensure_resources().offload_workload_devices(self.workload, cancel)
    async def unload(self):
        await self.chat_runtime.unload()
    async def __call__(self, request, *, model=None, operation='generate', job_id=None, on_event=None):
        model = model or self.select(request)
        if model is None:
            raise InferenceFailure('No model is ready. Load a model in Settings.')
        if operation == 'generate':
            return await self.chat_runtime.complete(request.messages, model,
                **({"on_event": on_event} if on_event else {}))
        finish = {}
        def emit(event):
            if 'finish_reason' in event or 'cache' in event or 'timing' in event:
                finish.update(event)
            if on_event is not None:
                on_event(event)
        text = await self.chat_runtime.complete(request.messages, model, on_event=emit,
            **({"conversation_id": request.conversation_id} if request.conversation_id and request.reuse_prefix else {}))
        if finish.get('finish_reason') not in ('stop', 'length'):
            raise InferenceFailure('Generation ended without a terminal event.', 502)
        return {'text': text, 'finish_reason': finish['finish_reason'], 'cache': finish.get('cache'), 'timing': finish.get('timing')}
