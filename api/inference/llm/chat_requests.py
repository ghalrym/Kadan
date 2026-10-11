"""Chat wrapper over the selected ChatRuntime/ReloadableAdapter."""
import asyncio
from contextlib import suppress

from api.inference.cancellation import run_cancellable_thread
from api.services.model_downloads import model_manager
from api.inference.errors import InferenceFailure
from api.inference.cancellation import await_cleanup


class ChatRequests:
    operations = ('generate', 'completion', 'load')
    name, workload = 'llm', 'llm'
    def __init__(self, chat_runtime):
        self.chat_runtime = chat_runtime
    @property
    def adapter(self):
        return self.chat_runtime.adapter
    def check_execution_state(self):
        check = getattr(self.adapter, 'check_execution_state', None)
        if check is not None:
            check()
    def select(self, request):
        return request.model or self.chat_runtime.model_id or model_manager._read_selected_model_id()
    def validate(self, payload, operation):
        # Route-owned models import lazily to avoid routes -> manager -> wrapper cycles.
        if operation == 'load':
            from api.routes.model_lifecycle import ModelLoadRequest
            return ModelLoadRequest.model_validate(payload)
        from api.routes.v1.chat.completions import CompletionRequest
        return CompletionRequest.model_validate(payload)
    async def load(self, model=None, **options):
        model = model or self.chat_runtime.model_id or model_manager._read_selected_model_id()
        if model is None:
            raise InferenceFailure('Select a language model before submitting a request.', 422)
        entry = model_manager._language_entry(model)
        if not entry.inference_available:
            raise InferenceFailure(f'{entry.repo_id} has no native language execution implementation in this build.', 422)
        if not options and self.chat_runtime.state in ('ready', 'offloaded') and self.chat_runtime.model_id == model:
            return self.adapter
        await run_cancellable_thread(model_manager.ensure_checkpoint, model)
        try:
            await self.chat_runtime.load(model, **options)
            if self.chat_runtime.task is not None:
                await asyncio.shield(self.chat_runtime.task)
        except asyncio.CancelledError:
            self.chat_runtime._cancel.set()
            if self.chat_runtime.task is not None:
                self.chat_runtime.task.cancel()
                with suppress(Exception, asyncio.CancelledError):
                    await await_cleanup(self.chat_runtime.task)
            await await_cleanup(asyncio.create_task(self.chat_runtime.unload()))
            raise
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
        if operation == 'load':
            if model is not None and request.model_id != model:
                raise InferenceFailure('Queued model selection does not match the load request.', 422)
            options = {'context_limit': request.context_limit} if 'context_limit' in request.model_fields_set else {}
            await self.load(request.model_id, **options)
            return self.chat_runtime.status()
        model = model or self.select(request)
        if model is None:
            raise InferenceFailure('Select a language model before submitting a request.', 422)
        await self.load(model)
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
        return {'model': model, 'text': text, 'finish_reason': finish['finish_reason'], 'cache': finish.get('cache'), 'timing': finish.get('timing')}
