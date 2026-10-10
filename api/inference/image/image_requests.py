"""Native image adapter sharing the existing request FIFO and memory owner."""
from api.inference.cancellation import run_cancellable_thread
from api.inference.errors import InferenceFailure
from api.inference.image.native_worker import MODEL, NativeImageRuntime, prepare, validate
from api.inference.resources import ResourceBusy, ResourceCancelled, ResourceExhausted
from api.services.chat_runtime import chat_runtime


class ImageRequests:
    operations = ('generate', 'edit')
    name, workload = 'image', 'image'
    def __init__(self, runtime=None):
        self.runtime = runtime or NativeImageRuntime(chat_runtime.ensure_resources)
    @property
    def adapter(self):
        return self.runtime.session
    def check_execution_state(self):
        self.runtime.check_execution_state()
    def select(self, request):
        return MODEL
    def validate(self, payload, operation):
        # Route-owned types are lazy to avoid manager construction cycles.
        from api.routes.v1.images.generations import ImageRequest
        from api.routes.v1.images.edits import ImageRequest as EditRequest
        return (EditRequest if operation == 'edit' else ImageRequest).model_validate(payload)
    def preflight(self, request):
        if hasattr(request, 'image'):
            raise InferenceFailure('No image provider is configured for editing. Native image generation is available when enabled.')
        validate(request)
        prepare()
    async def load(self, model=None):
        if model not in (None, MODEL):
            raise InferenceFailure('Unsupported native image model.', 422)
        await run_cancellable_thread(self.runtime.run, None)
        return self.adapter
    async def offload_to_ram(self):
        return None  # Already CPU resident; host pressure calls the real eviction callback.
    async def unload(self):
        await run_cancellable_thread(lambda cancel: self.runtime.unload())
    async def __call__(self, request, *, model=None, operation='generate', job_id=None):
        if operation != 'generate':
            raise InferenceFailure('No image provider is configured for editing. Native image generation is available when enabled.')
        if model not in (None, MODEL):
            raise InferenceFailure('Queued image model selection mismatch.', 422)
        try:
            return await run_cancellable_thread(self.runtime.run, request)
        except (ResourceExhausted, ResourceBusy, ResourceCancelled) as error:
            code = 499 if isinstance(error, ResourceCancelled) else 409 if isinstance(error, ResourceBusy) else 503
            raise InferenceFailure(str(error), code) from error
        except InferenceFailure:
            raise
        except Exception as error:
            raise InferenceFailure('Native image execution failed; incomplete cleanup retains its reservation.') from error
