"""Callable Qwen Image wrapper under the shared Redis inference lifecycle."""
from api.inference.feature import native_call
from api.inference.image.model import MODEL_ID
from api.services.images import image_manager
from api.services.runtime import RuntimeFailure


class ImageFeature:
    operations = ('generate', 'edit')
    name, workload = 'image', 'image'

    def __init__(self, service=None):
        self.service = service or image_manager

    @property
    def adapter(self):
        return self.service.native

    def select(self, request):
        return MODEL_ID

    def preflight(self, request):
        self.service.preflight()

    def validate(self, payload, operation):
        # Route-owned models import lazily to avoid manager construction cycles.
        from api.routes.v1.images.generations import ImageRequest
        from api.routes.v1.images.edits import ImageRequest as EditRequest
        return (EditRequest if operation == 'edit' else ImageRequest).model_validate(payload)

    async def load(self, model=None):
        if model not in (None, MODEL_ID):
            raise RuntimeFailure('Unsupported image model.', 422)
        return await native_call(self.service.load)

    async def offload_to_ram(self):
        await native_call(self.service.offload_to_ram)

    async def unload(self):
        await native_call(lambda cancel: self.service.close())

    async def __call__(self, request, *, model=None, operation='generate', job_id=None):
        if model not in (None, MODEL_ID) or operation not in self.operations:
            raise RuntimeFailure('Unsupported image model or operation.', 422)
        source = request.image if operation == 'edit' else None
        result = await native_call(lambda cancel: self.service.generate(request.prompt,
            request.aspect, request.count, request.seed, cancel, source))
        return {'image': result.model_dump(mode='json')}
