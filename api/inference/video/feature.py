"""Video wrapper owning the selected native provider through VideoJobs."""
from api.inference.feature import native_call
from api.inference.video import VideoSpec


class VideoFeature:
    operations = ('generate',)
    name, workload = 'video', 'video'
    def __init__(self, service):
        self.service = service
        self.model = None
    @property
    def adapter(self):
        return self.service.provider(self.model) if self.model is not None else None
    def select(self, request):
        return request.model
    def validate(self, payload, operation):
        # Route-owned models import lazily to avoid the manager construction cycle.
        from api.routes.v1.videos.generations import VideoGenerationRequest
        return VideoGenerationRequest.model_validate(payload)
    def preflight(self, request):
        self.service.validate(request.model, VideoSpec(**request.model_dump(exclude={'model'})))
    def queued(self, job_id, request):
        return self.service.prepare(job_id, VideoSpec(**request.model_dump(exclude={'model'})))
    async def load(self, model=None):
        selected = model or self.model
        await native_call(self.service.load, selected)
        self.model = selected
        return self.adapter
    async def offload_to_ram(self):
        await native_call(self.service.offload_to_ram)
    async def unload(self):
        await native_call(lambda cancel: self.service.unload())
        self.model = None
    async def __call__(self, request, *, model=None, operation='generate', job_id=None):
        model = model or self.select(request)
        self.model = model
        spec = VideoSpec(**request.model_dump(exclude={'model'}))
        # Native provider retains atomic load/forward leases and output publication.
        return await native_call(self.service.run, job_id, model, spec)
