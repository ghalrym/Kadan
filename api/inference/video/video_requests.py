"""Video wrapper owning the selected video provider through VideoJobs."""
from api.inference.cancellation import run_cancellable_thread
from api.inference.video import VideoSpec


class VideoRequests:
    operations = ('generate',)
    name, workload = 'video', 'video'
    def __init__(self, video_jobs):
        self.video_jobs = video_jobs
        self.model = None
    @property
    def adapter(self):
        return self.video_jobs.provider(self.model) if self.model is not None else None
    def select(self, request):
        return request.model
    def validate(self, payload, operation):
        # Route-owned models import lazily to avoid the manager construction cycle.
        from api.routes.v1.videos.generations import VideoGenerationRequest
        return VideoGenerationRequest.model_validate(payload)
    def preflight(self, request):
        self.video_jobs.validate(request.model, VideoSpec(**request.model_dump(exclude={'model'})))
    def queued(self, job_id, request):
        return self.video_jobs.prepare(job_id, VideoSpec(**request.model_dump(exclude={'model'})))
    async def load(self, model=None):
        selected = model or self.model
        await run_cancellable_thread(self.video_jobs.load, selected)
        self.model = selected
        return self.adapter
    async def offload_to_ram(self):
        await run_cancellable_thread(self.video_jobs.offload_to_ram)
    async def unload(self):
        await run_cancellable_thread(lambda cancel: self.video_jobs.unload())
        self.model = None
    async def __call__(self, request, *, model=None, operation='generate', job_id=None):
        model = model or self.select(request)
        self.model = model
        spec = VideoSpec(**request.model_dump(exclude={'model'}))
        # Video provider retains atomic load/forward leases and output publication.
        return await run_cancellable_thread(self.video_jobs.run, job_id, model, spec)
