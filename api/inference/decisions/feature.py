"""Device-aware Decisions wrapper over the retained Laya adapter."""
import asyncio
from api.inference.feature import native_call
from api.services.runtime import finish_cleanup


class DecisionsFeature:
    operations = ('generate',)
    name, workload = 'decisions', 'decision'
    def __init__(self, service):
        self.service = service
    @property
    def adapter(self):
        return self.service.agent
    def select(self, request):
        return 'laya'
    def validate(self, payload, operation):
        # Route-owned models import lazily to avoid the manager construction cycle.
        from api.routes.v1.decisions import DecisionRequest
        return DecisionRequest.model_validate(payload)
    async def preflight_execution(self, request):
        if hasattr(self.service, "preflight"):
            await self.service.preflight()
    async def load(self, model=None):
        await self.service.load()
        return self.adapter
    async def offload_to_ram(self):
        if hasattr(self.service, 'offload_to_ram'):
            await native_call(self.service.offload_to_ram)
    async def unload(self):
        await finish_cleanup(asyncio.create_task(self.service.close()))
    async def __call__(self, request, *, model=None, operation='generate', job_id=None):
        answers = await self.service.evaluate(request.state, request.questions)
        return [answer.model_dump(mode='json') if hasattr(answer, 'model_dump') else answer for answer in answers]
