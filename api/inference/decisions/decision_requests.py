"""Device-aware Decisions wrapper over the retained Laya adapter."""
import asyncio
from api.inference.feature import native_call
from api.services.runtime import finish_cleanup


class DecisionRequests:
    operations = ('generate',)
    name, workload = 'decisions', 'decision'
    def __init__(self, evaluator):
        self.evaluator = evaluator
    @property
    def adapter(self):
        return self.evaluator.agent
    def select(self, request):
        return 'laya'
    def validate(self, payload, operation):
        # Route-owned models import lazily to avoid the manager construction cycle.
        from api.routes.v1.decisions import DecisionRequest
        return DecisionRequest.model_validate(payload)
    def check_execution_state(self):
        check = getattr(self.evaluator, 'check_execution_state', None)
        if check is not None:
            check()
    async def preflight_execution(self, request):
        if hasattr(self.evaluator, "preflight"):
            await self.evaluator.preflight()
    async def load(self, model=None):
        await self.evaluator.load()
        return self.adapter
    async def offload_to_ram(self):
        if hasattr(self.evaluator, 'offload_to_ram'):
            await native_call(self.evaluator.offload_to_ram)
    async def unload(self):
        await finish_cleanup(asyncio.create_task(self.evaluator.close()))
    async def __call__(self, request, *, model=None, operation='generate', job_id=None):
        answers = await self.evaluator.evaluate(request.state, request.questions)
        return [answer.model_dump(mode='json') if hasattr(answer, 'model_dump') else answer for answer in answers]
