"""Native/HTTP unit tests bypass the broker; test_queue covers real Redis."""
import uuid

from api.memory_manager.queue import Job


def direct_feature(manager, feature):
    async def call(body, operation='generate', **kwargs):
        model = getattr(body, 'model', getattr(body, 'model_id', None))
        if feature == 'llm':
            model = model or manager.runtime.model_id
        if feature == 'stt':
            model = model or manager.transcription.selected()
        return await manager._execute(Job(id=uuid.uuid4().hex, feature=feature,
            operation=operation, model=model, payload=body.model_dump(mode='json')))
    return call
