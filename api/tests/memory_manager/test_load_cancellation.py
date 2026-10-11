import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from api.memory_manager.queue import InferenceQueue

class LoadCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_unload_cancels_and_joins_chat_preparation_only(self):
        jobs = {name: json.dumps(dict(id=name,feature=feature,operation='load',model='small',payload={}))
                for name,feature in [('a'*32,'llm'),('b'*32,'image')]}
        queue = SimpleNamespace(key=lambda value: value, cancel=AsyncMock(),wait=AsyncMock(),
            redis=SimpleNamespace(smembers=AsyncMock(return_value=set(jobs)),
                hget=AsyncMock(side_effect=lambda key,field:jobs[key.removeprefix('job:')])) )
        await InferenceQueue.cancel_feature(queue,'llm')
        queue.cancel.assert_awaited_once_with('a'*32)
        queue.wait.assert_awaited_once_with('a'*32)
