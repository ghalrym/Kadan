"""HTTP -> shared Redis FIFO -> synthetic decision IPC; no numerical claims."""
import asyncio
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi import FastAPI
import httpx

from api.inference.decisions.native import NativeDecisionManager, RAM_BYTES
from api.inference.resources import ResourceManager
from api.memory_manager import MemoryManager
from api.memory_manager.queue import InferenceQueue
from api.routes.v1.decisions import router
from api.services.runtime import RuntimeFailure
from api.tests.inference.decisions.test_native import SCRIPT, body


@unittest.skipUnless(os.getenv('KADAN_TEST_REDIS_URL'), 'Dedicated Redis URL required')
class NativeDecisionBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        script = root / 'worker.py'
        script.write_text(SCRIPT)
        self.resources = ResourceManager(RAM_BYTES * 2, {})
        self.native = NativeDecisionManager(self.resources, resolve=lambda: [sys.executable, str(script)])
        self.manager = MemoryManager(decisions=self.native, queue=object())
        self.manager.features = {'decisions': self.manager.decisions}
        self.prefix = 'kadan:test:' + uuid4().hex + ':'
        self.manager.queue = InferenceQueue(self.manager._execute, url=os.environ['KADAN_TEST_REDIS_URL'],
            prefix=self.prefix, lock_path=root / 'queue.lock')
        await self.manager.start()
        app = FastAPI()
        app.include_router(router)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test')
        self.route = patch('api.routes.v1.decisions.memory_manager', self.manager)
        self.route.start()

    async def asyncTearDown(self):
        self.route.stop()
        await self.client.aclose()
        await self.manager.queue.close()
        redis = self.manager.queue.redis
        keys = [key async for key in redis.scan_iter(match=self.prefix + '*')]
        if keys:
            await redis.delete(*keys)
        await redis.aclose()
        await self.native.close()
        self.assertFalse(self.resources.snapshot()['reservations'])
        self.temp.cleanup()

    async def test_http_json_roundtrip_preserves_order_and_noul_aliases(self):
        response = await self.client.post('/v1/decisions', json=body().model_dump(mode='json', by_alias=True))
        self.assertEqual(response.status_code, 200, response.text)
        answers = response.json()['answers']
        self.assertEqual([answer['type'] for answer in answers], ['Choice', 'Score', 'Noul'])
        self.assertEqual(answers[0]['value'], 'red')
        self.assertIsNone(answers[2]['probabilities'])
        worker = self.native.agent
        again = await self.client.post('/v1/decisions', json=body().model_dump(mode='json', by_alias=True))
        self.assertEqual(again.json(), response.json())
        self.assertIs(worker, self.native.agent)

    async def test_http_errors_remain_typed_and_cleanup(self):
        for state, status in [('reject', 422), ('wrong', 502)]:
            response = await self.client.post('/v1/decisions', json=body(state).model_dump(mode='json', by_alias=True))
            self.assertEqual(response.status_code, status, response.text)
            self.assertFalse(self.resources.snapshot()['reservations'])

    async def test_unicode_native_byte_limit_is_a_client_error(self):
        # Character-valid HTTP input can exceed the native UTF-8 byte limit.
        request = body('🙂' * 5000)
        self.assertLess(len(request.model_dump_json()), 16000)
        self.assertGreater(len(request.state.encode('utf-8')), 16384)
        response = await self.client.post('/v1/decisions', json=request.model_dump(mode='json', by_alias=True))
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn('decision_empty_or_long_text', response.json()['detail'])
        self.assertIsNone(self.native.agent)
        self.assertFalse(self.resources.snapshot()['reservations'])
        again = await self.client.post('/v1/decisions', json=body().model_dump(mode='json', by_alias=True))
        self.assertEqual(again.status_code, 200, again.text)

    async def test_fifo_cancel_waits_for_child_before_image_and_later_decision(self):
        events = []
        native = self.native
        class ImageLeaf:
            operations = ('generate',)
            validate = staticmethod(lambda payload, operation: payload)
            async def offload_to_ram(self):
                pass
            async def __call__(self, request, **kwargs):
                self_test.assertIsNone(native.agent)
                self_test.assertFalse(self_test.resources.snapshot()['reservations'])
                events.append('image')
                return {'ok': True}
        self_test = self
        self.manager.features['image'] = ImageLeaf()
        original = self.manager.decisions.__class__.__call__
        # Record real decision completion; wrapper still validates and uses IPC.
        async def record(wrapper, request, **kwargs):
            result = await original(wrapper, request, **kwargs)
            events.append('decision')
            return result
        with patch.object(self.manager.decisions.__class__, '__call__', record):
            first = await self.manager.queue.submit('decisions', 'generate', body('hang').model_dump(mode='json'), 'laya')
            for _ in range(200):
                if native.agent and native.agent.io_ready:
                    break
                await asyncio.sleep(.01)
            worker = native.agent
            self.assertIsNotNone(worker)
            second = await self.manager.queue.submit('image', 'generate', {}, 'synthetic-image')
            third = await self.manager.queue.submit('decisions', 'generate', body().model_dump(mode='json'), 'laya')
            await self.manager.queue.cancel(first)
            with self.assertRaises(RuntimeFailure) as caught:
                await self.manager.queue.wait(first)
            self.assertEqual(caught.exception.status_code, 499)
            self.assertIsNotNone(worker.process.poll())
            self.assertEqual(await self.manager.queue.wait(second), {'ok': True})
            await self.manager.queue.wait(third)
        self.assertEqual(events, ['image', 'decision'])
