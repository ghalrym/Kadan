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

from api.inference.decisions.laya_subprocess import LayaSubprocessEvaluator, DEFAULT_HOST_BUDGET_BYTES
from api.inference.resources import ResourceManager
from api.memory_manager import MemoryManager
from api.memory_manager.queue import InferenceQueue, Job
from api.routes.v1.decisions import router
from api.inference.errors import InferenceFailure
from api.tests.inference.decisions.test_laya_subprocess import SCRIPT, body


@unittest.skipUnless(os.getenv('KADAN_TEST_REDIS_URL'), 'Dedicated Redis URL required')
class DecisionWorkerBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        script = root / 'worker.py'
        script.write_text(SCRIPT)
        self.resources = ResourceManager(DEFAULT_HOST_BUDGET_BYTES * 2, {})
        self.worker_manager = LayaSubprocessEvaluator(self.resources, resolve=lambda: [sys.executable, str(script)])
        self.manager = MemoryManager(decisions=self.worker_manager, queue=object())
        self.manager.request_executors = {'decisions': self.manager.decisions}
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
        await self.worker_manager.close()
        self.assertFalse(self.resources.snapshot()['reservations'])
        self.temp.cleanup()

    async def test_http_json_roundtrip_preserves_order_and_noul_aliases(self):
        response = await self.client.post('/v1/decisions', json=body().model_dump(mode='json', by_alias=True))
        self.assertEqual(response.status_code, 200, response.text)
        answers = response.json()['answers']
        self.assertEqual([answer['type'] for answer in answers], ['Choice', 'Score', 'Noul'])
        self.assertEqual(answers[0]['value'], 'red')
        self.assertIsNone(answers[2]['probabilities'])
        worker = self.worker_manager.agent
        again = await self.client.post('/v1/decisions', json=body().model_dump(mode='json', by_alias=True))
        self.assertEqual(again.json(), response.json())
        self.assertIs(worker, self.worker_manager.agent)

    async def test_http_errors_remain_typed_and_cleanup(self):
        for state, status in [('reject', 422), ('wrong', 502)]:
            response = await self.client.post('/v1/decisions', json=body(state).model_dump(mode='json', by_alias=True))
            self.assertEqual(response.status_code, status, response.text)
            self.assertFalse(self.resources.snapshot()['reservations'])

    async def test_unicode_worker_byte_limit_is_a_client_error(self):
        # Character-valid HTTP input can exceed the worker UTF-8 byte limit.
        request = body('🙂' * 5000)
        self.assertLess(len(request.model_dump_json()), 16000)
        self.assertGreater(len(request.state.encode('utf-8')), 16384)
        response = await self.client.post('/v1/decisions', json=request.model_dump(mode='json', by_alias=True))
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn('decision_empty_or_long_text', response.json()['detail'])
        self.assertIsNone(self.worker_manager.agent)
        self.assertFalse(self.resources.snapshot()['reservations'])
        again = await self.client.post('/v1/decisions', json=body().model_dump(mode='json', by_alias=True))
        self.assertEqual(again.status_code, 200, again.text)

    async def test_fifo_cancel_waits_for_child_before_image_and_later_decision(self):
        events = []
        worker_manager = self.worker_manager
        class ImageLeaf:
            operations = ('generate',)
            validate = staticmethod(lambda payload, operation: payload)
            async def offload_to_ram(self):
                pass
            async def __call__(self, request, **kwargs):
                self_test.assertIsNone(worker_manager.agent)
                self_test.assertFalse(self_test.resources.snapshot()['reservations'])
                events.append('image')
                return {'ok': True}
        self_test = self
        self.manager.request_executors['image'] = ImageLeaf()
        original = self.manager.decisions.__class__.__call__
        # Record real decision completion; wrapper still validates and uses IPC.
        async def record(wrapper, request, **kwargs):
            result = await original(wrapper, request, **kwargs)
            events.append('decision')
            return result
        with patch.object(self.manager.decisions.__class__, '__call__', record):
            first = await self.manager.queue.submit('decisions', 'generate', body('hang').model_dump(mode='json'), 'laya')
            for _ in range(200):
                if worker_manager.agent and worker_manager.agent.io_ready:
                    break
                await asyncio.sleep(.01)
            worker = worker_manager.agent
            self.assertIsNotNone(worker)
            second = await self.manager.queue.submit('image', 'generate', {}, 'synthetic-image')
            third = await self.manager.queue.submit('decisions', 'generate', body().model_dump(mode='json'), 'laya')
            await self.manager.queue.cancel(first)
            with self.assertRaises(InferenceFailure) as caught:
                await self.manager.queue.wait(first)
            self.assertEqual(caught.exception.status_code, 499)
            self.assertIsNotNone(worker.process.poll())
            self.assertEqual(await self.manager.queue.wait(second), {'ok': True})
            await self.manager.queue.wait(third)
        self.assertEqual(events, ['image', 'decision'])


    async def test_failed_reap_blocks_other_feature_until_confirmed_cleanup(self):
        calls = []
        class ImageLeaf:
            operations = ('generate',)
            validate = staticmethod(lambda payload, operation: payload)
            async def __call__(self, request, **kwargs):
                calls.append('image')
                return 'done'
        self.manager.request_executors['image'] = ImageLeaf()
        first = await self.manager.queue.submit('decisions', 'generate',
            body('hang').model_dump(mode='json'), 'laya')
        for _ in range(200):
            if self.worker_manager.agent and self.worker_manager.agent.io_ready:
                break
            await asyncio.sleep(.01)
        worker = self.worker_manager.agent
        self.assertIsNotNone(worker)
        second = await self.manager.queue.submit('image', 'generate', {}, 'synthetic-image')
        cancelled = await self.manager.queue.submit('image', 'generate', {}, 'synthetic-image')
        # The hung first request keeps both later jobs pending until cancellation.
        await self.manager.queue.cancel(cancelled)
        with patch.object(worker, 'stop', side_effect=RuntimeError('reap unconfirmed')):
            await self.manager.queue.cancel(first)
            with self.assertRaises(InferenceFailure) as stopped:
                await asyncio.wait_for(self.manager.queue.wait(first), 2)
            self.assertEqual(stopped.exception.status_code, 499)
            self.assertTrue(self.worker_manager._quarantined)
            self.assertIsNone(worker.process.poll())
            with self.assertRaises(InferenceFailure) as blocked:
                await asyncio.wait_for(self.manager.queue.wait(second), 2)
            self.assertEqual(blocked.exception.status_code, 503)
            self.assertIn('cleanup is unconfirmed', str(blocked.exception))
            with self.assertRaises(InferenceFailure) as cancelled_result:
                await asyncio.wait_for(self.manager.queue.wait(cancelled), 2)
            self.assertEqual(cancelled_result.exception.status_code, 499)
            self.assertEqual(calls, [])
            self.assertEqual(sum(row['host_bytes'] for row in
                self.resources.snapshot()['reservations'].values()), DEFAULT_HOST_BUDGET_BYTES)
        # Actual stop/reap, not a flag reset, reopens execution and releases RAM.
        await self.worker_manager.close()
        self.assertIsNotNone(worker.process.poll())
        self.assertFalse(self.resources.snapshot()['reservations'])
        resumed = await self.manager.queue.submit('image', 'generate', {}, 'synthetic-image')
        self.assertEqual(await asyncio.wait_for(self.manager.queue.wait(resumed), 2), 'done')
        self.assertEqual(calls, ['image'])
        response = await self.client.post('/v1/decisions', json=body().model_dump(mode='json', by_alias=True))
        self.assertEqual(response.status_code, 200, response.text)

    async def test_healthy_cpu_resident_does_not_block_another_feature(self):
        response = await self.client.post('/v1/decisions', json=body().model_dump(mode='json', by_alias=True))
        self.assertEqual(response.status_code, 200)
        worker = self.worker_manager.agent
        class ImageLeaf:
            operations = ('generate',)
            validate = staticmethod(lambda payload, operation: payload)
            async def __call__(self, request, **kwargs):
                return 'done'
        self.manager.request_executors['image'] = ImageLeaf()
        job = await self.manager.queue.submit('image', 'generate', {}, 'synthetic-image')
        self.assertEqual(await asyncio.wait_for(self.manager.queue.wait(job), 2), 'done')
        self.assertIs(self.worker_manager.agent, worker)
        self.assertIsNone(worker.process.poll())


class ExecutionPreflightTests(unittest.IsolatedAsyncioTestCase):
    def manager_and_job(self, wrapper):
        # Only exercise the queue execution boundary; no model or Redis setup.
        manager = object.__new__(MemoryManager)
        manager.request_executors = {'decisions': wrapper}
        job = Job(id='a' * 32, feature='decisions', operation='generate', payload={})
        return manager, job

    async def test_execution_waits_for_preflight(self):
        entered, release = asyncio.Event(), asyncio.Event()
        events = []
        class Leaf:
            operations = ('generate',)
            validate = staticmethod(lambda payload, operation: payload)
            async def preflight_execution(self, request):
                events.append('preflight')
                entered.set()
                await release.wait()
            async def __call__(self, request, **kwargs):
                events.append('execute')
                return 'done'
        manager, job = self.manager_and_job(Leaf())
        task = asyncio.create_task(manager._execute(job))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            self.assertEqual(events, ['preflight'])
            self.assertFalse(task.done())
            release.set()
            self.assertEqual(await task, 'done')
            self.assertEqual(events, ['preflight', 'execute'])
        finally:
            release.set()
            await task

    async def test_preflight_failure_prevents_execution(self):
        class Leaf:
            operations = ('generate',)
            validate = staticmethod(lambda payload, operation: payload)
            async def preflight_execution(self, request):
                raise InferenceFailure('Checkpoint unavailable.', 503)
            async def __call__(self, request, **kwargs):
                raise AssertionError('Rejected preflight must not execute')
        manager, job = self.manager_and_job(Leaf())
        with self.assertRaises(InferenceFailure) as caught:
            await manager._execute(job)
        self.assertEqual(caught.exception.status_code, 503)

    async def test_features_without_preflight_still_execute(self):
        class Leaf:
            operations = ('generate',)
            validate = staticmethod(lambda payload, operation: payload)
            async def __call__(self, request, **kwargs):
                return 'unchanged'
        manager, job = self.manager_and_job(Leaf())
        self.assertEqual(await manager._execute(job), 'unchanged')
