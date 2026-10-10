"""Image worker errors and cancellation through the real Redis/public wrapper boundary."""
import asyncio
import os
from pathlib import Path
import tempfile
import threading
import unittest
import uuid
from unittest.mock import patch

from redis.asyncio import Redis

from api.inference.errors import InferenceFailure
from api.inference.image.image_requests import ImageRequests
from api.inference.image.native_worker import MODEL, NativeImageRuntime, NativeImageSession
from api.inference.line_protocol import LineProtocolError
from api.inference.resources import ResourceManager
from api.memory_manager.queue import InferenceQueue


class Child:
    def __init__(self):
        self.cancel_mode = False
        self.started = threading.Event()
        self.stopped = False
        self.before_stop = lambda: None

    def start(self, command, env=None):
        pass

    def read(self, timeout, cancel):
        return 'ready 1 1024'

    def exchange(self, command, timeout, cancel):
        self.started.set()
        if self.cancel_mode:
            if not cancel.wait(5):
                raise AssertionError('Cancellation was not delivered')
            raise InterruptedError('cancelled')
        raise LineProtocolError('Inference subprocess failed: image_generate_failed',
                                diagnostics=b'text_layer 30\nprivate-path-and-secret')

    def stop(self):
        self.before_stop()
        self.stopped = True


@unittest.skipUnless(os.getenv('KADAN_TEST_REDIS_URL'), 'Requires isolated Redis')
class ImageErrorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.environment = patch.dict(os.environ, {'KADAN_IMAGE_DEVICES': 'cpu'})
        self.environment.start()
        self.child = Child()
        self.resources = ResourceManager(96 * 1024**3, {})
        self.session = NativeImageSession('/unused-model', '/unused-worker', lambda: self.child)
        runtime = NativeImageRuntime(lambda: self.resources,
            lambda: (Path('/unused-model'), Path('/unused-worker')), lambda *args: self.session)
        self.wrapper = ImageRequests(runtime)
        async def execute(job):
            return await self.wrapper(self.wrapper.validate(job.payload, job.operation), model=job.model)
        self.queue = InferenceQueue(execute, url=os.environ['KADAN_TEST_REDIS_URL'],
            lock_path=Path(self.directory.name) / 'queue.lock', prefix='image-errors:' + uuid.uuid4().hex + ':')
        await self.queue.start()

    async def asyncTearDown(self):
        await self.queue.close(cleanup=self.wrapper.unload)
        # close() closes its Redis connection; use a new isolated client for cleanup.
        redis = Redis.from_url(os.environ['KADAN_TEST_REDIS_URL'], decode_responses=True)
        keys = [key async for key in redis.scan_iter(match=self.queue.prefix + '*')]
        if keys:
            await redis.delete(*keys)
        await redis.aclose()
        self.environment.stop()
        self.directory.cleanup()

    async def submit(self):
        return await self.queue.submit('image', 'generate',
            {'prompt': 'red ball', 'aspect': '1:1', 'count': 1, 'steps': 4, 'seed': 0}, MODEL)

    async def test_safe_stage_and_private_context_survive_cleanup(self):
        with self.assertLogs('api.inference.image.native_worker', level='ERROR') as captured:
            self.child.before_stop = lambda: self.assertIn('private-path-and-secret', '\n'.join(captured.output))
            job = await self.submit()
            with self.assertRaisesRegex(InferenceFailure, 'image_generate_failed') as raised:
                await self.queue.wait(job)
        record = await self.queue.get(job)
        self.assertEqual(record['state'], 'failed')
        self.assertIn('image_generate_failed', record['error'])
        self.assertNotIn('private-path', record['error'])
        self.assertNotIn('private-path', str(raised.exception))
        self.assertTrue(self.child.stopped)
        self.assertIsNone(self.session.workspace)
        self.assertEqual(self.resources.snapshot()['reservations'], {})

    async def test_cancel_waits_for_worker_cleanup(self):
        self.child.cancel_mode = True
        job = await self.submit()
        for _ in range(100):
            if self.child.started.is_set():
                break
            await asyncio.sleep(.01)
        self.assertTrue(self.child.started.is_set())
        await self.queue.cancel(job)
        with self.assertRaises(InferenceFailure):
            await self.queue.wait(job)
        self.assertEqual((await self.queue.get(job))['state'], 'cancelled')
        self.assertTrue(self.child.stopped)
        self.assertIsNone(self.session.workspace)
        self.assertEqual(self.resources.snapshot()['reservations'], {})
