"""Normal HTTP image contract through production manager, FIFO and admission code."""
import asyncio
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch

import httpx
from redis.asyncio import Redis

from api.inference.image.native_worker import NativeImageRuntime, NativeImageSession
from api.inference.resources import MemoryCapacity, ResourceManager
from api.memory_manager import MemoryManager
from api.memory_manager.queue import InferenceQueue
from api.server import app
from api.tests.inference.image.test_native_worker import Child

GIB = 1024**3


@unittest.skipUnless(os.getenv('KADAN_TEST_REDIS_URL'), 'Requires isolated Redis')
class ImageHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.events = []
        self.chat_started, self.chat_release = threading.Event(), threading.Event()
        self.park_started, self.park_ack = threading.Event(), threading.Event()
        self.available = MemoryCapacity(96 * GIB, {0: 1024})
        self.resources = ResourceManager(96 * GIB, {0: 1024}, probe=lambda: self.available)
        def park():
            self.events.append('park_requested')
            self.park_started.set()
            if not self.park_ack.wait(5):
                raise AssertionError('Missing fake park acknowledgement')
            self.events.append('park_acknowledged')
        self.resident = self.resources.reserve('existing-qwen', 'llm', device_bytes={0: 1024}, evict=park)
        def generate():
            with self.resident.lease():
                self.chat_started.set()
                if not self.chat_release.wait(5):
                    raise AssertionError('Missing fake chat completion')
            self.events.append('chat_completed')
            return 'reply'
        async def complete(messages, model, on_event=None):
            text = await asyncio.to_thread(generate)
            if on_event:
                on_event({'finish_reason': 'stop'})
            return text
        async def unload():
            self.resident.release()
        runtime = SimpleNamespace(adapter=None, model_id='large', complete=complete,
                                  unload=unload, ensure_resources=lambda: self.resources)
        self.child = Child()
        original = self.child.exchange
        def exchange(*args, **kwargs):
            self.events.append('image_generate')
            return original(*args, **kwargs)
        self.child.exchange = exchange
        self.session = NativeImageSession('/unused-model', '/unused-worker', lambda: self.child)
        image_runtime = NativeImageRuntime(runtime.ensure_resources,
            lambda: (Path('/unused-model'), Path('/unused-worker')), lambda *args: self.session)
        self.environment = patch.dict(os.environ, {'KADAN_IMAGE_DEVICES': 'cpu'})
        self.environment.start()
        # Only worker construction/asset existence are replaced. Manager, wrappers,
        # HTTP contract, Redis, lock and shared admission implementation remain real.
        self.factory = patch('api.inference.image.image_requests.NativeImageRuntime', return_value=image_runtime)
        self.factory.start()
        self.assets = patch('api.inference.image.image_requests.prepare', return_value=None)
        self.assets.start()
        self.manager = MemoryManager(runtime=runtime)
        await self.manager.queue.redis.aclose()
        self.lock = Path(self.directory.name) / 'inference.lock'
        self.manager.queue = InferenceQueue(self.manager._execute, url=os.environ['KADAN_TEST_REDIS_URL'],
            lock_path=self.lock, prefix='handoff:' + uuid.uuid4().hex + ':')
        self.assertTrue(await self.manager.start())
        self.routes = [patch('api.routes.v1.chat.completions.memory_manager', self.manager),
                       patch('api.routes.v1.images.generations.memory_manager', self.manager)]
        for route in self.routes:
            route.start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test')

    async def asyncTearDown(self):
        self.chat_release.set()
        self.park_ack.set()
        await self.client.aclose()
        await self.manager.close()
        redis = Redis.from_url(os.environ['KADAN_TEST_REDIS_URL'], decode_responses=True)
        keys = [key async for key in redis.scan_iter(match=self.manager.queue.prefix + '*')]
        if keys:
            await redis.delete(*keys)
        await redis.aclose()
        for route in self.routes:
            route.stop()
        self.assets.stop()
        self.factory.stop()
        self.environment.stop()
        self.directory.cleanup()

    async def wait_for(self, event):
        self.assertTrue(await asyncio.to_thread(event.wait, 3))

    async def image(self):
        return await self.client.post('/v1/images/generations', json={
            'prompt': 'A red ball on a white background.', 'aspect': '1:1', 'count': 1, 'steps': 50, 'seed': 0})

    async def test_fifo_waits_for_active_chat_then_acknowledged_resident_handoff(self):
        chat = asyncio.create_task(self.client.post('/v1/chat/completions', json={
            'messages': [{'role': 'user', 'text': 'Hello'}]}))
        await self.wait_for(self.chat_started)
        image = asyncio.create_task(self.image())
        for _ in range(100):
            if await self.manager.queue.redis.llen(self.manager.queue.key('pending')):
                break
            await asyncio.sleep(.01)
        self.assertEqual(await self.manager.queue.redis.llen(self.manager.queue.key('pending')), 1)
        self.assertEqual(self.child.starts, 0)
        self.assertFalse(self.park_started.is_set())
        self.chat_release.set()
        self.assertEqual((await chat).status_code, 200)
        await self.wait_for(self.park_started)
        self.assertFalse(image.done())
        self.assertEqual(self.child.starts, 0)
        self.assertIn('existing-qwen', self.resources.snapshot()['reservations'])
        self.park_ack.set()
        response = await image
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn('2048×2048', response.json()['image']['meta'])
        self.assertEqual(self.events, ['chat_completed', 'park_requested', 'park_acknowledged', 'image_generate'])
        await self.manager.image.unload()
        self.assertEqual(self.child.stops, 1)
        self.assertIsNone(self.session.workspace)
        self.assertEqual(self.resources.snapshot()['reservations'], {})

    async def test_physical_pressure_prevents_child_start(self):
        self.park_ack.set()
        self.available = MemoryCapacity(1, {0: 0})
        response = await self.image()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.child.starts, 0)
        self.assertNotIn('image_generate', self.events)
