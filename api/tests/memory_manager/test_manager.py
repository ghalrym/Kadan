import asyncio
import base64
import os
from pathlib import Path
import tempfile
import time
import threading
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch, Mock

import httpx
from PIL import Image

from api.inference.resources import ResourceBusy, ResourceManager
from api.memory_manager import MemoryManager
from api.inference.feature import InferenceFeature
from api.inference.image.model import REVISION
from api.inference.resources import ResourceCancelled
from api.memory_manager.queue import InferenceQueue
from api.routes.v1.audio.transcriptions import TranscriptionRequest
from api.routes.v1.chat.completions import CompletionRequest
from api.routes.v1.images.generations import ImageRequest
from api.routes.v1.videos.generations import VideoGenerationRequest
from api.server import app
from api.services.video_jobs import VideoJobs
from api.services.images import ImageManager
from api.services.runtime import RuntimeFailure
from api.tests.inference.stt.test_model import audio_url


class Resident:
    """Fake tensors, real resource ownership and admission callbacks."""
    def __init__(self, resources, workload, events):
        self.resources, self.workload, self.events = resources, workload, events
        self.host = self.device = None
        self.weights = None
        self.constructed = 0

    def offload(self):
        self.events.append(self.workload + ':ram')
        self.device.release()
        self.device = None

    def evict(self):
        if self.device:
            self.offload()
        self.weights = None
        self.host.release()
        self.host = None

    def infer(self, cancel=None):
        if self.host is None:
            self.host = self.resources.reserve(self.workload + ':host', self.workload, host_bytes=20, evict=self.evict)
            self.weights = object()
            self.constructed += 1
        with self.host.lease(cancel):
            if self.device is None:
                self.device = self.resources.reserve(self.workload + ':gpu', self.workload,
                    device_bytes={0: 40}, evict=self.offload)
                self.events.append(self.workload + ':gpu')
            with self.device.lease(cancel):
                self.events.append(self.workload + ':run')
                snapshot = self.resources.snapshot()['reservations']
                assert not any(v['workload'] != self.workload and any(v['device_bytes'].values()) for v in snapshot.values())
        return self.weights


@unittest.skipUnless(os.getenv('KADAN_TEST_REDIS_URL'), 'Real Redis URL required')
class ManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.resources = ResourceManager(100, {0: 50})
        self.events = []
        self.llm = Resident(self.resources, 'llm', self.events)
        self.stt = Resident(self.resources, 'speech', self.events)

        async def complete(messages, model, on_event=None):
            self.assertEqual(model, 'selected-llm')
            self.llm.infer()
            if on_event:
                on_event({'content': 'reply'})
                on_event({'finish_reason': 'stop'})
            return 'reply'

        def transcribe(audio, model, language, cancel):
            self.assertEqual(model, 'tiny')
            self.stt.infer(cancel)
            return dict(text='heard', raw_text='heard', model=model, language='en')

        async def unload():
            pass
        self.runtime = SimpleNamespace(adapter=None, unload=unload, model_id='selected-llm', ensure_resources=lambda: self.resources, complete=complete)
        transcription = SimpleNamespace(native=None, selected=lambda: 'tiny', transcribe=transcribe, close=lambda: None,
            offload_to_ram=lambda cancel: self.resources.offload_workload_devices('speech', cancel))
        downloads = Mock()
        downloads.get_checkpoint.side_effect = ValueError("missing checkpoint")
        images = ImageManager(self.directory.name, downloads, self.runtime)
        self.manager = MemoryManager(runtime=self.runtime, transcription=transcription, images=images)
        await self.manager.queue.redis.aclose()
        self.prefix = f'kadan:test:{uuid.uuid4().hex}:'
        self.manager.queue = InferenceQueue(self.manager._execute, url=os.environ['KADAN_TEST_REDIS_URL'],
            lock_path=Path(self.directory.name) / 'lock', prefix=self.prefix)
        await self.manager.start()

    async def asyncTearDown(self):
        for resident in (self.llm, self.stt):
            if resident.host:
                resident.evict()
        redis = self.manager.queue.redis
        keys = [key async for key in redis.scan_iter(match=self.prefix + '*')]
        if keys:
            await redis.delete(*keys)
        await self.manager.close()
        self.directory.cleanup()

    async def test_six_stable_callables_and_serial_ram_gpu_handoff(self):
        features = [self.manager.llm, self.manager.video, self.manager.image, self.manager.stt,
                    self.manager.tts, self.manager.decisions]
        self.assertEqual(len({id(feature) for feature in features}), 6)
        self.assertTrue(all(isinstance(feature, InferenceFeature) and callable(feature) for feature in features))
        chat = CompletionRequest(messages=[{'role': 'user', 'text': 'Hello'}])
        self.assertEqual(await self.manager.submit(chat, feature='llm'), 'reply')
        first_weights = self.llm.weights
        self.assertEqual((await self.manager.submit(TranscriptionRequest(audio=audio_url()), feature='stt'))['text'], 'heard')
        self.assertIs(self.llm.weights, first_weights)
        self.assertIsNone(self.llm.device)
        self.assertEqual(await self.manager.submit(chat, feature='llm'), 'reply')
        self.assertIs(self.llm.weights, first_weights)
        self.assertEqual(self.llm.constructed, 1)
        self.assertEqual(self.events, ['llm:gpu', 'llm:run', 'llm:ram', 'speech:gpu', 'speech:run', 'speech:ram', 'llm:gpu', 'llm:run'])

    async def test_active_lease_cannot_be_moved_and_pressure_can_evict_ram(self):
        self.llm.infer()
        with self.llm.device.lease():
            with self.assertRaises(ResourceBusy):
                self.resources.offload_inactive_devices('video')
            self.assertIsNotNone(self.llm.weights)
        self.resources.offload_inactive_devices('video')
        host = self.resources.reserve('large', 'video', host_bytes=100)
        self.assertIsNone(self.llm.weights)
        host.release()
        self.llm.infer()
        self.assertEqual(self.llm.constructed, 2)

    async def test_missing_image_checkpoint_preserves_existing_resident(self):
        self.llm.infer()
        with self.assertRaisesRegex(RuntimeFailure, 'Download Qwen'):
            await self.manager.submit(ImageRequest(prompt='tree'), feature='image')
        self.assertIsNotNone(self.llm.device)
        self.assertEqual(await self.manager.queue.redis.scard(self.manager.queue.key('unfinished')), 0)

    async def test_image_generation_and_edit_use_redis_and_publish_real_png(self):
        service = self.manager.image.service
        service.downloads.get_checkpoint.side_effect = None
        service.downloads.get_checkpoint.return_value = (SimpleNamespace(revision=REVISION), Path(self.directory.name))
        service.backend = Mock(side_effect=lambda *args, **kwargs: [Image.new('RGBA', (2, 2))])
        with patch('api.routes.v1.images.generations.memory_manager', self.manager), patch(
                'api.routes.v1.images.edits.memory_manager', self.manager):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                response = await client.post('/v1/images/generations', json={'prompt':'tree', 'count':1, 'seed':7})
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()['image']
                self.assertEqual(result['seeds'], [7])
                source = 'data:image/png;base64,' + base64.b64encode(service.file(result['id'], 0).read_bytes()).decode()
                response = await client.post('/v1/images/edits', json={'prompt':'blue sky', 'count':1, 'image':source})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()['image']['mode'], 'Edit')
                self.assertEqual(service.backend.call_args.kwargs['image'].mode, 'RGBA')
        self.assertEqual(len(service.history()), 2)
        self.assertEqual(await self.manager.queue.redis.scard(self.manager.queue.key('unfinished')), 0)

    async def test_image_queue_cancel_waits_for_native_cleanup_before_retry(self):
        service = self.manager.image.service
        service.downloads.get_checkpoint.side_effect = None
        service.downloads.get_checkpoint.return_value = (SimpleNamespace(revision=REVISION), Path(self.directory.name))
        started, stopped = threading.Event(), threading.Event()
        def forward(path, resources, prompt, aspect, seeds, cancel, **kwargs):
            reservation = resources.reserve('image-fixture', 'image', host_bytes=1)
            try:
                with reservation.lease(cancel):
                    started.set()
                    cancel.wait(3)
                    raise ResourceCancelled('cancelled')
            finally:
                reservation.release()
                stopped.set()
        service.backend = forward
        task = asyncio.create_task(self.manager.submit(ImageRequest(prompt='tree', count=1), feature='image'))
        self.assertTrue(await asyncio.to_thread(started.wait, 3))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(stopped.is_set())
        self.assertEqual(service.history(), [])
        self.assertEqual(self.resources.snapshot()['reservations'], {})
        service.backend = lambda *args, **kwargs: [Image.new('RGB', (1, 1))]
        result = await self.manager.submit(ImageRequest(prompt='retry', count=1), feature='image')
        self.assertEqual(result['image']['prompt'], 'retry')

    async def test_invalid_durable_payload_fails_and_next_request_succeeds(self):
        job_id = await self.manager.queue.submit('llm', 'generate', {'messages': []}, 'selected-llm')
        with self.assertRaises(RuntimeFailure) as caught:
            await self.manager.queue.wait(job_id)
        self.assertEqual(caught.exception.status_code, 422)
        self.assertEqual(await self.manager.submit(CompletionRequest(messages=[{'role': 'user', 'text': 'Hi'}]), feature='llm'), 'reply')

    async def test_http_chat_and_transcription_preserve_response_shapes_through_redis(self):
        with patch('api.routes.v1.chat.completions.memory_manager', self.manager), patch(
                'api.routes.v1.audio.transcriptions.memory_manager', self.manager):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                chat = await client.post('/v1/chat/completions', json={'messages': [{'role': 'user', 'text': 'Hi'}]})
                self.assertEqual(chat.status_code, 200)
                self.assertEqual(chat.json()['message'], {'role': 'assistant', 'text': 'reply', 'meta': None})
                audio = await client.post('/v1/audio/transcriptions', json={'audio': audio_url(), 'formatting': False})
                self.assertEqual(audio.status_code, 200)
                self.assertEqual(audio.json(), dict(text='heard', raw_text='heard', language='en', model='tiny',
                    formatting_status='disabled', formatting_model=None))
                self.assertEqual((await client.get('/health')).status_code, 200)

    async def test_video_queues_behind_active_chat_and_delete_prevents_execution(self):
        started, release = asyncio.Event(), asyncio.Event()
        generated = []

        class Provider:
            def offload_to_ram(self, cancel=None):
                pass
            def validate(self, spec):
                pass
            def generate(self, spec, output, cancel):
                generated.append(spec.prompt)
                output.write_bytes(b'controlled output')

        async def complete(messages, model, on_event=None):
            started.set()
            await release.wait()
            if on_event:
                on_event({'content': 'reply'})
                on_event({'finish_reason': 'stop'})
            return 'reply'

        self.runtime.complete = complete
        videos = VideoJobs(Path(self.directory.name) / 'videos', factory=lambda _: Provider())
        self.manager.videos = videos
        with patch('api.routes.v1.chat.completions.memory_manager', self.manager), patch(
                'api.routes.v1.videos.generations.memory_manager', self.manager), patch(
                'api.routes.v1.videos.memory_manager', self.manager), patch('api.routes.v1.videos.video_jobs', videos):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                chat = asyncio.create_task(client.post('/v1/chat/completions', json={'messages': [{'role': 'user', 'text': 'Hi'}]}))
                await started.wait()
                response = await client.post('/v1/videos/generations', json={'prompt': 'test'})
                self.assertEqual(response.status_code, 202)
                job_id = response.json()['job']['id']
                self.assertEqual(response.json()['job']['status'], 'Queued')
                self.assertEqual(generated, [])
                self.assertEqual((await client.delete('/v1/videos/' + job_id)).status_code, 200)
                release.set()
                await chat
                self.assertEqual((await client.get('/v1/videos/' + job_id)).json()['job']['status'], 'Cancelled')
                self.assertEqual(generated, [])

    async def test_video_handoff_failure_is_visible_in_polling(self):
        provider = SimpleNamespace(validate=lambda spec: None, offload_to_ram=lambda cancel: None)
        self.manager.videos = VideoJobs(Path(self.directory.name) / 'videos', factory=lambda _: provider)
        self.llm.infer()
        with self.llm.device.lease():
            job = await self.manager.submit(VideoGenerationRequest(prompt='test'), feature='video')
            with self.assertRaises(RuntimeFailure):
                await self.manager.queue.wait(job.id)
            self.assertEqual((await self.manager.video_job(job.id)).status, 'Failed')

    async def test_slow_first_resource_discovery_does_not_block_event_loop(self):
        original = self.runtime.ensure_resources
        ticks = []
        def discover():
            time.sleep(.2)
            return original()
        async def tick():
            await asyncio.sleep(.02)
            ticks.append(time.monotonic())
        self.runtime.ensure_resources = discover
        start = time.monotonic()
        await asyncio.gather(self.manager.submit(CompletionRequest(messages=[{'role': 'user', 'text': 'Hi'}]), feature='llm'), tick())
        self.assertLess(ticks[0] - start, .15)
