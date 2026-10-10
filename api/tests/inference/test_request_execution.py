"""Behavioral contract for six heterogeneous native feature objects."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from api.inference.request_execution import RequestExecutor
from api.inference.llm.chat_requests import ChatRequests
from api.inference.video.video_requests import VideoRequests
from api.inference.video.h3_worker import H3Provider, GIB
from api.tests.inference.video.test_h3_worker import Worker
from api.inference.image.image_requests import ImageRequests
from api.inference.stt.transcription_requests import TranscriptionRequests
from api.inference.stt.whisper_transcriber import WhisperTranscriber
from api.inference.stt.native_worker import NativeWhisper
from api.tests.inference.stt.test_native_worker import Child
from api.inference.resources import ResourceManager, ResourceBusy
from api.inference.errors import InferenceFailure
from api.services.video_jobs import VideoJobs
from api.tests.inference.stt.test_whisper_transcriber import audio_url
from api.routes.v1.chat.completions import CompletionRequest
from api.routes.v1.audio.transcriptions import TranscriptionRequest
from api.routes.v1.videos.generations import VideoGenerationRequest


class RequestExecutorContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_uses_selected_adapter_and_parks_only_its_device_allocations(self):
        resources = ResourceManager(100, {0: 50})
        host = resources.reserve('host', 'llm', host_bytes=20)
        device = resources.reserve('device', 'llm', device_bytes={0: 30}, evict=lambda: None)
        adapter = object()
        async def load(model):
            service.adapter, service.model_id, service.state = adapter, model, 'ready'
        async def complete(messages, model):
            self.assertIs(service.adapter, adapter)
            self.assertEqual(model, 'chat')
            return messages[0].text
        async def unload():
            host.release()
            service.adapter = None
        service = SimpleNamespace(adapter=None, state='unloaded', task=None, model_id=None,
            load=load, complete=complete, unload=unload, ensure_resources=lambda: resources)
        feature = ChatRequests(service)
        self.assertIsInstance(feature, RequestExecutor)
        self.assertIs(await feature.load('chat'), adapter)
        self.assertEqual(await feature(CompletionRequest(messages=[{'role':'user','text':'hello'}]), model='chat'), 'hello')
        with device.lease():
            with self.assertRaises(ResourceBusy):
                await feature.offload_to_ram()
        await feature.offload_to_ram()
        self.assertEqual(set(resources.snapshot()['reservations']), {'host'})
        self.assertIs(feature.adapter, adapter)
        await feature.unload()
        self.assertIsNone(feature.adapter)

    async def test_whisper_load_call_park_reload_and_unload_keep_one_inference_object(self):
        resources = ResourceManager(64 * GIB, {0: 8 * GIB})
        with tempfile.TemporaryDirectory() as directory:
            class ParkableChild(Child):
                def exchange(self, command, *args):
                    if command in ('park', 'resume'):
                        return {'park': 'parked', 'resume': 'resumed'}[command]
                    return super().exchange(command, *args)
            child = ParkableChild()
            resolver = Mock(return_value=['/native', '/model', 'weights', 'dimensions', 'assets'])
            native = NativeWhisper(resources, resolver, lambda: child)
            service = WhisperTranscriber(worker=native, store=SimpleNamespace(root=Path(directory)))
            feature = TranscriptionRequests(service)
            self.assertIsInstance(feature, RequestExecutor)
            with patch.dict('os.environ', {'KADAN_WHISPER_DEVICES':'0'}):
                self.assertIs(await feature.load('tiny'), native)
                result = await feature(TranscriptionRequest(audio=audio_url(), formatting=False), model='tiny')
                self.assertEqual(result['text'], ' Hello Andrew!')
                with native.device_admission.lease():
                    with self.assertRaises(ResourceBusy):
                        await feature.offload_to_ram()
                await feature.offload_to_ram()
                self.assertTrue(native.parked)
                self.assertIsNone(native.device_admission)
                self.assertIs(feature.adapter, native)
                await feature.load('tiny')
                self.assertFalse(native.parked)
                self.assertIsNotNone(native.device_admission)
                self.assertEqual(child.starts, 1)
                resolver.assert_called_once_with('tiny')
                await feature.unload()
                self.assertIs(feature.adapter, native)
                self.assertIsNone(native.child)
                self.assertEqual(resources.snapshot()['reservations'], {})

    async def test_video_explicit_load_park_generation_and_unload_share_provider(self):
        resources = ResourceManager(200 * GIB, {0: 8 * GIB, 1: 8 * GIB})
        children = []
        def child():
            worker = Worker()
            children.append(worker)
            return worker
        provider = H3Provider(resources=resources, resolve=lambda _: ['worker'], process_factory=child)
        def encode(raw, target, frames, cancel):
            target.write_bytes(b'controlled video')
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                'os.environ', {'KADAN_H3_DEVICES': '0,1'}), patch.object(provider, '_encode', encode):
            jobs = VideoJobs(directory, factory=lambda _: provider)
            feature = VideoRequests(jobs)
            self.assertIsInstance(feature, RequestExecutor)
            self.assertIs(await feature.load('h3-fl2va-int8-turbo'), provider)
            self.assertTrue(children[0].started)
            self.assertFalse(children[0].calls)
            await feature.offload_to_ram()
            self.assertTrue(children[0].stopped)
            self.assertEqual(resources.snapshot()['reservations'], {})
            result = await feature(VideoGenerationRequest(model='h3-fl2va-int8-turbo', prompt='ball',
                duration=4, resolution='480p'), model='h3-fl2va-int8-turbo', job_id='a'*32)
            self.assertEqual(result['status'], 'Done')
            self.assertEqual(len(children), 2)
            self.assertEqual(len(children[1].calls), 1)
            self.assertIs(feature.adapter, provider)
            await feature.unload()
            self.assertIsNone(feature.adapter)
            self.assertTrue(children[1].stopped)
            self.assertEqual(resources.snapshot()['reservations'], {})
            self.assertEqual(jobs.get('a'*32).status, 'Done')

    async def test_image_contract_is_honestly_unsupported(self):
        for feature in (ImageRequests(),):
            with self.subTest(feature=feature.name):
                self.assertIsInstance(feature, RequestExecutor)
                with self.assertRaises(InferenceFailure):
                    await feature.load('unimplemented')
                with self.assertRaises(InferenceFailure):
                    await feature(object())
                await feature.offload_to_ram()
                await feature.unload()
                self.assertIsNone(feature.adapter)
