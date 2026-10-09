"""Behavioral contract for six heterogeneous native feature objects."""
import asyncio
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from api.inference.feature import InferenceFeature
from api.inference.llm.feature import LLMFeature
from api.inference.video.feature import VideoFeature
from api.inference.video.h3 import H3Provider, H3_REVISION, GIB
from api.inference.image.feature import ImageFeature
from api.inference.stt.feature import STTFeature
from api.inference.stt.model import TranscriptionManager
from api.inference.stt.catalog import checkpoint
from api.inference.tts.feature import TTSFeature
from api.inference.decisions.decision_requests import DecisionRequests
from api.inference.decisions.laya_python import LayaPythonEvaluator
from api.inference.resources import ResourceManager, ResourceBusy
from api.services.runtime import RuntimeFailure
from api.services.video_jobs import VideoJobs
from api.tests.inference.stt.test_model import audio_url
from api.tests.inference.decisions.test_laya_python import questions
from api.routes.v1.chat.completions import CompletionRequest
from api.routes.v1.audio.transcriptions import TranscriptionRequest
from api.routes.v1.decisions import DecisionRequest
from api.routes.v1.videos.generations import VideoGenerationRequest


class FeatureContractTests(unittest.IsolatedAsyncioTestCase):
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
        feature = LLMFeature(service)
        self.assertIsInstance(feature, InferenceFeature)
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

    async def test_whisper_load_call_park_reload_and_unload_keep_one_native_object(self):
        resources = ResourceManager(8 * GIB, {0: 8 * GIB})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'tiny.pt').write_bytes(b'fixture')
            entry = replace(checkpoint('tiny'), sha256=hashlib.sha256(b'fixture').hexdigest())
            native = Mock()
            native.modules.return_value = []
            native.transcribe.return_value = {'text':'heard','language':'en'}
            factory = Mock(return_value=native)
            store = SimpleNamespace(root=root, get_checkpoint=lambda _: (None, root))
            service = TranscriptionManager(factory, resources, store)
            feature = STTFeature(service)
            self.assertIsInstance(feature, InferenceFeature)
            with patch('api.inference.stt.model.checkpoint', return_value=entry), patch.dict(
                    'os.environ', {'KADAN_WHISPER_DEVICE':'cuda:0'}):
                self.assertIs(await feature.load('tiny'), native)
                result = await feature(TranscriptionRequest(audio=audio_url(), formatting=False), model='tiny')
                self.assertEqual(result['text'], 'heard')
                with service.device_reservation.lease():
                    with self.assertRaises(ResourceBusy):
                        await feature.offload_to_ram()
                await feature.offload_to_ram()
                self.assertEqual(service.device, 'cpu')
                self.assertIs(feature.adapter, native)
                await feature.load('tiny')
                self.assertEqual(service.device, 'cuda:0')
                factory.assert_called_once()
                await feature.unload()
                self.assertIsNone(feature.adapter)
                self.assertEqual(resources.snapshot()['reservations'], {})

    async def test_cpu_decisions_load_is_prediction_free_and_offload_preserves_agent(self):
        resources = ResourceManager(100, {})
        native = Mock()
        native.predict.return_value = {'answers':{'urgent':{'type':'noul','noul':.7}},'usage':{}}
        loader = Mock(return_value=native)
        feature = DecisionRequests(LayaPythonEvaluator(loader, resources, check=Mock(), ram_bytes=60))
        self.assertIsInstance(feature, InferenceFeature)
        self.assertIs(await feature.load(), native)
        native.predict.assert_not_called()
        await feature.offload_to_ram()
        self.assertIs(feature.adapter, native)
        result = await feature(DecisionRequest(state='state', questions=[questions()[2]]))
        self.assertEqual(result[0]['value'], .7)
        loader.assert_called_once()
        await feature.unload()
        self.assertIsNone(feature.adapter)
        self.assertEqual(resources.snapshot()['reservations'], {})

    async def test_video_explicit_load_park_generation_and_unload_share_provider(self):
        resources = ResourceManager(400 * GIB, {0:18 * GIB})
        entry = SimpleNamespace(revision=H3_REVISION, estimated_bytes=69_000_000_000)
        provider = H3Provider()
        session = Mock()
        with tempfile.TemporaryDirectory() as directory, patch(
                'api.inference.video.h3.check_media_tools'), patch(
                'api.inference.video.h3.model_manager.get_checkpoint', return_value=(entry, Path('/fixture'))), patch(
                'api.inference.video.h3.runtime.ensure_resources', return_value=resources), patch(
                'api.inference.video.h3_pipeline.H3Session', return_value=session) as construct:
            jobs = VideoJobs(directory, factory=lambda _: provider)
            feature = VideoFeature(jobs)
            self.assertIsInstance(feature, InferenceFeature)
            self.assertIs(await feature.load('h3-fl2va-int8-turbo'), provider)
            session.load.assert_called_once()
            session.render.assert_not_called()
            await feature.offload_to_ram()
            session.park.assert_called_once_with(GIB)
            def render(spec, checkpoint, output, cancellation, devices):
                Path(output).write_bytes(b'controlled video')
            with patch.object(provider, '_run', side_effect=render):
                result = await feature(VideoGenerationRequest(model='h3-fl2va-int8-turbo', prompt='ball',
                    duration=4, resolution='480p'), model='h3-fl2va-int8-turbo', job_id='a'*32)
            self.assertEqual(result['status'], 'Done')
            construct.assert_called_once()
            self.assertIs(feature.adapter, provider)
            await feature.unload()
            self.assertIsNone(feature.adapter)
            session.close.assert_called_once()
            self.assertEqual(resources.snapshot()['reservations'], {})
            self.assertEqual(jobs.get('a'*32).status, 'Done')

    async def test_image_contract_is_honestly_unsupported(self):
        for feature in (ImageFeature(),):
            with self.subTest(feature=feature.name):
                self.assertIsInstance(feature, InferenceFeature)
                with self.assertRaises(RuntimeFailure):
                    await feature.load('unimplemented')
                with self.assertRaises(RuntimeFailure):
                    await feature(object())
                await feature.offload_to_ram()
                await feature.unload()
                self.assertIsNone(feature.adapter)
