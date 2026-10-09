import asyncio
import threading
import unittest
from unittest.mock import patch

from api.inference.request_execution import RequestExecutor
from api.inference.resources import ResourceManager, ResourceBusy
from api.inference.tts.speech_requests import SpeechRequests
from api.inference.tts.speech_runtime import SpeechRegistry, SpeechRuntime
from api.routes.v1.audio.speech import SpeechRequest
from api.services import speech
from api.tests.inference.tts.test_speech_runtime import AlternateProvider


class SpeechRequestsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.resources = ResourceManager(1000, {0: 1000})
        self.provider = AlternateProvider(self.resources)
        self.registry = SpeechRegistry()
        self.registry.register(self.provider)
        self.runtime = SpeechRuntime(self.registry, lambda: self.resources)
        self.feature = SpeechRequests(self.runtime)

    async def asyncTearDown(self):
        await self.feature.unload()

    async def test_common_contract_retains_same_adapter_and_protects_active_device(self):
        self.assertIsInstance(self.feature, RequestExecutor)
        await self.feature.load('alternate')
        self.assertIs(self.feature.adapter, self.provider)
        self.resources.offload_workload_devices('speech')
        self.assertIsNotNone(self.runtime._device_reservation)
        with self.runtime._device_reservation.lease():
            with self.assertRaises(ResourceBusy):
                await self.feature.offload_to_ram()
        await self.feature.offload_to_ram()
        self.assertIs(self.feature.adapter, self.provider)
        self.assertEqual(self.provider.loads, 1)
        self.assertIsNone(self.runtime._device_reservation)
        self.assertIsNotNone(self.runtime._reservation)
        with patch.object(speech, 'speech_runtime', self.runtime):
            request = SpeechRequest(script='Hello', language='Martian', voice={'mode':'custom','speaker':'Ada'})
        result = await self.feature(request, model='alternate')
        self.assertEqual(result['mime_type'], 'audio/wav')
        self.assertEqual(self.provider.loads, 1)
        await self.feature.unload()
        self.assertIsNone(self.feature.adapter)
        self.assertEqual(self.resources.snapshot()['reservations'], {})

    async def test_repeated_unload_cancellation_waits_for_inference_cleanup(self):
        await self.feature.load('alternate')
        started, release = threading.Event(), threading.Event()
        original = self.provider.unload
        def slow_unload():
            started.set()
            release.wait(5)
            original()
        with patch.object(self.provider, 'unload', side_effect=slow_unload):
            task = asyncio.create_task(self.feature.unload())
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                task.cancel()
                await asyncio.sleep(.01)
                task.cancel()
                await asyncio.sleep(.01)
                self.assertFalse(task.done())
                self.assertTrue(self.resources.snapshot()['reservations'])
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(self.resources.snapshot()['reservations'], {})
