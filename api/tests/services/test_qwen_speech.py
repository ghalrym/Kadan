import base64
import io
import os
from pathlib import Path
import threading
import types
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import soundfile as sf
import torch

from api.inference.qwen_speech import QwenSpeechProvider, QwenSpeechSession
from api.inference.resources import ResourceCancelled, ResourceManager
from api.inference.speech import SpeechInput, SpeechPlan, SpeechRegistry, SpeechRuntime
from api.services.qwen_tts_catalog import SPEECH_MODELS


class InProcessQwenTests(unittest.TestCase):
    def setUp(self):
        self.resources = ResourceManager(1000, {})
        self.provider = QwenSpeechProvider()
        registry = SpeechRegistry()
        registry.register(self.provider)
        self.runtime = SpeechRuntime(registry, lambda: self.resources)
        self.session = QwenSpeechSession(Path('/local'), 'cpu', 'fixture')
        self.request = SpeechInput('Hello', {'mode': 'describe', 'description': 'warm'}, model_id='qwen-tts-1.7b-design')
        self.cancel = threading.Event()
        self.loads = 0
        self.calls = []
        owner = self
        class Model:
            @classmethod
            def from_pretrained(cls, checkpoint, **options):
                owner.loads += 1
                owner.assertTrue(options['local_files_only'])
                owner.assertEqual(options['device_map'], 'cpu')
                owner.assertTrue(owner.resources.snapshot()['reservations'])
                model = cls()
                owner.model_ref = weakref.ref(model)
                return model
            def generate_voice_design(self, **options):
                owner.calls.append(('describe', options))
                return [np.zeros(240, dtype=np.float32)], 24000
            def generate_custom_voice(self, **options):
                owner.calls.append(('custom', options))
                return [np.zeros(240, dtype=np.float32)], 24000
            def generate_voice_clone(self, **options):
                owner.calls.append(('clone', options))
                return [np.zeros(240, dtype=np.float32)], 24000
        self.model_class = Model
        self.patches = [patch.dict('sys.modules', qwen_tts=types.SimpleNamespace(Qwen3TTSModel=Model)),
            patch.object(self.provider, 'enabled', return_value=True),
            patch.object(self.provider, 'prepare', return_value=SpeechPlan(('local',), 400, lambda: self.session))]
        for item in self.patches: item.start()
        self.addCleanup(lambda: [item.stop() for item in reversed(self.patches)])
        self.addCleanup(self.runtime.unload)

    def test_reuses_in_process_model_and_frees_before_accounting_release(self):
        for _ in range(2):
            result = self.runtime.generate(self.request, self.cancel)
            self.assertTrue(result.wav.startswith(b'RIFF'))
        self.assertEqual(self.loads, 1)
        state = next(iter(self.resources.snapshot()['reservations'].values()))
        self.assertEqual(state['active_leases'], 0)
        self.assertIsNotNone(self.model_ref())
        reservation = self.runtime._reservation
        release = reservation.release
        def checked_release():
            self.assertIsNone(self.model_ref())
            release()
        with patch.object(reservation, 'release', side_effect=checked_release):
            self.runtime.unload()
        self.assertEqual(self.resources.snapshot()['reservations'], {})

    def test_failed_generation_drops_tensor_tracebacks_before_releasing_memory(self):
        tensors = []
        def fail(**options):
            allocated = torch.zeros(32)
            tensors.append(weakref.ref(allocated))
            raise RuntimeError('failed after tensor allocation')
        original_reserve = self.resources.reserve
        def reserve(*args, **kwargs):
            reservation = original_reserve(*args, **kwargs)
            original_release = reservation.release
            def release():
                self.assertTrue(tensors)
                self.assertIsNone(tensors[0]())
                self.assertIsNone(self.model_ref())
                original_release()
            reservation.release = release
            return reservation
        with patch.object(self.resources, 'reserve', side_effect=reserve), \
             patch.object(self.model_class, 'generate_voice_design', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'tensor allocation'):
                self.runtime.generate(self.request, self.cancel)
        self.assertEqual(self.resources.snapshot()['reservations'], {})

    def test_all_voice_modes_and_clone_options(self):
        self.runtime.load(self.request, self.cancel)
        self.session.generate(self.request, self.cancel)
        self.session.generate(SpeechInput('Hi', {'mode':'custom','speaker':'Ryan','instruction':'calm'}), self.cancel)
        audio = io.BytesIO()
        sf.write(audio, np.zeros((240,2)), 24000, format='WAV')
        self.session.generate(SpeechInput('Hi', {'mode':'clone','sample':base64.b64encode(audio.getvalue()).decode(),'transcript':'hello','speaker_only':False}), self.cancel)
        self.assertEqual([mode for mode, _ in self.calls], ['describe','custom','clone'])
        self.assertEqual(self.calls[1][1]['instruct'], 'calm')
        self.assertEqual(self.calls[2][1]['ref_audio'][0].shape, (240,))
        self.assertEqual(self.calls[2][1]['ref_text'], 'hello')
        self.assertFalse(self.calls[2][1]['x_vector_only_mode'])

    def test_cooperative_cancel_discards_result_and_unloads_under_ownership(self):
        def generate(**options):
            self.assertTrue(self.resources.snapshot()['reservations'])
            self.cancel.set()
            options['stopping_criteria'](torch.ones((1,1),dtype=torch.long), None)
            return [np.zeros(240)], 24000
        with patch.object(self.model_class, 'generate_voice_design', side_effect=generate):
            with self.assertRaises(ResourceCancelled):
                self.runtime.generate(self.request, self.cancel)
        self.assertIsNone(self.session.model)
        self.assertEqual(self.resources.snapshot()['reservations'], {})

class QwenAdapterTests(unittest.TestCase):
    def test_validation_and_capabilities_stay_in_adapter(self):
        provider = QwenSpeechProvider()
        models = {item.id: item for item in provider.models()}
        self.assertTrue(models['qwen-tts-1.7b-custom'].supports_instruction)
        self.assertFalse(models['qwen-tts-0.6b-custom'].supports_instruction)
        self.assertIn('Ryan', models['qwen-tts-1.7b-custom'].speakers)
        self.assertEqual(models['qwen-tts-1.7b-custom'].default_speaker, 'Ryan')
        for request in (
            SpeechInput('Hi', {'mode': 'custom', 'speaker': 'Ryan', 'instruction': 'warm'}, model_id='qwen-tts-0.6b-custom'),
            SpeechInput('Hi', {'mode': 'custom', 'speaker': 'Ada'}, model_id='qwen-tts-1.7b-custom'),
            SpeechInput('Hi', {'mode': 'clone', 'sample': 'UklGRg=='}, model_id='qwen-tts-1.7b-base'),
            SpeechInput('Hi', {'mode': 'clone', 'sample': 'https://example.org', 'speaker_only': True}, model_id='qwen-tts-1.7b-base'),
        ):
            with self.subTest(request=request), self.assertRaises(ValueError):
                provider.validate(request)

    def test_prepare_checks_local_revision_and_estimates_host_and_device(self):
        model = SPEECH_MODELS['qwen-tts-1.7b-design']
        provider = QwenSpeechProvider()
        request = SpeechInput('Hi', {'mode': 'describe', 'description': 'warm'}, model_id=model.id)
        with patch.dict(os.environ, KADAN_QWEN_TTS_DEVICE='cuda:1'), \
             patch('api.inference.qwen_speech.model_manager.get_checkpoint', return_value=(model, Path('/local')), create=True):
            plan = provider.prepare(request)
            self.assertEqual(plan.host_bytes, model.estimated_bytes * 3 + 2 * 1024**3)
            self.assertEqual(plan.device_bytes, {1: plan.host_bytes})
            self.assertIn(model.revision, plan.identity)
            self.assertIsNone(plan.create().model)
