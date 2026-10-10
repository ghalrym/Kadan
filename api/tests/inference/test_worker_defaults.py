"""Shipped routing chooses admitted native workers without opt-in flags."""
import asyncio
import hashlib
import json
import tempfile
import os
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch
from api.inference.errors import InferenceFailure
from api.inference.native_compute import DEFAULT_DEVICE_BYTES, native_compute
from api.inference.resources import ResourceManager
from api.inference.tts.native_worker import NativeSpeechSession
from api.inference.tts.qwen import QwenSpeechProvider
from api.inference.tts.speech_runtime import SpeechInput
from api.inference.stt.native_worker import NativeWhisper, resolve, HOST_BUDGET, MAX_PCM
from api.inference.stt.catalog import checkpoint
from api.inference.tts.speech_runtime import SpeechRegistry, SpeechRuntime
from api.inference.tts.speech_requests import SpeechRequests
from api.inference.stt.whisper_transcriber import WhisperTranscriber
from api.services.chat_runtime import ChatRuntime


class WorkerDefaultTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ,{},clear=True))
        self.resources=ResourceManager(128*1024**3,{0:DEFAULT_DEVICE_BYTES-1,1:DEFAULT_DEVICE_BYTES})
    def test_absent_or_empty_selection_uses_available_gpu(self):
        for feature in ('IMAGE','TTS','WHISPER'):
            for value in (None,''):
                with self.subTest(feature=feature,value=value):
                    if value is None:os.environ.pop('KADAN_'+feature+'_DEVICES',None)
                    else:os.environ['KADAN_'+feature+'_DEVICES']=value
                    self.assertEqual(native_compute(feature,self.resources).devices,(1,))
    def test_cpu_requires_explicit_selection_or_no_admitted_gpu(self):
        with patch.dict(os.environ,{'KADAN_TTS_DEVICES':'cpu'}):
            self.assertEqual(native_compute('TTS',self.resources).devices,())
        self.assertEqual(native_compute('TTS',ResourceManager(1024,{})).devices,())
        with self.assertRaises(InferenceFailure):native_compute('TTS')
    def test_default_tts_provider_uses_installed_worker_and_shared_placement(self):
        model=QwenSpeechProvider().models()[0]
        request=SpeechInput('Hello',{'mode':'custom','speaker':'Ryan'},'Auto',model.id)
        store=SimpleNamespace(root=Path('/models'),get_checkpoint=Mock(return_value=(SimpleNamespace(revision='0c0e3051f131929182e2c023b9537f8b1c68adfe'),Path('/checkpoint'))))
        with patch('api.inference.tts.qwen.model_manager',store),patch.object(Path,'is_file',return_value=True),patch('api.inference.tts.qwen.verify_tokenizer') as verify:
            plan=QwenSpeechProvider().prepare(request,self.resources)
        worker=plan.create()
        self.assertIsInstance(worker,NativeSpeechSession)
        self.assertEqual(worker.binary,Path('/opt/kadan/bin/kadan-tts-worker'))
        self.assertEqual(plan.device_bytes,{1:DEFAULT_DEVICE_BYTES})
        verify.assert_called_once_with(Path('/checkpoint'),Path('/models/native/tts/tokenizer.json'))
    def test_default_whisper_never_selects_python_model(self):
        self.assertIsInstance(WhisperTranscriber()._cpp,NativeWhisper)
    def test_default_chat_constructs_resident_worker(self):
        runtime=ChatRuntime();runtime.resources=self.resources
        adapter=Mock()
        with patch('api.inference.llm.qwen_residency.build_resident_qwen',return_value=adapter) as build:
            self.assertIs(runtime._construct(SimpleNamespace(),Path('/checkpoint'),threading.Event()),adapter)
        build.assert_called_once();self.assertEqual(build.call_args.kwargs['device'],'auto')
        adapter.configure_context.assert_called_once()
    def test_python_chat_backend_is_not_an_inference_fallback(self):
        with patch.dict(os.environ,{'KADAN_LLM_BACKEND':'python'}):
            with self.assertRaisesRegex(InferenceFailure,'native worker'):
                ChatRuntime()._construct(SimpleNamespace(),Path('/checkpoint'),threading.Event())

    def test_whisper_default_paths_resolve_and_verify_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); model=root/'native/whisper/tiny';assets=model/'assets';assets.mkdir(parents=True)
            weight=b'fixture weights';tokens=b'fixture tokens';mel=b'fixture mel'
            digest=lambda data:hashlib.sha256(data).hexdigest()
            dimensions=dict(n_mels=80,n_audio_ctx=1500,n_audio_state=384,n_audio_head=6,n_audio_layer=4,n_vocab=51865,n_text_ctx=448,n_text_state=384,n_text_head=6,n_text_layer=4)
            (model/'dimensions.txt').write_text(' '.join(map(str,dimensions.values()))+'\n')
            (model/'model.safetensors').write_bytes(weight)
            (model/'export.json').write_text(json.dumps({'source_sha256':checkpoint('tiny').sha256,
                'output_bytes':len(weight),'output_sha256':digest(weight),'dimensions':dimensions}))
            (assets/'english.tokens').write_bytes(tokens);(assets/'mel-80.f32').write_bytes(mel)
            (assets/'assets.json').write_text(json.dumps({'language':'en','whisper_version':'20250625','vocabulary':51865,
                'token_sha256':digest(tokens),'mel_sha256':{'80':digest(mel)}}))
            with patch('api.inference.stt.native_worker.model_manager',SimpleNamespace(root=root)),patch(
                    'api.inference.stt.native_worker.subprocess.run',return_value=SimpleNamespace(stdout=f'whisper 1 cpu en {MAX_PCM} {HOST_BUDGET}\n'.encode())) as capabilities:
                result=resolve('tiny')
                for key in dimensions:
                    altered=dimensions.copy();altered[key]+=1
                    (model/'dimensions.txt').write_text(' '.join(map(str,altered.values()))+'\n')
                    with self.subTest(dimension=key),self.assertRaisesRegex(InferenceFailure,'dimensions'):
                        resolve('tiny')
                (model/'dimensions.txt').unlink()
                (root/'linked-dimensions.txt').write_text(' '.join(map(str,dimensions.values()))+'\n')
                (model/'dimensions.txt').symlink_to(root/'linked-dimensions.txt')
                with self.assertRaisesRegex(InferenceFailure,'regular file'):resolve('tiny')
            self.assertEqual(result,['/opt/kadan/bin/kadan-whisper-worker',str(model),'model.safetensors',str(model/'dimensions.txt'),str(assets)])
            self.assertEqual(capabilities.call_count,12)

    def test_speech_wrapper_preload_allows_empty_script_without_generation(self):
        provider=QwenSpeechProvider();session=Mock()
        registry=SpeechRegistry();registry.register(provider)
        runtime=SpeechRuntime(registry, resources=lambda: self.resources)
        wrapper=SpeechRequests(runtime)
        store=SimpleNamespace(root=Path('/models'),get_checkpoint=Mock(return_value=(SimpleNamespace(revision='0c0e3051f131929182e2c023b9537f8b1c68adfe'),Path('/checkpoint'))))
        with patch('api.inference.tts.qwen.model_manager',store),patch.object(Path,'is_file',return_value=True),patch(
                'api.inference.tts.qwen.verify_tokenizer'),patch('api.inference.tts.qwen.ensure_assets'),patch('api.inference.tts.qwen.NativeSpeechSession',return_value=session):
            self.assertIs(asyncio.run(wrapper.load('qwen-tts-1.7b-custom')),session)
            asyncio.run(wrapper.load('qwen-tts-1.7b-custom'))
            session.load.assert_called_once();session.generate.assert_not_called()
            asyncio.run(wrapper.unload())
        self.assertEqual(self.resources.snapshot()['reservations'],{})
