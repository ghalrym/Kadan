import os
import asyncio
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

from api.inference.errors import InferenceFailure
from api.inference.native_compute import NativeCompute, DEFAULT_DEVICE_BYTES, native_compute
from api.inference.resources import MemoryCapacity, ResourceManager
from api.inference.image.image_requests import ImageRequests
from api.inference.image.native_worker import NativeImageRuntime, NativeImageSession
from api.inference.stt.native_worker import NativeWhisper
from api.inference.tts.native_worker import NativeSpeechSession, PROCESS_BUDGET as SPEECH_BUDGET
from api.inference.tts.speech_runtime import SpeechInput, SpeechModel, SpeechPlan, SpeechRegistry, SpeechRuntime
from api.tests.inference.image.test_native_worker import Child as ImageChild
from api.tests.inference.stt.test_native_worker import Child as WhisperChild, audio
from api.tests.inference.tts.test_native_worker import Child as SpeechChild


class Controls:
    def start(self, command, env=None):
        self.environment = env
        self.controls = []
        self.bad_park = False
        super().start(command, env)

    def exchange(self, command, timeout, cancel=None):
        if command in ('park', 'resume'):
            self.controls.append(command)
            return 'unconfirmed' if self.bad_park else {'park': 'parked', 'resume': 'resumed'}[command]
        return super().exchange(command, timeout, cancel)


class ImageProcess(Controls, ImageChild): pass
class WhisperProcess(Controls, WhisperChild): pass
class SpeechProcess(Controls, SpeechChild): pass


class ComputeTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'KADAN_IMAGE_DEVICES': '0,1', 'KADAN_TTS_DEVICES': '0,1',
            'KADAN_WHISPER_DEVICES': '0,1', 'KADAN_NATIVE_GPU_BUDGET_BYTES': str(DEFAULT_DEVICE_BYTES)})
        self.env.start();self.addCleanup(self.env.stop)
        self.resources = ResourceManager(128*1024**3, {0: DEFAULT_DEVICE_BYTES, 1: DEFAULT_DEVICE_BYTES})

    def test_auto_and_explicit_placement(self):
        with patch.dict(os.environ, {'KADAN_IMAGE_DEVICES': 'auto'}):
            self.assertEqual(native_compute('IMAGE', self.resources).devices, (0,1))
            self.assertEqual(native_compute('IMAGE', ResourceManager(1024,{1:DEFAULT_DEVICE_BYTES})).devices,(1,))
            self.assertEqual(native_compute('IMAGE', ResourceManager(1024,{})).devices,())
        for value in ('0,0','2','1,2','-1','0,'):
            with patch.dict(os.environ, {'KADAN_IMAGE_DEVICES': value}), self.assertRaises(InferenceFailure): native_compute('IMAGE', self.resources)
        with self.assertRaises(InferenceFailure): native_compute('IMAGE', ResourceManager(1024,{}))

    def test_auto_placement_waits_for_feasible_devices_despite_temporary_pressure(self):
        resources = ResourceManager(128*1024**3, {0: DEFAULT_DEVICE_BYTES, 1: DEFAULT_DEVICE_BYTES},
            probe=lambda: MemoryCapacity(128*1024**3, {0: 0, 1: 0}))
        with patch.dict(os.environ, {'KADAN_IMAGE_DEVICES': 'auto'}):
            self.assertEqual(native_compute('IMAGE', resources).devices, (0, 1))

    def test_media_default_uses_available_headroom_for_retention(self):
        with patch.dict(os.environ, {'KADAN_IMAGE_DEVICES': '0,1'}, clear=True):
            resources=ResourceManager(128*1024**3,{0:22*1024**3,1:20*1024**3})
            plan=native_compute('IMAGE',resources)
            self.assertEqual(plan.device_bytes,{0:22*1024**3-64*1024**2,1:20*1024**3-64*1024**2})
            self.assertEqual(plan.environment()['KADAN_NATIVE_GPU_BUDGETS'],f'0:{22*1024**3-64*1024**2},1:{20*1024**3-64*1024**2}')
            self.assertEqual(native_compute('TTS',resources).device_bytes,plan.device_bytes)
            self.assertEqual(native_compute('WHISPER',resources).device_bytes,plan.device_bytes)
            self.assertEqual(native_compute('IMAGE',ResourceManager(128*1024**3,{0:40*1024**3,1:40*1024**3})).budget,24*1024**3)
            with patch.dict(os.environ, {'KADAN_NATIVE_GPU_BUDGET_BYTES':str(DEFAULT_DEVICE_BYTES)}):
                self.assertEqual(native_compute('IMAGE',resources).budget,DEFAULT_DEVICE_BYTES)

    def test_automatic_headroom_accounts_for_external_use_and_reclaim(self):
        gib = 1024**3
        resources = ResourceManager(128*gib, {0:24*gib, 1:24*gib},
            probe=lambda: MemoryCapacity(128*gib, {0:7*gib, 1:13*gib}))
        with patch.dict(os.environ, {}, clear=True):
            for feature in ('IMAGE', 'WHISPER', 'TTS'):
                self.assertEqual(native_compute(feature, resources).device_bytes,
                                 {0:7*gib-64*1024**2, 1:13*gib-64*1024**2})

    def test_frozen_environment(self):
        plan=native_compute('IMAGE',self.resources)
        with patch.dict(os.environ, {'KADAN_IMAGE_DEVICES': '1', 'KADAN_NATIVE_GPU_BUDGET_BYTES':'1'}):
            self.assertEqual(plan.environment()['KADAN_IMAGE_DEVICES'],'0,1')
            self.assertEqual(plan.environment()['KADAN_NATIVE_GPU_BUDGET_BYTES'],str(DEFAULT_DEVICE_BYTES))

    def owners(self):
        image_child=ImageProcess();image_session=NativeImageSession('/model','/worker',lambda:image_child)
        image=NativeImageRuntime(lambda:self.resources,lambda:(Path('/model'),Path('/worker')),lambda *args:image_session)
        whisper_child=WhisperProcess();whisper=NativeWhisper(self.resources,lambda model:['/worker','/model','weights','dims','assets'],lambda:whisper_child)
        self.addCleanup(image.unload);self.addCleanup(whisper.close)
        return image,image_child,whisper,whisper_child

    def test_shared_handoff_parks_preserves_ram_and_resumes_same_child(self):
        image,ic,whisper,wc=self.owners();request=SimpleNamespace(prompt='ball',aspect='1:1',count=1,seed=0)
        image.run(request,threading.Event());whisper.transcribe(audio(),'large-v3')
        self.assertEqual(ic.controls,['park']);self.assertEqual(ic.stops,0)
        self.assertEqual(ic.environment['KADAN_IMAGE_DEVICES'],'0,1');self.assertEqual(wc.environment['KADAN_WHISPER_DEVICES'],'0,1')
        image.run(request,threading.Event());self.assertEqual(wc.controls,['park']);self.assertEqual(ic.controls,['park','resume']);self.assertEqual(ic.starts,1)
        image.unload();whisper.close();self.assertEqual(self.resources.snapshot()['reservations'],{})

    def test_explicit_image_offload_acknowledges_before_releasing_devices(self):
        image, child, _, _ = self.owners()
        request=SimpleNamespace(prompt='ball',aspect='1:1',count=1,seed=0)
        image.run(request,threading.Event())
        wrapper=ImageRequests(image)
        child.bad_park=True
        with self.assertRaises(Exception):
            asyncio.run(wrapper.offload_to_ram())
        self.assertTrue(any(v['device_bytes'] for v in self.resources.snapshot()['reservations'].values()))
        child.bad_park=False
        asyncio.run(wrapper.offload_to_ram())
        reservations=self.resources.snapshot()['reservations']
        self.assertFalse(any(v['device_bytes'] for v in reservations.values()))
        self.assertTrue(any(v['host_bytes'] for v in reservations.values()))
        self.assertEqual(child.stops,0)
        image.run(request,threading.Event())
        self.assertEqual(child.starts,1)
        self.assertEqual(child.controls[-1],'resume')

    def test_failed_park_keeps_gpu_reservation(self):
        image,ic,whisper,wc=self.owners();image.run(SimpleNamespace(prompt='ball',aspect='1:1',count=1,seed=0),threading.Event());ic.bad_park=True
        with self.assertRaises(Exception): whisper.transcribe(audio(),'large-v3')
        self.assertEqual(wc.starts,0)
        self.assertTrue(any(v['device_bytes'] for v in self.resources.snapshot()['reservations'].values()))
        ic.bad_park=False

    def test_tts_owner_parks_and_restores_under_two_device_leases(self):
        child=SpeechProcess();compute=NativeCompute('TTS',(0,1));session=NativeSpeechSession('/model','/tokens','/worker','Ryan',lambda:child,compute=compute)
        class Provider:
            def models(self):return (SpeechModel('qwen-tts-1.7b-custom','Native','custom',('Ryan',)),)
            def enabled(self, model):return True
            def validate(self, request):pass
            def prepare(self, request):return SpeechPlan(('native',),SPEECH_BUDGET,lambda:session,compute.device_bytes)
        registry=SpeechRegistry();registry.register(Provider());runtime=SpeechRuntime(registry,lambda:self.resources);self.addCleanup(runtime.unload)
        request=SpeechInput('Hello Andrew.',{'mode':'custom','speaker':'Ryan'},'English','qwen-tts-1.7b-custom')
        runtime.generate(request,threading.Event());self.resources.offload_inactive_devices('image')
        self.assertTrue(session.parked);self.assertEqual(child.stops,0);self.assertTrue(runtime._ready)
        self.assertIsNone(runtime._device_reservation);runtime.generate(request,threading.Event())
        self.assertEqual(child.controls,['park','resume']);self.assertEqual(child.starts,1)
        runtime.unload();self.assertEqual(self.resources.snapshot()['reservations'],{})

    def test_tts_park_acknowledgement_and_resume(self):
        child=SpeechProcess();session=NativeSpeechSession('/model','/tokens','/worker','Ryan',lambda:child,compute=NativeCompute('TTS',(0,1)))
        self.addCleanup(session.unload);session.load(threading.Event());child.bad_park=True
        with self.assertRaises(Exception):session.offload_to_ram()
        self.assertFalse(session.parked);child.bad_park=False;session.offload_to_ram();self.assertTrue(session.parked)
        session.restore(threading.Event());self.assertFalse(session.parked);self.assertEqual(child.starts,1);self.assertEqual(child.stops,0)

if __name__=='__main__':unittest.main()
