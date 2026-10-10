import io
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
import wave
from api.inference.resources import ResourceManager, ResourceCancelled, ResourceExhausted
from api.inference.tts.native_worker import NativeSpeechSession, PROCESS_BUDGET, validate
from api.inference.tts.speech_runtime import SpeechInput, SpeechModel, SpeechPlan, SpeechRegistry, SpeechRuntime, SpeechUnavailable


class Child:
    def __init__(self):
        self.starts=self.stops=self.requests=0;self.failure=None;self.fail_stop=False;self.ready='ready 1 1024';self.report=None
    def start(self,command,env=None):self.starts+=1;self.workspace=Path(command[-1])
    def read(self,timeout,cancel=None):return self.ready
    def exchange(self,command,timeout,cancel=None):
        self.requests+=1
        if self.failure:raise self.failure
        assert command=='speak' and json.loads((self.workspace/'request.json').read_text())['script']=='Hello Andrew.'
        out=io.BytesIO()
        with wave.open(out,'wb') as source:source.setparams((1,2,24000,0,'NONE','NONE'));source.writeframes(b'\0\0'*1920)
        (self.workspace/'audio.wav').write_bytes(out.getvalue())
        return self.report or 'done 3884 1 1024'
    def stop(self):
        self.stops+=1
        if self.fail_stop:raise RuntimeError('reap failed')


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.resources=ResourceManager(32*1024**3,{});self.child=Child()
        self.session=NativeSpeechSession('/model','/tokens','/worker','Ryan',lambda:self.child)
        self.request=SpeechInput('Hello Andrew.',dict(mode='custom',speaker='Ryan'),'English','qwen-tts-1.7b-custom')
        session=self.session
        class Provider:
            def models(self):return (SpeechModel('qwen-tts-1.7b-custom','Native','custom',('Ryan',)),)
            def enabled(self,model):return True
            def validate(self,request):validate(request)
            def prepare(self,request):return SpeechPlan(('native',),PROCESS_BUDGET,lambda:session)
        registry=SpeechRegistry();registry.register(Provider());self.runtime=SpeechRuntime(registry,lambda:self.resources);self.addCleanup(self.cleanup)
    def cleanup(self):self.child.fail_stop=False;self.runtime.unload()
    def run_request(self):return self.runtime.generate(self.request,threading.Event())
    def test_reuses_and_retains_until_eviction(self):
        for _ in range(2):self.assertEqual(len(self.run_request().wav),3884)
        self.assertEqual(self.child.starts,1);self.assertEqual(self.child.requests,2)
        values=list(self.resources.snapshot()['reservations'].values());self.assertEqual(len(values),1);self.assertEqual(values[0]['host_bytes'],PROCESS_BUDGET);self.assertEqual(values[0]['active_leases'],0)
        self.runtime.unload();self.assertEqual(self.resources.snapshot()['reservations'],{});self.assertEqual(self.child.stops,1)
    def test_controls_are_forwarded_without_reloading(self):
        for speaker,language,instruction in [('Vivian','Chinese','warm'),('Ono_Anna','Japanese',''),('Ryan','Auto','')]:
            self.request=SpeechInput('Hello Andrew.',dict(mode='custom',speaker=speaker,instruction=instruction),language,'qwen-tts-1.7b-custom')
            self.run_request()
            sent=json.loads((self.session.workspace/'request.json').read_text())
            self.assertEqual(sent,dict(script='Hello Andrew.',speaker=speaker,language=language,instruction=instruction))
        self.assertEqual(self.child.starts,1)
    def test_precancel_does_not_spawn(self):
        event=threading.Event();event.set()
        with self.assertRaises(ResourceCancelled):self.runtime.generate(self.request,event)
        self.assertEqual(self.child.starts,0);self.assertEqual(self.resources.snapshot()['reservations'],{})
    def test_cancel_reaps_before_release(self):
        self.child.failure=InterruptedError()
        with self.assertRaises(ResourceCancelled):self.run_request()
        self.assertEqual(self.child.stops,1);self.assertEqual(self.resources.snapshot()['reservations'],{})
    def test_failed_reap_retains_admission(self):
        self.child.failure=InterruptedError();self.child.fail_stop=True
        with self.assertRaisesRegex(RuntimeError,'reap failed'):self.run_request()
        self.assertEqual(len(self.resources.snapshot()['reservations']),1);self.assertTrue(self.session.workspace.exists())
        with self.assertRaises(SpeechUnavailable):self.session.check_execution_state()
    def test_empty_generation_rejected(self):
        with self.assertRaises(ValueError):self.runtime.generate(SpeechInput('',dict(mode='custom',speaker='Ryan'),'English','qwen-tts-1.7b-custom'),threading.Event())
        self.assertEqual(self.child.starts,0)
        self.assertEqual(self.resources.snapshot()['reservations'],{})
    def test_invalid_memory_response(self):
        self.child.report='done 3884 1 2048'
        with self.assertRaisesRegex(RuntimeError,'completion'):self.run_request()
        self.assertEqual(self.child.stops,1);self.assertEqual(self.resources.snapshot()['reservations'],{})
    def test_invalid_ready(self):
        self.child.ready='ready 1 0'
        with self.assertRaisesRegex(RuntimeError,'ready'):self.run_request()
        self.assertEqual(self.child.stops,1);self.assertEqual(self.resources.snapshot()['reservations'],{})
    def test_unsupported_options_do_not_spawn(self):
        for request in [SpeechInput('Hi',dict(mode='custom',speaker='Unknown'),'English','qwen-tts-1.7b-custom'),SpeechInput('Hi',dict(mode='custom',speaker='Ryan'),'Klingon','qwen-tts-1.7b-custom')]:
            with self.assertRaises(ValueError):self.runtime.generate(request,threading.Event())
        self.assertEqual(self.child.starts,0)
    def test_memory_pressure_does_not_spawn(self):
        pressure=self.resources.reserve('pressure','tts',host_bytes=32*1024**3)
        try:
            cancel=threading.Event()
            with patch.object(cancel, 'wait', side_effect=lambda _: cancel.set()):
                with self.assertRaises(ResourceCancelled):self.runtime.generate(self.request,cancel)
            self.assertEqual(self.child.starts,0)
        finally:pressure.release()

if __name__=='__main__':unittest.main()
