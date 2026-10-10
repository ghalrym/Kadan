import io
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch
from PIL import Image
from api.inference.errors import InferenceFailure
from api.inference.image.native_worker import NativeImageRuntime, NativeImageSession, PROCESS_BUDGET, SIZES
from api.inference.resources import ResourceManager, ResourceCancelled, ResourceExhausted

class Child:
    def __init__(self):
        self.starts=self.stops=self.requests=0;self.failure=None;self.fail_stop=False;self.ready='ready 1 1024';self.report=None;self.bad_artifact=False
    def start(self, command, env=None):self.starts+=1;self.workspace=Path(command[-1])
    def read(self, timeout, cancel=None):return self.ready
    def exchange(self, command, timeout, cancel=None):
        self.requests+=1
        if self.failure:raise self.failure
        assert command=='generate'
        if self.bad_artifact:(self.workspace/'image.png').write_bytes(b'bad')
        else:
            request=json.loads((self.workspace/'request.json').read_text())
            Image.new('RGBA',(request['width'],request['height']),(255,0,0,255)).save(self.workspace/'image.png')
        return self.report or 'done 1024'
    def stop(self):
        self.stops+=1
        if self.fail_stop:raise RuntimeError('reap failed')

class NativeTests(unittest.TestCase):
    def setUp(self):
        self.resources=ResourceManager(96*1024**3,{0:1024});self.child=Child();self.session=NativeImageSession('/model','/worker',lambda:self.child)
        self.runtime=NativeImageRuntime(lambda:self.resources,lambda:(Path('/model'),Path('/worker')),lambda *args:self.session)
        self.request=SimpleNamespace(prompt='red ball',aspect='1:1',count=1,seed=0);self.addCleanup(self.cleanup)
    def cleanup(self):self.child.fail_stop=False;self.runtime.unload()
    def run_request(self):return self.runtime.run(self.request,threading.Event())
    def test_reuse_and_real_artifact_contract(self):
        for _ in range(2):self.assertEqual(len(self.run_request()['image']['images_base64']),1)
        self.assertEqual(self.child.starts,1);self.assertEqual(self.child.requests,2);self.assertEqual(next(iter(self.resources.snapshot()['reservations'].values()))['host_bytes'],PROCESS_BUDGET)
        self.runtime.unload();self.assertEqual(self.resources.snapshot()['reservations'],{});self.assertEqual(self.child.stops,1)
    def test_all_original_profiles_dispatch_and_validate_artifact(self):
        expected={'1:1':(2048,2048),'4:3':(2400,1792),'3:4':(1792,2400),'16:9':(2752,1536)}
        self.assertEqual(SIZES,expected)
        for aspect,size in expected.items():
            self.request.aspect=aspect
            result=self.run_request()['image']
            sent=json.loads((self.session.workspace/'request.json').read_text())
            self.assertEqual((sent['width'],sent['height']),size)
            self.assertIn(f'{size[0]}×{size[1]}',result['meta'])
        self.assertEqual(self.child.starts,1)
    def test_precancel_no_spawn(self):
        event=threading.Event();event.set()
        with self.assertRaises(ResourceCancelled):self.runtime.run(self.request,event)
        self.assertEqual(self.child.starts,0);self.assertEqual(self.resources.snapshot()['reservations'],{})
    def test_cancel_reaps_before_release(self):
        self.child.failure=InterruptedError()
        with self.assertRaises(ResourceCancelled):self.run_request()
        self.assertEqual(self.child.stops,1);self.assertEqual(self.resources.snapshot()['reservations'],{})
    def test_failed_reap_quarantines_and_retains(self):
        self.child.failure=InterruptedError();self.child.fail_stop=True
        with self.assertRaisesRegex(RuntimeError,'reap failed'):self.run_request()
        self.assertEqual(len(self.resources.snapshot()['reservations']),1);self.assertTrue(self.session.workspace.exists())
        with self.assertRaises(InferenceFailure):self.runtime.check_execution_state()
        with self.assertRaises(InferenceFailure):self.run_request()
        self.assertEqual(self.child.requests,1)
    def test_invalid_completion_cleans(self):
        self.child.report='done 2048'
        with self.assertRaisesRegex(RuntimeError,'completion'):self.run_request()
        self.assertEqual(self.resources.snapshot()['reservations'],{})
    def test_invalid_ready_cleans(self):
        self.child.ready='ready 1 0'
        with self.assertRaisesRegex(RuntimeError,'ready'):self.run_request()
        self.assertEqual(self.resources.snapshot()['reservations'],{})
    def test_invalid_png_cleans(self):
        self.child.bad_artifact=True
        with self.assertRaises(Exception):self.run_request()
        self.assertEqual(self.resources.snapshot()['reservations'],{})
    def test_budget_before_spawn(self):
        pressure=self.resources.reserve('pressure','tts',host_bytes=96*1024**3)
        try:
            cancel=threading.Event()
            with patch.object(cancel, 'wait', side_effect=lambda _: cancel.set()):
                with self.assertRaises(ResourceCancelled):self.runtime.run(self.request,cancel)
            self.assertEqual(self.child.starts,0)
        finally:pressure.release()
    def test_invalid_options_before_spawn(self):
        for aspect,seed in [('5:2',0),('1:1',2**64)]:
            self.request.aspect=aspect;self.request.seed=seed
            with self.assertRaises(InferenceFailure):self.run_request()
        self.assertEqual(self.child.starts,0)
    def test_gpu_handoff_and_host_pressure_eviction(self):
        calls=[];self.resources.reserve('text-device','llm',device_bytes={0:1024},evict=lambda:calls.append('offloaded'))
        self.run_request();self.assertEqual(calls,['offloaded'])
        other=self.resources.reserve('other','llm',host_bytes=96*1024**3);self.assertEqual(self.child.stops,1);self.assertIsNone(self.runtime.session);other.release()
    def test_multi_image_seed_wrap_and_one_load(self):
        self.request.count=2;self.request.seed=2**64-1
        result=self.run_request()['image'];self.assertEqual(result['seeds'],[2**64-1,0]);self.assertEqual(len(result['images_base64']),2);self.assertEqual(self.child.starts,1)

if __name__=='__main__':unittest.main()
