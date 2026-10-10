import base64
import io
from pathlib import Path
import threading
import unittest
import wave
from unittest.mock import patch

from api.inference.errors import InferenceFailure
from api.inference.resources import ResourceManager, ResourceCancelled
from api.inference.stt.native_worker import NativeWhisper, PROCESS_BUDGET, pcm, resolve


def audio(frames=160):
    out=io.BytesIO()
    with wave.open(out,'wb') as source:
        source.setparams((1,2,16000,0,'NONE','NONE'));source.writeframes(b'\0\0'*frames)
    return 'data:audio/wav;base64,'+base64.b64encode(out.getvalue()).decode()


class Child:
    def __init__(self):
        self.starts=self.stops=self.requests=0;self.failure=None;self.fail_stop=False
    def start(self,command,env=None):
        self.starts+=1;self.workspace=Path(command[-1])
    def read(self,timeout,cancel=None):return 'ready 1 1024'
    def exchange(self,command,timeout,cancel=None):
        self.requests+=1
        if self.failure:raise self.failure
        assert command=='transcribe' and (self.workspace/'pcm.f32').stat().st_size==640
        (self.workspace/'text.txt').write_text(' Hello Andrew!')
        return 'done 14 4 1024'
    def stop(self):
        self.stops+=1
        if self.fail_stop:raise RuntimeError('reap failed')


class NativeWorkerTests(unittest.TestCase):
    def test_missing_export_acquires_checkpoint_with_cancellation_before_spawning(self):
        cancel = threading.Event()
        with patch.dict('os.environ', {}, clear=True), \
                patch('api.inference.stt.native_worker.model_manager') as manager, \
                patch('api.inference.stt.native_worker.ensure_assets') as export, \
                patch('api.inference.stt.native_worker.subprocess.run') as spawn, \
                patch.object(Path, 'exists', return_value=False):
            manager.root = Path('/models')
            manager.ensure_checkpoint.side_effect = ResourceCancelled('cancelled')
            with self.assertRaises(ResourceCancelled):
                resolve('large-v3', cancel=cancel)
            manager.ensure_checkpoint.assert_called_once_with('whisper-large-v3', cancel)
            export.assert_not_called()
            spawn.assert_not_called()

    def setUp(self):
        self.resources=ResourceManager(64*1024**3,{})
        self.child=Child();self.resolves=[]
        def resolve(model):self.resolves.append(model);return ['/native','/model','weights','dims','assets']
        self.owner=NativeWhisper(self.resources,resolve,lambda:self.child)
        self.addCleanup(self.cleanup)
    def cleanup(self):
        self.child.fail_stop=False;self.owner.close()
    def test_reuses_process_and_keeps_lifetime_admission(self):
        for _ in range(2):self.assertEqual(self.owner.transcribe(audio(),'large-v3')['text'],' Hello Andrew!')
        self.assertEqual(self.child.starts,1);self.assertEqual(self.resolves,['large-v3'])
        reservations=self.resources.snapshot()['reservations'];self.assertEqual(len(reservations),1)
        self.assertEqual(next(iter(reservations.values()))['host_bytes'],PROCESS_BUDGET)
        self.owner.offload_to_ram();self.assertEqual(self.child.stops,0)
        self.owner.close();self.assertEqual(self.resources.snapshot()['reservations'],{})
    def test_invalid_input_never_spawns(self):
        for value in ['https://example.com/audio.wav','data:audio/wav;base64,wrong',audio(480001)]:
            with self.assertRaises(InferenceFailure):self.owner.transcribe(value,'large-v3')
        with self.assertRaises(InferenceFailure):self.owner.transcribe(audio(),'large-v3','fr')
        self.assertEqual(self.child.starts,0);self.assertEqual(self.resources.snapshot()['reservations'],{})
    def test_cancelled_exchange_reaps_before_releasing(self):
        self.child.failure=InterruptedError('cancelled')
        with self.assertRaises(ResourceCancelled):self.owner.transcribe(audio(),'large-v3')
        self.assertEqual(self.child.stops,1);self.assertEqual(self.resources.snapshot()['reservations'],{})
    def test_failed_reap_keeps_accounting_and_quarantines(self):
        self.child.failure=InterruptedError('cancelled');self.child.fail_stop=True
        with self.assertRaisesRegex(RuntimeError,'reap failed'):self.owner.transcribe(audio(),'large-v3')
        self.assertTrue(self.owner.quarantined);self.assertTrue(self.owner.workspace.is_dir())
        self.assertEqual(len(self.resources.snapshot()['reservations']),1)
        with self.assertRaises(InferenceFailure):self.owner.check_execution_state()
    def test_precancel_no_allocations(self):
        cancel=threading.Event();cancel.set()
        with self.assertRaises(ResourceCancelled):self.owner.transcribe(audio(),'large-v3',cancel=cancel)
        self.assertEqual(self.child.starts,0);self.assertEqual(self.resources.snapshot()['reservations'],{})
    def test_false_memory_report_rejected_and_reaped(self):
        self.child.exchange=lambda *a,**k:'done 14 4 2048'
        with self.assertRaisesRegex(RuntimeError,'completion'):self.owner.transcribe(audio(),'large-v3')
        self.assertEqual(self.child.stops,1);self.assertEqual(self.resources.snapshot()['reservations'],{})
    def test_ready_failure_reaps(self):
        self.child.read=lambda *a,**k:'ready 1 0'
        with self.assertRaisesRegex(RuntimeError,'ready'):self.owner.load('large-v3')
        self.assertEqual(self.child.stops,1);self.assertEqual(self.resources.snapshot()['reservations'],{})
