import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock,patch
from .gpu_gate import GpuGate,Snapshot,UUID
from .gpu_stage import GPUCorrectnessStage,HostTelemetry,NATIVE_ARGS,NATIVE_HOST_BYTES,LOGITS_DIAGNOSTICS_BYTES,HOST_BYTES,journal_records,process_start_time
from .container_stage import DockerReferenceStage
from .supervisor import Observation

CID='a'*64


class GPUStageTests(unittest.TestCase):
    def sample(self):return Snapshot({UUID:{'index':0,'free_bytes':24000*1024**2,'temperature':50}},(),False)

    def test_exact_one_bos_command_and_second_start_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            s=object.__new__(GPUCorrectnessStage);s.verified_id=CID;s.child=None;s.logs=[];s.evidence=Path(folder)
            s.telemetry=Mock();s.telemetry.sample.return_value=self.sample();s.gate=GpuGate(self.sample())
            with patch('native.tests.reference_capture.gpu_stage.subprocess.Popen') as spawn:
                s.start(CID,1)
                self.assertEqual(spawn.call_args.args[0],['docker','exec','-e','LD_LIBRARY_PATH=/opt/kadan/lib',CID,*NATIVE_ARGS])
                self.assertEqual(NATIVE_ARGS[-1],'248044');self.assertEqual(NATIVE_ARGS[4],'1')
                # Independent numeric contract: native envelope plus FP32 logits.
                self.assertEqual(NATIVE_HOST_BYTES,271525888)
                self.assertEqual(LOGITS_DIAGNOSTICS_BYTES,248320 * 4)
                self.assertEqual(HOST_BYTES,272519168)
                self.assertEqual(NATIVE_ARGS[5:8],['272519168','21434868224','536870912'])
                with self.assertRaises(ValueError):s.start(CID,1)
            for log in s.logs:log.close()

    def test_telemetry_timeout_retains_unreaped_query(self):
        telemetry=HostTelemetry('s=pin')
        child=Mock();child.wait.side_effect=subprocess.TimeoutExpired(['query'],1);child.poll.return_value=None
        with patch('native.tests.reference_capture.gpu_stage.subprocess.Popen',return_value=child),self.assertRaises(subprocess.TimeoutExpired):
            telemetry.command(['fake-query'],1)
        self.assertEqual(telemetry.children,[child]);child.kill.assert_called_once();child.wait.assert_called_once()

    def test_no_cleanup_when_gpu_baseline_or_ownership_uncertain(self):
        s=object.__new__(GPUCorrectnessStage);s.gate=GpuGate(self.sample());s.telemetry=Mock()
        base=Observation(CID,False,0,0,True,True,True,False,0,True,0,0)
        sample=self.sample();sample.devices[UUID]['free_bytes']-=65*1024**2;s.telemetry.sample.return_value=sample
        with patch.object(DockerReferenceStage,'observe',return_value=base),self.assertRaisesRegex(ValueError,'unexplained_gpu0_allocation'):
            s.observe(CID,1)
        s.telemetry.sample.return_value=self.sample()
        with patch.object(DockerReferenceStage,'observe',return_value=base):
            self.assertTrue(s.observe(CID,1).cleanup_known(CID))

    def test_journal_requires_access_and_cursor_continuity(self):
        record={'__CURSOR':'cursor','_BOOT_ID':'a'*32,'_TRANSPORT':'kernel','MESSAGE':'ordinary kernel message'}
        self.assertEqual(journal_records(json.dumps(record),'cursor'),('cursor','ordinary kernel message'))
        for raw in ('',json.dumps({**record,'__CURSOR':'new'}),json.dumps({**record,'_TRANSPORT':'syslog'}),json.dumps({**record,'_BOOT_ID':None})):
            with self.subTest(raw=raw),self.assertRaises(ValueError):journal_records(raw,'cursor')
        xid=json.dumps(record)+'\n'+json.dumps({**record,'__CURSOR':'next','MESSAGE':'NVRM: Xid error'})
        cursor,messages=journal_records(xid,'cursor')
        self.assertEqual(cursor,'next');self.assertIn('NVRM: Xid',messages)

    def test_warning_preserved_and_rejected_even_on_zero_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            telemetry=HostTelemetry('cursor',Path(folder))
            def spawn(*args,**kwargs):
                kwargs['stderr'].write(b'not seeing messages from other users')
                kwargs['stderr'].flush()
                return Mock(wait=Mock(return_value=0))
            with patch('native.tests.reference_capture.gpu_stage.subprocess.Popen',side_effect=spawn),self.assertRaisesRegex(ValueError,'warning_or_access_incomplete'):
                telemetry.command(['fake-journal'],1)
            saved=json.loads((Path(folder)/'telemetry-00000.json').read_text())
            self.assertEqual(saved['exit'],0)
            self.assertIn('not seeing messages',saved['stderr'])

    def test_proc_stat_start_time_handles_complex_comm(self):
        fields=['S']+['0']*18+['12345']
        self.assertEqual(process_start_time('42 (name with ) spaces) '+' '.join(fields)),12345)
        with self.assertRaises(ValueError):process_start_time('42 (bad) S')

    def test_unreviewed_plan_never_starts_telemetry(self):
        with patch('native.tests.reference_capture.gpu_stage.bounded_json',return_value={'schema':1,'stage':'one-bos-gpu-correctness','technical_release':False}),self.assertRaises(ValueError):
            GPUCorrectnessStage('/tmp/unopened-plan','0'*64)


if __name__=='__main__':unittest.main()
