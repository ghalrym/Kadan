import json
import io
import time
import contextlib
import subprocess
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, mock_open, patch

import torch
from PIL import Image

from api_baseline import CASES, CRITERIA, PROTOCOL, compare, generate, publish, settings, verify
import launch_api_baseline as launcher
import supervisor_api_baseline as supervisor
from launch_api_baseline import require_review, restore_exact, storage


class Scheduler:
    def __init__(self): self.config={'kind':'fixture'}
    @classmethod
    def from_config(cls, config): return cls()
    def step(self, prediction, timestep, latent): return (latent,)


class Pipeline:
    def __init__(self, steps=40, nonfinite=False):
        self.scheduler=Scheduler()
        self.transformer=SimpleNamespace(forward=lambda **kwargs: None)
        self.image_processor=SimpleNamespace(postprocess=lambda *args, **kwargs: [Image.new('RGBA',(2048,2048))])
        self.steps, self.nonfinite=steps, nonfinite
        self.received=None
    def __call__(self, **kwargs):
        self.received=kwargs
        for index in range(self.steps):
            self.transformer.forward(kv_cache_mode='extract' if index==0 else 'cached')
            self.scheduler.step(None,40-index,None)
        return SimpleNamespace(images=self.image_processor.postprocess(
            torch.tensor([float('nan') if self.nonfinite else 0.]),output_type='pil'))


class BaselineTests(unittest.TestCase):
    def test_default_launcher_has_no_host_or_gpu_calls(self):
        with patch('sys.argv',['launch_api_baseline.py']), patch.object(launcher.host,'run') as run, contextlib.redirect_stdout(io.StringIO()) as output:
            launcher.main()
        run.assert_not_called()
        self.assertFalse(json.loads(output.getvalue())['gpu_access'])

    def test_supervisor_timeout_kills_and_reaps_before_releasing_lock(self):
        child=Mock(pid=12345)
        child.wait.side_effect=[subprocess.TimeoutExpired('baseline',1),0]
        def read(path,*args,**kwargs):
            return {'cpu.max':'200000 100000','memory.max':str(128*1024**3),'memory.swap.max':'0'}[path.name]
        with patch('sys.argv',['supervisor_api_baseline.py','A','a'*40]), patch.dict('os.environ',{'KADAN_STAGE_SECONDS':'1'}), \
                patch.object(Path,'read_text',read), patch.object(Path,'mkdir'), patch.object(Path,'write_text') as write, \
                patch('builtins.open',mock_open()), patch.object(supervisor.fcntl,'flock') as lock, \
                patch.object(supervisor.os,'statvfs',return_value=SimpleNamespace(f_blocks=262144,f_frsize=4096)), \
                patch.object(supervisor.os,'fstat',return_value=SimpleNamespace(st_ino=987)), \
                patch.object(supervisor.subprocess,'Popen',return_value=child), \
                patch.object(supervisor.os,'killpg') as kill, patch.object(supervisor.signal,'signal'):
            with self.assertRaises(subprocess.TimeoutExpired):supervisor.main()
        lock.assert_called_once()
        kill.assert_called_once_with(12345,supervisor.signal.SIGKILL)
        self.assertEqual(child.wait.call_count,2)
        report=json.loads(write.call_args.args[0])
        self.assertTrue(report['child_reaped'])
        self.assertEqual(report['queue_inode'],987)
        self.assertEqual(report['shm_bytes'],1024**3)

    def test_isolated_cases_use_frozen_settings_fresh_scheduler_and_seed(self):
        engines=[]
        for case in ('A','B','A'):
            pipeline=Pipeline(); old=pipeline.scheduler
            image, report=generate(pipeline,torch,case)
            engines.append(pipeline)
            self.assertIsNot(pipeline.scheduler,old)
            self.assertEqual(pipeline.received['prompt'],CASES[case]['prompt'])
            self.assertEqual(pipeline.received['generator'].initial_seed(),CASES[case]['seed'])
            self.assertEqual(pipeline.received['generator'].device.type,'cpu')
            self.assertNotIn('seed',pipeline.received)
            self.assertEqual({k:pipeline.received[k] for k in ('width','height','num_inference_steps','num_images_per_prompt','true_cfg_scale','use_kv_cache','output_type')},
                dict(width=2048,height=2048,num_inference_steps=40,num_images_per_prompt=1,true_cfg_scale=1.0,use_kv_cache=True,output_type='pil'))
            self.assertEqual(report['completed_steps'],40)
            self.assertTrue(report['raw_vae_finite'])
            self.assertEqual(image.mode,'RGBA')
        self.assertIsNot(engines[0].received['generator'],engines[2].received['generator'])

    def test_invalid_completion_and_nonfinite_restore_hooks(self):
        for pipeline in (Pipeline(39),Pipeline(41),Pipeline(nonfinite=True)):
            forward=pipeline.transformer.forward
            post=pipeline.image_processor.postprocess
            with self.assertRaises(ValueError):generate(pipeline,torch,'A')
            self.assertIs(pipeline.transformer.forward,forward)
            self.assertIs(pipeline.image_processor.postprocess,post)
            self.assertEqual(pipeline.scheduler.step.__func__,Scheduler.step)

    def test_output_is_no_overwrite_and_verification_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            image,details=generate(Pipeline(),torch,'A')
            artifact=publish(image,root)
            report=dict(protocol=PROTOCOL,case='A',source_commit='a'*40,settings=settings('A'),
                criteria=CRITERIA,status='passed',artifact=artifact,**details)
            (root/'manifest.json').write_text(json.dumps(report))
            verify(root,'A','a'*40)
            self.assertTrue(compare(root/'output.png',root,'A','a'*40)['passed'])
            altered=Image.new('RGBA',(2048,2048),(0,0,0,1))
            altered.save(root/'altered.png')
            comparison=compare(root/'altered.png',root,'A','a'*40)
            self.assertFalse(comparison['passed'])
            self.assertEqual(comparison['alpha_mae'],1.0)
            with self.assertRaises(FileExistsError):publish(image,root)
            self.assertFalse((root/'output.partial.png').exists())
            with self.assertRaises(ValueError):verify(root,'B','a'*40)
            with (root/'output.png').open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):verify(root,'A','a'*40)

    def test_review_binds_both_case_and_exact_source(self):
        record=dict(protocol=PROTOCOL,source_commit='a'*40,case='A',settings=settings('A'),
            criteria=CRITERIA,decision='approved-for-bounded-execution',reviewer='independent',review_reference='review')
        require_review(record,'a'*40,'A')
        for key,value in [('source_commit','b'*40),('case','B'),('reviewer',''),('settings',settings('B'))]:
            with self.assertRaises(ValueError):require_review(dict(record,**{key:value}),'a'*40,'A')

    def test_exact_restore_rejects_identity_changes_and_records_failed_baseline(self):
        before=dict(Id='original-id',Image='image',Config={},HostConfig={},Mounts=[],Path='python',Args=[],
            State=dict(Running=True,Health=dict(Status='healthy')))
        for fault in (None,'api','other','source'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                def run(*args,**kwargs):
                    return SimpleNamespace(stdout=('changed' if fault=='source' else 'a'*40) if 'rev-parse' in args else '')
                def inspect(name):
                    if name=='other': return dict(Id='changed' if fault=='other' else 'other-id')
                    if name==launcher.host.API and fault=='api': return dict(before,Id='changed')
                    return before
                def response(url,**kwargs):
                    stream=io.BytesIO(json.dumps(dict(state='ready')).encode());stream.status=200;return stream
                stack.enter_context(patch.object(launcher.host,'run',side_effect=run))
                restore=stack.enter_context(patch.object(launcher.host,'restore_container'))
                stack.enter_context(patch.object(launcher.host,'inspect_container',side_effect=inspect))
                stack.enter_context(patch.object(launcher.urllib.request,'urlopen',side_effect=response))
                root=Path(temporary);now=time.monotonic()
                args=(before,{'other':'other-id'},'a'*40,root,now+10,now,False,'digest')
                if fault:
                    with self.assertRaises(RuntimeError):restore_exact(*args)
                    self.assertFalse((root/'pause-result.json').exists())
                else:
                    restore_exact(*args)
                    restore.assert_called_once_with(before)
                    report=json.loads((root/'pause-result.json').read_text())
                    self.assertTrue(report['restored'])
                    self.assertFalse(report['baseline_passed'])
                    self.assertEqual(report['api_id'],'original-id')
                if fault=='source':restore.assert_not_called()

    def test_storage_rejects_overlap_and_exhaustion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); old=root/'old'; old.mkdir(); child=old/'child';child.mkdir()
            (old/'evidence').write_bytes(b'abc')
            self.assertEqual(storage([old],root/'new'),3)
            for roots,new in [([old,old],root/'new'),([old,child],root/'new'),([old],child/'new')]:
                with self.assertRaises(ValueError):storage(roots,new)
            with patch('launch_api_baseline.LIMIT',2):
                with self.assertRaises(ValueError):storage([old],root/'new')


if __name__=='__main__':unittest.main()
