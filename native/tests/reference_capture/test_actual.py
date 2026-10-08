"""Actual gates use invented metadata; never open an installed snapshot."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
from contextlib import ExitStack
from types import SimpleNamespace

from native.tests.reference_capture import capture_actual as actual
from native.tests.reference_capture.artifacts import sha256
from native.tests.reference_capture.limits import Limits, ACTUAL_LIMITS


def manifest():
    files = {name:{'size':10,'sha256':digest} for name,digest in actual.METADATA.items()}
    files.update({'model.safetensors.index.json':{'size':10,'sha256':'0'*64},
                  'tokenizer.json':{'size':10,'sha256':'0'*64}})
    return dict(schema=1, execution_authorized=True, snapshot_writer_exclusion=True,
                snapshot=actual.SNAPSHOT,revision=actual.REVISION,input_ids=[248044],capacity=1,
                timeout_seconds=1800,host_bytes=40*1024**3,request_bytes=512*1024**2,
                image_id=json.loads(actual.PINS.read_text())['image_id'],
                backend_pins_sha256=sha256(actual.PINS.read_bytes()),source_revision='a'*40,
                wrapper_sources={p.name:sha256(p.read_bytes()) for p in actual.PINS.parent.glob('*.py')},files=files)


class ActualGateTests(unittest.TestCase):
    def test_manifest_rejects_before_payload_or_import(self):
        changes = {'execution_authorized':False,'snapshot_writer_exclusion':False,
                   'snapshot':'other','revision':'b'*40,'input_ids':[2],'capacity':2,
                   'timeout_seconds':1801,'host_bytes':41*1024**3,'request_bytes':513*1024**2,
                   'image_id':'sha256:'+'0'*64,'backend_pins_sha256':'0'*64,
                   'source_revision':'main','wrapper_sources':{}}
        for key,value in changes.items():
            m=manifest();m[key]=value
            with self.subTest(key=key), patch.object(actual,'checked_file',side_effect=AssertionError('payload')), \
                 patch.object(actual.importlib,'import_module',side_effect=AssertionError('numerical import')), self.assertRaises(ValueError):
                actual.preflight(Path('/never-open-actual-checkpoint'),m)

    def test_missing_approval_intent_is_first_gate(self):
        with patch.object(actual,'bounded_json',side_effect=AssertionError('file read')),self.assertRaisesRegex(ValueError,'execution_intent'):
            actual.run('/unused','/unused','0'*64,'/unused')

    def test_metadata_pin_size_and_path_rejections(self):
        for corruption in ('digest','size','traversal','too_many','total'):
            m=manifest()
            if corruption=='digest':m['files']['config.json']['sha256']='0'*64
            elif corruption=='size':m['files']['config.json']['size']=actual.BUFFER+1
            elif corruption=='traversal':m['files']['../payload']={'size':1,'sha256':'0'*64}
            elif corruption=='too_many':m['files'].update({str(i):{'size':1,'sha256':'0'*64} for i in range(129)})
            else:m['files']['payload']={'size':actual.MAX_TOTAL,'sha256':'0'*64}
            with self.subTest(corruption=corruption),self.assertRaises(ValueError):actual.validate_manifest(m)

    def test_stream_hash_generated_cpu_bytes_and_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'synthetic.bin';data=b'a'*(actual.BUFFER+13);p.write_bytes(data)
            before=actual.checked_file(p,len(data),sha256(data))
            self.assertEqual(before[2],len(data))
            with self.assertRaisesRegex(ValueError,'file_digest'):actual.checked_file(p,len(data),'0'*64)
            link=Path(folder)/'link';link.symlink_to(p)
            with self.assertRaisesRegex(ValueError,'symlink'):actual.checked_file(link,len(data),sha256(data))
            p.write_bytes(b'b'*len(data))
            with self.assertRaisesRegex(ValueError,'file_digest'):actual.checked_file(p,len(data),sha256(data))

    def test_ledger_defaults_and_ceilings(self):
        self.assertEqual((Limits().host_bytes,Limits().request_bytes),(128*1024**2,4*1024**2))
        self.assertEqual(ACTUAL_LIMITS.owner,'actual-reference')
        for args in ((41*1024**3,1),(40*1024**3,513*1024**2),(10,10),(True,1)):
            with self.assertRaises(ValueError):Limits(*args)

    def test_cgroup_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.dict(actual.os.environ,{'CUDA_VISIBLE_DEVICES':'0','NVIDIA_VISIBLE_DEVICES':'void'}), \
                 self.assertRaisesRegex(ValueError,'gpu_visibility'):actual.enforce_isolation(root)


class MockedActualRunTests(unittest.TestCase):
    def exercise(self, fault=None):
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            root=Path(folder)/'synthetic';root.mkdir()
            destination=Path(folder)/'out'
            m={'synthetic_control_flow_fixture':True}
            files={'synthetic.bin':{'size':1,'sha256':'0'*64}}
            report=copy.deepcopy(json.loads(actual.PINS.read_text())['runtime_build'])
            report.update(zero_reservations=True,calls={'model':1})
            backend=SimpleNamespace(forward=Mock(return_value=([0.]*248320,report)))
            stack.enter_context(patch.object(actual,'validate_manifest'))
            stack.enter_context(patch.object(actual,'enforce_isolation'))
            stack.enter_context(patch.object(actual.signal,'signal'))
            stack.enter_context(patch.object(actual.signal,'alarm'))
            stack.enter_context(patch.object(actual.importlib,'import_module',return_value=backend))
            stack.enter_context(patch.object(actual,'bounded_json',side_effect=[m,{} if fault=='manifest' else m]))
            stack.enter_context(patch.object(actual,'preflight',side_effect=[files,
                ValueError('source_drift') if fault=='source' else files]))
            stack.enter_context(patch.object(actual,'checked_file',side_effect=[(1,2,3),
                (1,2,4) if fault=='snapshot' else (1,2,3)]))
            if fault=='fsync':stack.enter_context(patch.object(actual.os,'fsync',side_effect=OSError('publication fsync')))
            if fault=='write':stack.enter_context(patch.object(actual.os,'write',side_effect=OSError('publication write')))
            if fault=='capture_close':
                original=actual.ExclusiveOutput.finish
                count=[0]
                def finish(output):
                    count[0]+=1
                    if count[0]==2:raise OSError('capture close failed')
                    return original(output)
                stack.enter_context(patch.object(actual.ExclusiveOutput,'finish',finish))
            if fault:
                with self.assertRaises((ValueError,OSError)):
                    actual.run(root,Path(folder)/'manifest','0'*64,destination,execute=True)
                self.assertTrue((destination/'reference.capture').exists())
                if fault!='capture_close':
                    self.assertEqual((destination/'reference.capture').read_bytes(),b'')
            else:
                result=actual.run(root,Path(folder)/'manifest','0'*64,destination,execute=True)
                self.assertEqual(result['status'],'complete')
                self.assertEqual((destination/'reference.capture').stat().st_size,993316)
            backend.forward.assert_called_once_with(root,248044,ACTUAL_LIMITS)

    def test_mocked_single_forward_success(self):self.exercise()

    def test_post_forward_drift_never_publishes_completion(self):
        for fault in ('manifest','source','snapshot'):
            with self.subTest(fault=fault):self.exercise(fault)

    def test_publication_failures_are_failures_even_after_done_bytes(self):
        for fault in ('fsync','write','capture_close'):
            with self.subTest(fault=fault):self.exercise(fault)

    def test_remaining_isolation_branches_with_synthetic_cgroup_files(self):
        for fault in (None,'gpu_device','dri','network','memory','pids','swap','cpu','missing'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
                root=Path(folder)
                values={'memory.max':str(40*1024**3),'pids.max':'256','memory.swap.max':'0','cpu.max':'200000 100000'}
                changes={'memory':('memory.max','max'),'pids':('pids.max','257'),
                         'swap':('memory.swap.max','1'),'cpu':('cpu.max','max 100000')}
                if fault in changes:
                    name,value=changes[fault];values[name]=value
                for name,value in values.items():(root/name).write_text(value)
                if fault=='missing':(root/'memory.max').unlink()
                stack.enter_context(patch.dict(actual.os.environ,{'CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'void'}))
                stack.enter_context(patch.object(Path,'glob',return_value=[Path('/dev/nvidia0')] if fault=='gpu_device' else []))
                original=Path.exists
                stack.enter_context(patch.object(Path,'exists',lambda p: fault=='dri' if str(p)=='/dev/dri' else original(p)))
                stack.enter_context(patch.object(Path,'iterdir',return_value=iter([Path('lo'),Path('eth0')] if fault=='network' else [Path('lo')])))
                if fault:
                    with self.assertRaises((ValueError,OSError)):actual.enforce_isolation(root)
                else:actual.enforce_isolation(root)


if __name__=='__main__':unittest.main()
