"""Actual gates use invented metadata; never open an installed snapshot."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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


if __name__=='__main__':unittest.main()
