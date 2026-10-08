"""Hand-authored format/diagnostic failures; no model or numerical dependency."""
import json
import os
from pathlib import Path
import struct
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from native.tests.reference_capture.artifacts import ExclusiveOutput, compare, decode, encode, read
from native.tests.reference_capture.capture import preflight, verify_backend_sources


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.data = encode(2, 0, False, [4., 2., 1.])

    def test_roundtrip_lowest_greedy_tie_and_signed_zero(self):
        self.assertEqual(decode(self.data)['logits'], [4.,2.,1.])
        tied = decode(encode(2,0,False,[4.,4.,1.]))
        self.assertEqual(tied['selected'],0)
        zero = compare(decode(encode(0,0,False,[0.,0.])),decode(encode(0,0,False,[-0.,0.])))
        self.assertTrue(zero['accepted']);self.assertEqual(zero['max_float32_ulp_distance'],0)
        self.assertIsNone(compare(decode(encode(0,0,False,[1.])),decode(encode(0,0,False,[1.])))['actual_token_margin'])

    def test_nonfinite_in_either_operand_is_rejected(self):
        for value in (float('nan'),float('inf'),-float('inf')):
            with self.assertRaises(ValueError):encode(2,0,False,[4.,value,1.])
            bad=bytearray(self.data);struct.pack_into('<f',bad,36,value)
            with self.assertRaises(ValueError):decode(bad)

    def test_malformed_bounds_records_and_completion(self):
        for offset,value in ((0,0),(4,2),(8,0),(8,262145),(12,2),(16,3),(20,3),(24,2),(28,0),(28,2)):
            bad=bytearray(self.data);struct.pack_into('<I',bad,offset,value)
            with self.subTest(offset=offset,value=value),self.assertRaises(ValueError):decode(bad)
        for bad in (self.data[:-1],self.data[:-4],self.data+b'X',self.data[:-4]+b'FAIL',b''):
            with self.assertRaises(ValueError):decode(bad)
        for values,chosen in (([4.,2.,1.],1),([4.,4.,1.],1)):
            bad=bytearray(self.data);struct.pack_into('<I',bad,20,chosen);struct.pack_into('<3f',bad,32,*values)
            with self.assertRaises(ValueError):decode(bad)

    def test_exact_comparison_never_uses_margin_or_ulp_to_pass(self):
        expected=decode(self.data)
        next_float=struct.unpack('<f',struct.pack('<I',0x40000001))[0]
        for value in (next_float,2.015625):
            report=compare(expected,decode(encode(2,0,False,[4.,value,1.])))
            self.assertFalse(report['accepted']);self.assertEqual(report['mismatch_count'],1)
            self.assertGreater(report['max_float32_ulp_distance'],0)
            self.assertGreater(report['actual_token_margin'],1)
        self.assertEqual(compare(expected,decode(encode(2,0,False,[4.,2.015625,1.])))['max_bfloat16_step_distance'],1)
        for data in (encode(1,0,False,[4.,2.,1.]),encode(2,0,True,[4.,2.,1.])):
            report=compare(expected,decode(data));self.assertFalse(report['accepted']);self.assertEqual(report['mismatch_count'],0)
        # A changed greedy winner remains a failure even with a large margin.
        report=compare(expected,decode(encode(2,1,False,[4.,100.,1.])))
        self.assertFalse(report['accepted']);self.assertFalse(report['record_equal'])

    def test_exclusive_output_and_symlink_preserve_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'capture';path.write_bytes(b'preserve')
            with self.assertRaises(FileExistsError):ExclusiveOutput(path)
            link=Path(folder)/'link';link.symlink_to(path)
            with self.assertRaises(FileExistsError):ExclusiveOutput(link)
            self.assertEqual(path.read_bytes(),b'preserve')
            fresh=Path(folder)/'fresh';output=ExclusiveOutput(fresh);output.write(self.data);output.finish();output.close()
            self.assertEqual(fresh.stat().st_mode & 0o777,0o600);self.assertEqual(read(fresh),decode(self.data))

    def test_partial_write_and_fsync_failures_are_not_success(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'partial';output=ExclusiveOutput(path)
            real_write=os.write
            def truncated(fd,data):
                real_write(fd,data[:8]);raise OSError('injected write failure')
            with patch('os.write',truncated),self.assertRaises(OSError):output.write(self.data)
            output.close()
            with self.assertRaises(ValueError):read(path)
            path=Path(folder)/'sync-failed';output=ExclusiveOutput(path);output.write(self.data)
            with patch('os.fsync',side_effect=OSError('injected sync failure')),self.assertRaises(OSError):output.finish()
            output.close()  # Presence of DONE cannot override the raised failure.

    def test_real_dimensions_rejected_before_shard_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'config.json').write_text(json.dumps({'model_type':'qwen3_5_moe','text_config':{'hidden_size':2048}}))
            (root/'fixture.safetensors').symlink_to('/not-a-fixture')
            with self.assertRaisesRegex(ValueError,'synthetic_dimensions_only'):preflight(root,2)

    def test_compare_cli_exit_status_and_oversized_file(self):
        with tempfile.TemporaryDirectory() as folder:
            expected, actual = Path(folder)/'expected', Path(folder)/'actual'
            expected.write_bytes(self.data)
            for data, code in ((self.data, 0), (encode(2,0,False,[4.,2.015625,1.]), 1), (b'partial', 1)):
                actual.write_bytes(data)
                result = subprocess.run([sys.executable, '-B', '-m', 'native.tests.reference_capture.compare',
                                         str(expected), str(actual)], capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, code)
                self.assertEqual(json.loads(result.stdout)['accepted'], code == 0)
            actual.write_bytes(self.data + b'\0' * (262144 * 4))
            with self.assertRaises(ValueError):read(actual)

    def test_runtime_source_drift_is_rejected(self):
        # No dependency import/installation needed to test this fail-closed gate.
        with patch('importlib.metadata.version', return_value='unreviewed'), self.assertRaisesRegex(ValueError,'backend_package'):
            verify_backend_sources()


if __name__=='__main__':unittest.main()
