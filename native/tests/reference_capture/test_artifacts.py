"""Hand-authored format/diagnostic failures; no model or numerical dependency."""
import json
import hashlib
import io
import os
from pathlib import Path
import struct
import tempfile
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from native.tests.reference_capture.artifacts import ExclusiveOutput, compare, decode, encode, read
from native.tests.reference_capture.capture import preflight, verify_backend_sources
from native.tests.reference_capture.cache_contract import validate_cache
from native.tests.reference_capture import run_fixtures


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.data = encode(2, 0, False, [4., 2., 1.])

    def test_pinned_reference_sources_exist_and_match(self):
        root = Path(__file__).resolve().parents[3]
        pins = json.loads(Path(__file__).with_name('backend-pins.json').read_text())
        for name, expected in pins['reference_sources'].items():
            with self.subTest(source=name):
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)
        backend = Path(__file__).with_name('cpu_backend.py').read_text()
        self.assertIn('from .qwen_reference import build_qwen', backend)
        self.assertFalse((root / 'api/inference/llm/qwen.py').exists())

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


class CacheContractTests(unittest.TestCase):
    def fixture(self):
        config = SimpleNamespace(num_hidden_layers=2, layer_types=['linear_attention', 'full_attention'],
                                 linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=2,
                                 linear_value_head_dim=4, linear_conv_kernel_dim=3,
                                 num_key_value_heads=2, head_dim=8)
        def tensor(shape, dtype='torch.bfloat16'):
            return SimpleNamespace(shape=shape, dtype=dtype)
        linear = SimpleNamespace(conv_states={0: tensor([1,24,3])},
                                 recurrent_states={0: tensor([1,4,2,4], 'torch.float32')})
        full = SimpleNamespace(keys=tensor([1,2,1,8]), values=tensor([1,2,1,8]))
        return SimpleNamespace(layers=[linear, full], get_seq_length=lambda: 1), config

    def test_valid_exact_cardinality_and_shapes(self):
        cache, config = self.fixture()
        records = validate_cache(cache, config)
        self.assertEqual([len(tensors) for _, tensors in records], [2,2])

    def test_missing_extra_layers_and_wrong_length_rejected(self):
        for operation in ('missing', 'extra', 'config_count', 'kinds_count', 'length', 'kind'):
            cache, config = self.fixture()
            if operation == 'missing':cache.layers.pop()
            elif operation == 'extra':cache.layers.append(cache.layers[-1])
            elif operation == 'config_count':config.num_hidden_layers += 1
            elif operation == 'kinds_count':config.layer_types.pop()
            elif operation == 'length':cache.get_seq_length = lambda: 0
            else:config.layer_types[0] = 'unknown'
            with self.subTest(operation=operation), self.assertRaises(ValueError):validate_cache(cache, config)

    def test_empty_extra_wrong_linear_state_slots_rejected(self):
        for name in ('conv_states', 'recurrent_states'):
            for keys in ([], [0,1], [1], [False]):
                cache, config = self.fixture()
                tensor = getattr(cache.layers[0], name)[0]
                setattr(cache.layers[0], name, {key: tensor for key in keys})
                with self.subTest(name=name, keys=keys), self.assertRaisesRegex(ValueError, 'cache_state_slots'):
                    validate_cache(cache, config)

    def test_each_state_shape_dtype_and_presence_rejected(self):
        for position in range(4):
            for corruption in ('shape', 'dtype', 'missing'):
                cache, config = self.fixture()
                linear, full = cache.layers
                owners = [(linear.conv_states, 0), (linear.recurrent_states, 0), (full, 'keys'), (full, 'values')]
                owner, key = owners[position]
                tensor = owner[key] if isinstance(owner, dict) else getattr(owner, key)
                if corruption == 'shape':tensor.shape[-1] += 1
                elif corruption == 'dtype':tensor.dtype = 'torch.float64'
                elif isinstance(owner, dict):owner[key] = None
                else:setattr(owner, key, None)
                with self.subTest(position=position, corruption=corruption), self.assertRaises(ValueError):
                    validate_cache(cache, config)


class RunnerFailureTests(unittest.TestCase):
    def test_timeout_preserves_raw_streams_and_failure_status(self):
        for stdout, stderr in ((b'partial\xff\n', b'failure\xfe\n'), (None, None)):
            with tempfile.TemporaryDirectory() as folder:
                destination = Path(folder)/'run'
                error = subprocess.TimeoutExpired(['capture'], 90, output=stdout, stderr=stderr)
                with patch.object(run_fixtures, 'create'), patch.object(run_fixtures.subprocess, 'run', side_effect=error), \
                     patch.object(sys, 'argv', ['run_fixtures', '--out', str(destination), '--cases', 'tiny']), \
                     patch.object(sys, 'stdout', io.StringIO()):
                    self.assertEqual(run_fixtures.main(), 1)
                self.assertEqual((destination/'tiny.stdout').read_bytes(), stdout or b'')
                self.assertEqual((destination/'tiny.stderr').read_bytes(), stderr or b'')
                report = json.loads((destination/'results.json').read_text())['tiny']
                self.assertFalse(report['accepted'])
                self.assertIn('timed out', report['error'])


if __name__=='__main__':unittest.main()
