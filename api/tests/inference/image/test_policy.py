"""CPU coverage of image deployment policy and per-device admission."""
from datetime import timedelta
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from api.inference.image.dual import DualImage
from api.inference.image.model import NativeImage
from api.inference.image.policy import GIB, ImagePolicy
from api.inference.image.rank_worker import configure_execution
from api.inference.resources import MemoryCapacity, ResourceManager
from api.services.runtime import RuntimeFailure, RuntimeManager


class ImagePolicyTests(unittest.TestCase):
    def test_default_policy_inherits_execution_settings(self):
        policy = ImagePolicy.from_environment({})
        self.assertIsNone(policy.cpus)
        self.assertIsNone(policy.threads)
        self.assertFalse(policy.blocking_sync)
        self.assertEqual(policy.affinity({3, 7, 9}), [3, 7, 9])
        self.assertEqual(policy.host_budget(10 * GIB, 3), 38 * GIB)
        self.assertEqual(policy.execution_bytes, 20 * GIB)

    def test_validated_overrides_keep_independent_deadlines_and_budgets(self):
        policy = ImagePolicy.from_environment({
            'KADAN_IMAGE_CPUS': '[3,7]', 'KADAN_IMAGE_THREADS': '4',
            'KADAN_IMAGE_HOST_BYTES': str(64 * GIB),
            'KADAN_IMAGE_WORKSPACE_BYTES': str(10 * GIB),
            'KADAN_IMAGE_DUAL_EXECUTION_BYTES': str(22 * GIB),
            'KADAN_IMAGE_OPERATION_SECONDS': '1800',
            'KADAN_IMAGE_COLLECTIVE_SECONDS': '240',
            'KADAN_IMAGE_CLEANUP_SECONDS': '45', 'KADAN_IMAGE_CUDA_WAIT': 'blocking'})
        self.assertEqual(policy.affinity({3, 7, 9}), [3, 7])
        self.assertEqual(policy.host_budget(10 * GIB, 3), 64 * GIB)
        self.assertEqual((policy.operation_seconds, policy.collective_seconds, policy.cleanup_seconds), (1800, 240, 45))
        with self.assertRaises(ValueError):
            policy.affinity({3})
        with self.assertRaises(ValueError):
            policy.host_budget(30 * GIB, 3)

    def test_invalid_configuration_fails_before_admission(self):
        cases = {
            'KADAN_IMAGE_CPUS': ['[]', '[true]', '[1,1]', 'null', '[-1]'],
            'KADAN_IMAGE_THREADS': ['0', '-1', '1.5', 'true'],
            'KADAN_IMAGE_WORKSPACE_BYTES': ['1', '-1'],
            'KADAN_IMAGE_DUAL_EXECUTION_BYTES': ['1', '0'],
            'KADAN_IMAGE_OPERATION_SECONDS': ['nan', 'inf', '0', '1'],
            'KADAN_IMAGE_COLLECTIVE_SECONDS': ['nan', '-1', '901'],
            'KADAN_IMAGE_CLEANUP_SECONDS': ['nan', 'inf', '0'],
            'KADAN_IMAGE_CUDA_WAIT': ['spin', 'true']}
        for name, values in cases.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    ImagePolicy.from_environment({name: value})

    def test_collective_deadline_rejects_timedelta_overflow_before_execution(self):
        rounded_limit = timedelta.max.total_seconds()
        below_limit = math.nextafter(rounded_limit, 0)
        policy = ImagePolicy.from_environment({
            'KADAN_IMAGE_OPERATION_SECONDS': str(rounded_limit),
            'KADAN_IMAGE_COLLECTIVE_SECONDS': str(below_limit)})
        self.assertEqual(policy.collective_seconds, below_limit)
        self.assertLessEqual(timedelta(seconds=policy.collective_seconds), timedelta.max)
        for value in (rounded_limit, 1e100):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'timedelta'):
                ImagePolicy.from_environment({
                    'KADAN_IMAGE_OPERATION_SECONDS': str(value),
                    'KADAN_IMAGE_COLLECTIVE_SECONDS': str(value)})

    def test_worker_inherits_threads_and_cuda_wait_unless_requested(self):
        torch = Mock()
        torch.cuda.get_device_properties.return_value.total_memory = 24 * GIB
        config = dict(device=1, execution_bytes=20 * GIB, threads=None, blocking_sync=False)
        with patch('api.inference.image.rank_worker.configure_blocking_sync') as blocking:
            self.assertEqual(configure_execution(torch, config), {'policy': 'default'})
            torch.set_num_threads.assert_not_called()
            torch.set_num_interop_threads.assert_not_called()
            blocking.assert_not_called()
            torch.cuda.set_per_process_memory_fraction.assert_called_once_with(20 / 24)
            config.update(threads=3, blocking_sync=True)
            configure_execution(torch, config)
            torch.set_num_threads.assert_called_once_with(3)
            torch.set_num_interop_threads.assert_called_once_with(3)
            blocking.assert_called_once_with(1)

    def test_default_and_varied_hardware_admit_each_device_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'weights.safetensors').write_bytes(b'fixture')
            for sizes in ({0: 24 * GIB, 1: 24 * GIB}, {2: 32 * GIB, 5: 48 * GIB}):
                with self.subTest(sizes=sizes), patch.dict('os.environ', {}, clear=True), patch(
                        'api.services.runtime.probe_memory', return_value=MemoryCapacity(64 * GIB, sizes)):
                    resources = RuntimeManager().ensure_resources()
                    dual = DualImage(path, resources, devices=list(sizes))
                    self.assertEqual(dual.budget.host_bytes, 8 * GIB + 21)
                    self.assertEqual(resources.snapshot()['reservations'], {})
                    self.assertEqual(dual.session.operation_timeout, 900)

    def test_single_image_uses_checkpoint_and_configured_workspace(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', {
                'KADAN_IMAGE_WORKSPACE_BYTES': str(10 * GIB)}, clear=True):
            path = Path(directory)
            (path / 'weights.safetensors').write_bytes(b'fixture')
            model = NativeImage(path, ResourceManager(20 * GIB, {0: 24 * GIB}))
            device, budget, offloaded = model._plan()
            self.assertEqual((device, budget, offloaded), ('cuda:0', 10 * GIB + 7, False))
            self.assertEqual(model.policy.host_budget(model.weights, 2), 10 * GIB + 14)


class SharedBudgetPolicyTests(unittest.TestCase):
    def budgets(self, environment, capacity):
        with patch.dict('os.environ', environment, clear=True), patch(
                'api.services.runtime.probe_memory', return_value=capacity):
            return RuntimeManager().ensure_resources().capacity

    def test_default_headroom_scales_for_small_and_large_cards(self):
        result = self.budgets({}, MemoryCapacity(64 * GIB, {0: 4 * GIB, 1: 24 * GIB}))
        self.assertEqual(result.device_bytes, {0: int(4 * GIB * .8), 1: 23 * GIB})

    def test_explicit_host_headroom_and_device_budget_precedence(self):
        result = self.budgets({'KADAN_HOST_BUDGET_BYTES': str(60 * GIB),
            'KADAN_GPU_HEADROOM_BYTES': str(2 * GIB),
            'KADAN_GPU_BUDGET_BYTES': '{"1": 1024}'},
            MemoryCapacity(64 * GIB, {0: 24 * GIB, 1: 24 * GIB}))
        self.assertEqual(result.host_bytes, 60 * GIB)
        self.assertEqual(result.device_bytes, {0: 22 * GIB, 1: 1024})

    def test_invalid_and_overcommitted_host_settings_fail_closed(self):
        for name, values in [('KADAN_HOST_BUDGET_BYTES', ['0', '-1', '101', 'true']),
                             ('KADAN_GPU_HEADROOM_BYTES', ['-1', 'nan', '1.2'])]:
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(RuntimeFailure):
                    self.budgets({name: value}, MemoryCapacity(100, {0: 100}))
        self.assertEqual(self.budgets({'KADAN_GPU_HEADROOM_BYTES': '101'},
            MemoryCapacity(100, {0: 100})).device_bytes, {0: 0})
