from contextlib import nullcontext
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import weakref

from PIL import Image

from api.inference.image.model import NativeImage, GIB
from api.inference.resources import ResourceManager, ResourceCancelled, ResourceBusy, ResourceExhausted


class NativeImageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        (self.path / 'weights.safetensors').write_bytes(b'fixture')
        self.resources = ResourceManager(100 * GIB, {0: 12 * GIB, 1: 24 * GIB})
        self.pipeline = Mock(return_value=SimpleNamespace(images=[Image.new('RGBA', (2, 2))]))
        self.factory = Mock(return_value=self.pipeline)
        self.torch = SimpleNamespace(float32='fp32', bfloat16='bf16', Generator=Mock(),
            cuda=SimpleNamespace(device=lambda _: nullcontext(), empty_cache=Mock(), synchronize=Mock()))
        self.modules = lambda: (self.torch, SimpleNamespace(QwenImage21Pipeline=SimpleNamespace(from_pretrained=self.factory)))
        self.native = NativeImage(self.path, self.resources, modules=self.modules)
        self.addCleanup(self.native.close)

    def test_resident_reuse_then_offload_and_reload_under_pressure(self):
        self.native.generate('Tree', '16:9', [10, 11], threading.Event())
        self.assertEqual(self.native.device, 'cuda:1')
        self.assertIsNotNone(self.native.gpu)
        self.pipeline.enable_sequential_cpu_offload.assert_not_called()
        self.assertEqual(self.pipeline.call_args.kwargs['width'], 2752)
        self.assertEqual(self.pipeline.call_args.kwargs['num_inference_steps'], 40)
        self.native.generate('Other', '1:1', [12], threading.Event())
        self.factory.assert_called_once()
        self.assertEqual(len(self.resources.snapshot()['reservations']), 2)
        self.native.offload_to_ram()
        self.assertIsNone(self.native.gpu)
        self.assertIsNotNone(self.native.pipeline)
        self.pipeline.to.assert_called_with('cpu')
        pressure = self.resources.reserve('pressure', 'video', host_bytes=100 * GIB)
        self.assertIsNone(self.native.pipeline)
        self.assertIsNone(self.native.host)
        pressure.release()
        self.native.generate('Reload', '1:1', [1], threading.Event())
        self.assertEqual(self.factory.call_count, 2)

    def test_sequential_offload_parks_hooks_and_releases_gpu(self):
        with (self.path / 'weights.safetensors').open('wb') as stream:
            stream.truncate(20 * GIB)
        self.native.weights = 20 * GIB
        self.native.generate('Edit', '1:1', [1], threading.Event(), image=Image.new('RGBA', (2, 2)))
        self.pipeline.enable_sequential_cpu_offload.assert_called_once_with(gpu_id=1)
        self.pipeline.remove_all_hooks.assert_called_once()
        self.assertIsNone(self.native.gpu)
        self.assertIsNotNone(self.native.host)
        self.assertEqual(self.pipeline.call_args.kwargs['image'].mode, 'RGBA')

    def test_active_leases_prevent_eviction_and_cancellation_clears_residency(self):
        cancel = threading.Event()
        def forward(**kwargs):
            with self.assertRaises(ResourceBusy):
                self.resources.offload_workload_devices('image')
            cancel.set()
            kwargs['callback_on_step_end'](self.pipeline, 0, 0, {})
        self.pipeline.side_effect = forward
        with self.assertRaises(ResourceCancelled):
            self.native.generate('x', '1:1', [1], cancel)
        self.assertIsNone(self.native.pipeline)
        self.assertEqual(self.resources.snapshot()['reservations'], {})

    def test_explicit_cpu_and_invalid_or_insufficient_gpu_fail_truthfully(self):
        self.native.requested = 'cpu'
        self.native.generate('x', '1:1', [1], threading.Event())
        self.assertEqual(self.factory.call_args.kwargs['torch_dtype'], 'fp32')
        self.assertIsNone(self.native.gpu)
        self.native.close()
        self.native.requested = 'cuda:9'
        with self.assertRaises(ResourceExhausted):
            self.native.load()
        self.native.requested = '0,1'
        with self.assertRaises(ValueError):
            self.native.load()

    def test_single_gpu_fit_is_preferred_and_capacity_is_never_pooled(self):
        self.native.weights = 8 * GIB
        self.assertEqual(self.native._plan(), ('cuda:1', 16 * GIB, False))
        self.native.weights = 40 * GIB
        self.assertEqual(self.native._plan(), ('cuda:1', 8 * GIB, True))
        self.native.requested = 'cuda:0'
        self.assertEqual(self.native._plan(), ('cuda:0', 8 * GIB, True))

    def test_shared_placement_capacity_view_is_consumed_when_present(self):
        self.resources.available_devices = lambda: {0: 12 * GIB, 1: 0}
        self.assertEqual(self.native._plan()[0], 'cuda:0')

    def test_exception_frames_are_cleared_before_releasing_reservations(self):
        class Allocation:
            pass
        for during_load in (True, False):
            references = []
            def fail(*args, **kwargs):
                allocation = Allocation()
                references.append(weakref.ref(allocation))
                try:
                    raise ValueError('allocation failure')
                except ValueError as exc:
                    raise RuntimeError('native failure') from exc
            self.native.requested = 'cpu'
            self.factory.side_effect = fail if during_load else None
            self.pipeline.side_effect = None if during_load else fail
            release = self.resources._release
            def checked_release(owner, token):
                self.assertTrue(all(reference() is None for reference in references))
                release(owner, token)
            self.resources._release = checked_release
            try:
                with self.assertRaisesRegex(RuntimeError, 'native failure'):
                    self.native.generate('x', '1:1', [1], threading.Event())
                self.assertEqual(self.resources.snapshot()['reservations'], {})
            finally:
                self.resources._release = release
