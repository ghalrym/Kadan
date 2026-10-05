import asyncio
import base64
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from PIL import Image

from api.inference.qwen_image import generate, REVISION
from api.inference.resources import ResourceManager, ResourceCancelled
from api.services.images import ImageManager, decode_source
from api.services.runtime import RuntimeFailure


class NativeImageTests(unittest.TestCase):
    def test_native_arguments_admission_cleanup_and_cancellation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'weights.safetensors').write_bytes(b'fixture')
            resources = ResourceManager(100 * 1024**3, {0: 4 * 1024**3})
            cancel = threading.Event()
            output = Image.new('RGBA', (2, 2))
            pipeline = Mock(return_value=SimpleNamespace(images=[output]))
            def construct(*args, **kwargs):
                self.assertEqual(args, (folder,))
                self.assertTrue(kwargs['local_files_only'])
                self.assertTrue(resources.snapshot()['reservations']['qwen-image-2.1']['active_leases'])
                return pipeline
            pipeline_type = SimpleNamespace(from_pretrained=Mock(side_effect=construct))
            torch = SimpleNamespace(float32='fp32', bfloat16='bf16', Generator=Mock(),
                cuda=SimpleNamespace(device=lambda _: nullcontext(), empty_cache=Mock()))
            modules = lambda: (torch, SimpleNamespace(QwenImage21Pipeline=pipeline_type))
            result = generate(path, resources, 'Tree', '16:9', [10, 11], cancel, modules=modules)
            self.assertEqual(len(result), 2)
            pipeline.enable_sequential_cpu_offload.assert_called_once_with(gpu_id=0)
            kwargs = pipeline.call_args.kwargs
            self.assertEqual((kwargs['width'], kwargs['height'], kwargs['num_inference_steps']), (2752, 1536, 40))
            self.assertEqual(resources.snapshot()['reservations'], {})
            def cancelled(**kwargs):
                cancel.set()
                kwargs['callback_on_step_end'](pipeline, 0, 0, {})
            pipeline.side_effect = cancelled
            with self.assertRaises(ResourceCancelled):
                generate(path, resources, 'Tree', '1:1', [10], cancel, modules=modules)
            self.assertEqual(resources.snapshot()['reservations'], {})

    def test_pipeline_failure_releases_resources(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'weights.safetensors').write_bytes(b'x')
            resources = ResourceManager(100 * 1024**3, {})
            factory = Mock(side_effect=RuntimeError('broken load'))
            torch = SimpleNamespace(float32='fp32', bfloat16='bf16')
            modules = lambda: (torch, SimpleNamespace(QwenImage21Pipeline=SimpleNamespace(from_pretrained=factory)))
            with self.assertRaisesRegex(RuntimeError, 'broken load'):
                generate(path, resources, 'x', '1:1', [1], threading.Event(), device='cpu', modules=modules)
            self.assertEqual(resources.snapshot()['reservations'], {})


class ImageManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.downloads = Mock()
        self.downloads.get_checkpoint.return_value = (SimpleNamespace(revision=REVISION), Path(self.temp.name))
        self.backend = Mock(side_effect=lambda path, resources, prompt, aspect, seeds, cancel, **kwargs:
            [Image.new('RGBA', (3, 2), (255, 0, 0, 100)) for _ in seeds])
        self.manager = ImageManager(self.temp.name, self.downloads, Mock(), self.backend)

    def test_real_png_publication_history_and_seed_reproducibility(self):
        result = self.manager.generate('Tree', '4:3', 2, 20, threading.Event())
        self.assertEqual(result.seeds, [20, 21])
        self.assertEqual(self.manager.history(), [result])
        with Image.open(self.manager.file(result.id, 0)) as output:
            self.assertEqual(output.mode, 'RGBA')
            self.assertEqual(output.size, (3, 2))
        self.assertEqual(len(result.urls), 2)
        with self.assertRaises(RuntimeFailure):
            self.manager.file('../secret', 0)
        with self.assertRaises(RuntimeFailure):
            self.manager.file(result.id, 2)

    def test_cancel_failure_and_retry_do_not_publish_partial_results(self):
        cancel = threading.Event()
        def cancelled(*args, **kwargs):
            cancel.set()
            return [Image.new('RGB', (1, 1))]
        self.backend.side_effect = cancelled
        with self.assertRaises(RuntimeFailure):
            self.manager.generate('x', '1:1', 1, 0, cancel)
        self.assertEqual(self.manager.history(), [])
        self.assertEqual(list(self.manager.root.iterdir()), [])
        self.backend.side_effect = RuntimeError('GPU failure')
        with self.assertRaisesRegex(RuntimeFailure, 'GPU failure'):
            self.manager.generate('x', '1:1', 1, 0, threading.Event())
        self.backend.side_effect = lambda *args, **kwargs: [Image.new('RGB', (1, 1))]
        self.assertEqual(self.manager.generate('x', '1:1', 1, 0, threading.Event()).seeds, [0])

    def test_only_inline_source_accepted_and_passed_to_native_editor(self):
        for source in ['https://example.com/a.png', 'file:///etc/passwd', 'data:text/plain;base64,eA==']:
            with self.assertRaises(ValueError):
                decode_source(source)
        stream = BytesIO()
        Image.new('RGB', (2, 2)).save(stream, format='PNG')
        source = 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()
        result = self.manager.generate('Edit', '1:1', 1, 0, threading.Event(), source)
        self.assertEqual(result.mode, 'Edit')
        self.assertEqual(self.backend.call_args.kwargs['image'].mode, 'RGBA')

    def test_missing_checkpoint_is_actionable_before_inference(self):
        self.downloads.get_checkpoint.side_effect = ValueError('incomplete')
        with self.assertRaisesRegex(RuntimeFailure, 'Download Qwen'):
            self.manager.generate('x', '1:1', 1, None, threading.Event())
        self.backend.assert_not_called()

    def test_concurrent_request_rejected(self):
        self.manager._gate.acquire()
        try:
            with self.assertRaisesRegex(RuntimeFailure, 'already active'):
                self.manager.generate('x', '1:1', 1, 0, threading.Event())
        finally:
            self.manager._gate.release()

    def test_disconnect_waits_for_native_cleanup(self):
        stopped = threading.Event()
        def backend(path, resources, prompt, aspect, seeds, cancel, **kwargs):
            cancel.wait(2)
            stopped.set()
            raise ResourceCancelled('cancelled')
        self.backend.side_effect = backend
        async def disconnect():
            return True
        async def run():
            with self.assertRaises(RuntimeFailure):
                await self.manager.run(SimpleNamespace(is_disconnected=disconnect),
                    SimpleNamespace(prompt='x', aspect='1:1', count=1, seed=0))
        asyncio.run(run())
        self.assertTrue(stopped.is_set())
        self.assertFalse(self.manager._gate.locked())
