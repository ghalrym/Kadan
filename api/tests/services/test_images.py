import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from uuid import uuid4
from unittest.mock import Mock, patch

from PIL import Image

from api.inference.image.model import REVISION
from api.inference.image.feature import ImageFeature
from api.inference.resources import ResourceManager, ResourceCancelled
from api.services.images import ImageManager, decode_source
from api.services.runtime import RuntimeFailure


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

    def test_output_scratch_is_admitted_until_publication_returns_or_fails(self):
        self.manager.backend = None
        resources = ResourceManager(2 * 1024**3, {})
        self.manager.runtime.ensure_resources.return_value = resources
        def publish(*args):
            rows = resources.snapshot()['reservations']
            self.assertEqual(len(rows), 1)
            self.assertTrue(next(iter(rows.values()))['active_leases'])
            raise RuntimeFailure('publication failed')
        self.manager._generate = publish
        with self.assertRaisesRegex(RuntimeFailure, 'publication failed'):
            self.manager.generate('x', '1:1', 1, 0, threading.Event())
        self.assertEqual(resources.snapshot()['reservations'], {})

    def test_disconnect_waits_for_native_cleanup(self):
        stopped = threading.Event()
        started = threading.Event()
        def backend(path, resources, prompt, aspect, seeds, cancel, **kwargs):
            started.set()
            cancel.wait(2)
            stopped.set()
            raise ResourceCancelled('cancelled')
        self.backend.side_effect = backend
        async def run():
            feature = ImageFeature(self.manager)
            task = asyncio.create_task(feature(SimpleNamespace(prompt='x', aspect='1:1', count=1, seed=0)))
            await asyncio.to_thread(started.wait, 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(run())
        self.assertTrue(stopped.is_set())
        self.assertFalse(self.manager._gate.locked())


    def test_unpublished_manifest_is_hidden_until_atomic_rename(self):
        entered, release = threading.Event(), threading.Event()
        rename = Path.rename
        def barrier(path, target):
            if path.parent == self.manager.root and path.name.startswith('.'):
                entered.set()
                if not release.wait(3):
                    raise TimeoutError('publication barrier')
            return rename(path, target)
        with patch.object(Path, 'rename', barrier), ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.manager.generate, 'Tree', '1:1', 1, 0, threading.Event())
            try:
                self.assertTrue(entered.wait(2))
                self.assertTrue(list(self.manager.root.glob('*/result.json')))
                self.assertEqual(self.manager.history(), [])
            finally:
                release.set()
            result = future.result(timeout=3)
        self.assertEqual(self.manager.history(), [result])
        published = self.manager.root / result.id
        published.rename(self.manager.root / str(uuid4()))
        self.assertEqual(self.manager.history(), [])  # Manifest identity must match.

    def test_invalid_source_admission_never_evicts_resident_text(self):
        resources = ResourceManager(1024**3, {0: 50})
        evict = Mock()
        host = resources.reserve('text', 'llm', host_bytes=1, evict=evict)
        self.manager.runtime = SimpleNamespace(ensure_resources=lambda: resources)
        with self.assertRaises(RuntimeFailure) as caught:
            self.manager.validate_source('invalid')
        self.assertEqual(caught.exception.status_code, 503)
        evict.assert_not_called()
        self.assertEqual(set(resources.snapshot()['reservations']), {'text'})
        host.release()

    def test_deployment_offload_setting_validated_and_reconfigures_existing_pipeline(self):
        self.manager.backend=None
        path=Path(self.temp.name)
        for part in ('text_encoder','transformer','vae'):
            (path/part).mkdir();(path/part/'model.safetensors').write_bytes(b'fixture')
        self.manager.runtime.ensure_resources.return_value=ResourceManager(100*1024**3,{0:24*1024**3})
        with patch.dict('os.environ',{'KADAN_IMAGE_DEVICE':'cuda:0','KADAN_IMAGE_OFFLOAD':'invalid'}):
            with self.assertRaises(RuntimeFailure) as caught:self.manager.validate_request(None)
            self.assertEqual(caught.exception.status_code,422)
        with patch.dict('os.environ',{'KADAN_IMAGE_DEVICE':'cuda:0','KADAN_IMAGE_OFFLOAD':'component'}):
            self.manager.validate_request(None)
            old=Mock(path=path,requested='cuda:0',offload_mode='sequential')
            self.manager.native=old
            with patch('api.services.images.qwen_image.NativeImage') as factory:
                self.manager.load()
                old.close.assert_called_once()
                self.assertEqual(factory.call_args.kwargs['offload_mode'],'component')
