"""Single-action load tests use the real settings store and controlled adapters."""
import asyncio
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from api.inference.resources import ResourceManager
from api.services.model_catalog import CATALOG
from api.services.model_downloads import ModelManager
from api.services.chat_runtime import ChatRuntime
from api.inference.errors import InferenceFailure
from api.tests.services.test_chat_runtime import Adapter


class ConfigureLoadTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.models = ModelManager(Path(temporary.name))
        for entry in CATALOG.values():
            directory = self.models._checkpoint_directory(entry)
            directory.mkdir()
            (directory / 'config.json').write_text(json.dumps({'max_position_embeddings': 131072}))
        # Completeness/integrity has its own downloader suite. Use tiny configs,
        # never download weights or invoke native inference in lifecycle tests.
        complete = patch.object(self.models, '_checkpoint_complete', return_value=True)
        complete.start()
        self.addCleanup(complete.stop)
        singleton = patch('api.services.chat_runtime.model_manager', self.models)
        singleton.start()
        self.addCleanup(singleton.stop)
        self.factory = Mock(side_effect=lambda *args, **kwargs: Adapter())
        self.runtime = ChatRuntime(self.factory, ResourceManager(1000, {0: 1000}))
        self.addAsyncCleanup(self.runtime.close)

    async def ready(self, model='small', **kwargs):
        response = await self.runtime.load(model, **kwargs)
        self.assertEqual(response['state'], 'loading')
        await self.runtime.task
        self.assertEqual(self.runtime.state, 'ready')
        return self.runtime.adapter

    async def test_start_restores_saved_selection_and_context(self):
        self.models.select('small')
        self.models.set_context('small', 32000)
        await self.runtime.start()
        self.assertEqual(self.runtime.state, 'loading')
        await self.runtime.task
        self.assertEqual(self.runtime.state, 'ready')
        self.assertEqual(self.runtime.model_id, 'small')
        self.assertEqual(self.runtime.adapter.configured_context_limit, 32000)

    async def test_start_without_selection_does_not_load(self):
        await self.runtime.start()
        self.assertEqual(self.runtime.state, 'unloaded')
        self.factory.assert_not_called()

    async def test_start_with_missing_checkpoint_exposes_error(self):
        self.models.select('small')
        with patch.object(self.models, '_checkpoint_complete', return_value=False):
            await self.runtime.start()
        self.assertEqual(self.runtime.state, 'error')
        self.assertEqual(self.runtime.model_id, 'small')
        self.assertTrue(self.runtime.error)
        self.factory.assert_not_called()

    async def test_start_with_invalid_context_exposes_error_and_releases_selection(self):
        self.models.select('small')
        (self.models.root / 'context.json').write_text('invalid')
        await self.runtime.start()
        self.assertEqual(self.runtime.state, 'error')
        self.assertFalse(self.models._in_use)

    async def test_start_with_failed_constructor_exposes_error(self):
        self.models.select('small')
        self.factory.side_effect = RuntimeError('CUDA allocation failed')
        await self.runtime.start()
        await self.runtime.task
        self.assertEqual(self.runtime.state, 'error')
        self.assertEqual(self.runtime.error, 'CUDA allocation failed')
        self.assertFalse(self.models._in_use)

    async def test_default_and_explicit_null_switch_context(self):
        old = await self.ready()
        self.assertEqual(old.configured_context_limit, 65536)
        self.assertEqual(self.models.get_selected()[0].id, 'small')
        new = await self.ready(context_limit=None)
        self.assertTrue(old.closed)
        self.assertIsNone(new.configured_context_limit)
        self.assertEqual(self.runtime.status()['effective_context_limit'], 131072)
        self.assertIsNone(self.models.configured_context('small'))

    async def test_switch_ready_model_in_one_action(self):
        old = await self.ready()
        await self.ready('medium', context_limit=32768)
        self.assertTrue(old.closed)
        self.assertEqual(self.models.get_selected()[0].id, 'medium')
        self.assertEqual(self.models.configured_context('medium'), 32768)
        self.assertEqual(self.factory.call_count, 2)

    async def test_invalid_or_incomplete_target_keeps_ready_model_and_store(self):
        old = await self.ready()
        before = (self.models.root / 'context.json').read_bytes()
        for value in (131073, True, '65536', 0):
            with self.assertRaises(InferenceFailure) as error:
                await self.runtime.load('medium', value)
            self.assertEqual(error.exception.status_code, 422)
        with patch.object(self.models, '_checkpoint_complete', return_value=False):
            with self.assertRaises(InferenceFailure) as error:
                await self.runtime.load('medium', 65536)
            self.assertEqual(error.exception.status_code, 409)
        self.assertFalse(old.closed)
        self.assertEqual(self.runtime.state, 'ready')
        self.assertEqual(self.models.get_selected()[0].id, 'small')
        self.assertEqual((self.models.root / 'context.json').read_bytes(), before)

    async def test_duplicate_loading_and_ready_reuse_one_constructor(self):
        entered, finish = threading.Event(), threading.Event()
        def factory(*args, cancel_event, **kwargs):
            entered.set()
            finish.wait(2)
            return Adapter()
        self.factory.side_effect = factory
        try:
            responses = await asyncio.gather(self.runtime.load('small', 65536), self.runtime.load('small', 65536))
            self.assertEqual([r['state'] for r in responses], ['loading', 'loading'])
            await asyncio.to_thread(entered.wait, 1)
            with self.assertRaises(InferenceFailure) as error:
                await self.runtime.load('medium', 65536)
            self.assertEqual(error.exception.status_code, 409)
        finally:
            finish.set()
        await self.runtime.task
        self.assertEqual((await self.runtime.load('small', 65536))['state'], 'ready')
        self.assertEqual(self.factory.call_count, 1)

    async def test_active_generation_rejects_switch_without_mutation(self):
        old = await self.ready()
        async with self.runtime._generation:
            with self.assertRaises(InferenceFailure) as error:
                await self.runtime.load('medium', 32768)
        self.assertEqual(error.exception.status_code, 409)
        self.assertFalse(old.closed)
        self.assertEqual(self.models.get_selected()[0].id, 'small')

    async def test_persist_failure_restores_previous_settings_and_reports_unloaded(self):
        old = await self.ready()
        files = {name: (self.models.root / name).read_bytes() for name in ('context.json', 'selection.json')}
        with patch.object(self.models, 'select', side_effect=OSError('disk full')):
            with self.assertRaises(InferenceFailure) as error:
                await self.runtime.load('medium', 32768)
        self.assertEqual(error.exception.status_code, 503)
        self.assertTrue(old.closed)
        self.assertEqual(self.runtime.state, 'unloaded')
        self.assertFalse(self.models._in_use)
        for name, content in files.items():
            self.assertEqual((self.models.root / name).read_bytes(), content)
        self.assertEqual(self.factory.call_count, 1)

    async def test_factory_failure_is_visible_and_releases_selection(self):
        self.factory.side_effect = RuntimeError('native allocation failed')
        response = await self.runtime.load('medium', 32768)
        self.assertEqual(response['state'], 'loading')
        await self.runtime.task
        self.assertEqual(self.runtime.state, 'error')
        self.assertIn('native allocation failed', self.runtime.error)
        self.assertFalse(self.models._in_use)
        self.assertEqual(self.models.get_selected()[0].id, 'medium')
        self.assertEqual(self.models.configured_context('medium'), 32768)

    async def test_cancelled_switch_retains_cleanup_ownership_until_close_finishes(self):
        old = await self.ready()
        entered, finish = threading.Event(), threading.Event()
        def close():
            entered.set()
            finish.wait(3)
            old.closed = True
        old.close = Mock(side_effect=close)
        switching = asyncio.create_task(self.runtime.load('medium', 32768))
        await asyncio.to_thread(entered.wait, 1)
        switching.cancel()
        await asyncio.sleep(0)
        switching.cancel()
        competing = asyncio.create_task(self.runtime.unload())
        try:
            await asyncio.sleep(.02)
            self.assertFalse(switching.done())
            self.assertFalse(competing.done())
            self.assertEqual(old.close.call_count, 1)
            self.assertTrue(self.models._in_use)
        finally:
            finish.set()
        with self.assertRaises(asyncio.CancelledError):
            await switching
        await competing
        self.assertEqual(old.close.call_count, 1)
        self.assertEqual(self.runtime.state, 'unloaded')
        self.assertFalse(self.models._in_use)
        self.assertEqual(self.factory.call_count, 1)
        await self.ready('medium', context_limit=32768)

    async def test_switch_marks_unloading_before_cleanup_can_yield(self):
        await self.ready()
        entered, finish = asyncio.Event(), asyncio.Event()
        original = self.runtime._finish_unload
        async def paused_cleanup():
            entered.set()
            await finish.wait()
            return await original()
        with patch.object(self.runtime, '_finish_unload', paused_cleanup):
            switching = asyncio.create_task(self.runtime.load('medium', 32768))
            await entered.wait()
            try:
                self.assertEqual(self.runtime.state, 'unloading')
                with self.assertRaises(InferenceFailure):
                    await self.runtime.complete([], None)
            finally:
                finish.set()
            await switching
        await self.runtime.task
        self.assertEqual(self.runtime.model_id, 'medium')
