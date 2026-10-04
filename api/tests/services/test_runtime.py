import asyncio
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from api.inference.resources import ResourceManager
from api.pydantic_models.chat import ChatMessage
from api.services.runtime import RuntimeFailure, RuntimeManager


class Adapter:
    def __init__(self):
        self.is_resident = True
        self.closed = False
        self.calls = []

    def configure_context(self, value):
        self.configured_context_limit = value

    def generate(self, messages, max_new_tokens, cancel_event):
        self.calls.append(messages)
        self.is_resident = True
        return 'Controlled adapter response'

    def close(self):
        self.closed = True
        self.is_resident = False


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.adapter = Adapter()
        self.factory = Mock(return_value=self.adapter)
        self.manager = RuntimeManager(self.factory, ResourceManager(1000, {0: 1000}))
        context_patch = patch('api.services.runtime.read_context_settings', return_value=dict(
            configured_context_limit=None, effective_context_limit=131072, supported_context_limit=131072))
        context_patch.start()
        self.addCleanup(context_patch.stop)
        self.models = Mock()
        self.models.configured_context.return_value = None
        self.models.acquire_runtime_model.return_value = (SimpleNamespace(id='medium'), Path('/models/pinned'))
        patched = patch('api.services.model_downloads.model_manager', self.models)
        patched.start()
        self.addCleanup(patched.stop)

    async def ready(self):
        await self.manager.load()
        await self.manager.task
        self.assertEqual(self.manager.state, 'ready')

    async def test_context_errors_preserve_ready_model_and_selection(self):
        from api.inference.context import ContextLimitError, ContextMemoryError
        await self.ready()
        for error, status in ((ContextLimitError('too many tokens'), 422), (ContextMemoryError('does not fit'), 503)):
            self.adapter.generate = Mock(side_effect=error)
            with self.assertRaises(RuntimeFailure) as caught:
                await self.manager.complete([], None)
            self.assertEqual(caught.exception.status_code, status)
            self.assertEqual(self.manager.state, 'ready')
            self.assertFalse(self.adapter.closed)
            self.models.release_runtime_model.assert_not_called()
        self.assertEqual(self.manager.status()['effective_context_limit'], 131072)
        await self.manager.close()

    async def test_direct_adapter_receives_owned_resources_and_selected_path(self):
        await self.ready()
        args, kwargs = self.factory.call_args
        self.assertEqual(args[0].id, 'medium')
        self.assertEqual(args[1], Path('/models/pinned'))
        self.assertIs(args[2], self.manager.resources)
        self.assertEqual(kwargs['device'], 'cuda:0')
        answer = await self.manager.complete([ChatMessage(role='user', text='Hello')], None)
        self.assertEqual(answer, 'Controlled adapter response')
        self.assertEqual(self.adapter.calls, [[{'role': 'user', 'text': 'Hello'}]])
        await self.manager.unload()
        self.assertTrue(self.adapter.closed)
        self.models.release_runtime_model.assert_called_once()

    async def test_unloaded_and_wrong_model_fail(self):
        with self.assertRaises(RuntimeFailure):
            await self.manager.complete([], None)
        await self.ready()
        with self.assertRaises(RuntimeFailure) as error:
            await self.manager.complete([], 'small')
        self.assertEqual(error.exception.status_code, 409)
        await self.manager.close()

    async def test_load_failure_never_reports_ready_and_releases_selection(self):
        self.factory.side_effect = ValueError('Unsupported checkpoint tensor layout')
        await self.manager.load()
        await self.manager.task
        self.assertEqual(self.manager.state, 'error')
        self.assertIn('Unsupported', self.manager.error)
        self.models.release_runtime_model.assert_called_once()

    async def test_load_cancel_waits_for_loader_before_closing(self):
        entered = threading.Event()
        def build(*args, cancel_event, **kwargs):
            entered.set()
            cancel_event.wait(2)
            return self.adapter
        self.factory.side_effect = build
        await self.manager.load()
        await asyncio.to_thread(entered.wait, 1)
        await self.manager.unload()
        self.assertTrue(self.adapter.closed)
        self.assertEqual(self.manager.state, 'unloaded')

    async def test_concurrent_generation_rejected(self):
        await self.ready()
        async with self.manager._generation:
            with self.assertRaises(RuntimeFailure) as error:
                await self.manager.complete([], None)
        self.assertEqual(error.exception.status_code, 429)
        await self.manager.close()

    async def test_offloaded_model_restores_on_next_generation(self):
        await self.ready()
        self.adapter.is_resident = False
        self.assertEqual(self.manager.status()['state'], 'offloaded')
        await self.manager.complete([], None)
        self.assertEqual(self.manager.status()['state'], 'ready')
        await self.manager.close()

    async def test_cancelled_generation_finishes_before_close(self):
        entered, finished = threading.Event(), threading.Event()
        def generate(*args, cancel_event, **kwargs):
            entered.set()
            cancel_event.wait(2)
            finished.set()
            return 'cancelled result must not escape'
        self.adapter.generate = generate
        self.adapter.close = Mock(side_effect=lambda: self.assertTrue(finished.is_set()))
        await self.ready()
        task = asyncio.create_task(self.manager.complete([], None))
        await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.adapter.close.assert_called_once()
        self.assertEqual(self.manager.state, 'unloaded')

    async def test_bad_response_is_error_and_actual_adapter_is_closed(self):
        self.adapter.generate = Mock(return_value='')
        await self.ready()
        with self.assertRaises(RuntimeFailure):
            await self.manager.complete([], None)
        self.assertEqual(self.manager.state, 'error')
        self.assertTrue(self.adapter.closed)

    async def test_cleanup_failure_retains_adapter_for_retry(self):
        await self.ready()
        self.adapter.close = Mock(side_effect=[ValueError('cleanup failed'), None])
        with self.assertRaises(ValueError):
            await self.manager.unload()
        self.assertIs(self.manager.adapter, self.adapter)
        self.models.release_runtime_model.assert_not_called()
        await self.manager.unload()
        self.assertIsNone(self.manager.adapter)

    async def test_multiple_gpu_selection_rejected(self):
        with patch.dict('os.environ', {'KADAN_GPU': '0,1'}):
            await self.manager.load()
            await self.manager.task
        self.assertEqual(self.manager.state, 'error')
        self.factory.assert_not_called()
