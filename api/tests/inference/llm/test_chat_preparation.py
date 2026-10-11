import asyncio
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from api.inference.errors import InferenceFailure
from api.inference.llm.chat_requests import ChatRequests

class PreparationTests(unittest.IsolatedAsyncioTestCase):
    def runtime(self):
        return SimpleNamespace(state='unloaded',model_id=None,adapter=object(),task=None,_cancel=threading.Event())
    async def test_prepare_then_load_before_completion(self):
        runtime=self.runtime()
        async def load(model):
            runtime.state='ready';runtime.model_id=model
        runtime.load=AsyncMock(side_effect=load)
        runtime.complete=AsyncMock(return_value='mock result')
        manager=SimpleNamespace(_language_entry=Mock(return_value=SimpleNamespace(inference_available=True)),ensure_checkpoint=Mock())
        with patch('api.inference.llm.chat_requests.model_manager',manager):
            await ChatRequests(runtime)(SimpleNamespace(model='small',messages=[]))
        manager.ensure_checkpoint.assert_called_once()
        runtime.load.assert_awaited_once_with('small')
        runtime.complete.assert_awaited_once()
    async def test_completion_returns_frozen_model_after_automatic_load(self):
        runtime=self.runtime()
        wrapper=ChatRequests(runtime)
        wrapper.load=AsyncMock()
        async def complete(messages, model, on_event):
            runtime.model_id='another-selection'
            on_event({'finish_reason':'stop'})
            return 'mock result'
        runtime.complete=complete
        request=SimpleNamespace(model=None,messages=[],conversation_id=None,reuse_prefix=False)
        result=await wrapper(request, model='large', operation='completion')
        self.assertEqual(result['model'],'large')
        wrapper.load.assert_awaited_once_with('large')

    async def test_unsupported_architecture_rejected_before_download(self):
        manager=SimpleNamespace(_language_entry=Mock(return_value=SimpleNamespace(inference_available=False,repo_id='GLM')),ensure_checkpoint=Mock())
        with patch('api.inference.llm.chat_requests.model_manager',manager), self.assertRaisesRegex(InferenceFailure,'no native'):
            await ChatRequests(self.runtime()).load('large')
        manager.ensure_checkpoint.assert_not_called()
    async def test_cancellation_signals_and_joins_loader(self):
        runtime=self.runtime();started=asyncio.Event()
        async def pending():
            started.set();await asyncio.Event().wait()
        async def load(model):
            runtime.task=asyncio.create_task(pending())
        runtime.load=load
        async def unload():
            runtime.state='unloaded';runtime.task=None
        runtime.unload=AsyncMock(side_effect=unload)
        manager=SimpleNamespace(_language_entry=Mock(return_value=SimpleNamespace(inference_available=True)),ensure_checkpoint=Mock())
        with patch('api.inference.llm.chat_requests.model_manager',manager):
            job=asyncio.create_task(ChatRequests(runtime).load('small'))
            await started.wait();job.cancel()
            with self.assertRaises(asyncio.CancelledError):await job
        self.assertTrue(runtime._cancel.is_set())
        self.assertIsNone(runtime.task)
        self.assertEqual(runtime.state, 'unloaded')
        runtime.unload.assert_awaited_once()
