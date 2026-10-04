import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from api.pydantic_models.chat import ChatMessage
from api.services.runtime import RuntimeFailure, RuntimeManager


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_command_enforces_partial_expert_cache_single_gpu_and_loopback(self):
        manager = RuntimeManager()
        command = manager.command(Path('/models/small'), 'small', 12345)
        self.assertIn('offload', command)
        self.assertIn('--moe-cache-auto', command)
        self.assertEqual(command[command.index('--host') + 1], '127.0.0.1')
        self.assertIn('moe.nvfp4=triton', command)
        self.assertNotIn('--tp', command)
        self.assertNotIn('--moe-cpu-layers', command)
        with patch.dict('os.environ', {'KADAN_FT_GPU': '0,1'}):
            with self.assertRaises(RuntimeFailure):
                manager.command(Path('/models/small'), 'small', 12345)

    async def test_missing_runtime_and_wrong_model_fail(self):
        manager = RuntimeManager()
        with self.assertRaises(RuntimeFailure):
            await manager.complete([], None)
        manager.state, manager.model_id = 'ready', 'small'
        manager.process = SimpleNamespace(returncode=None)
        with self.assertRaises(RuntimeFailure) as error:
            await manager.complete([], 'medium')
        self.assertEqual(error.exception.status_code, 409)

    async def test_chat_translation_through_http_transport(self):
        manager = RuntimeManager()
        manager.state, manager.model_id, manager.port = 'ready', 'small', 12345
        manager.process = SimpleNamespace(returncode=None)
        def handle(request):
            import json
            body = json.loads(request.content)
            self.assertEqual(body['messages'], [{'role': 'user', 'content': 'hello'}])
            self.assertFalse(body['stream'])
            return httpx.Response(200, json={'choices': [{'message': {'content': 'real response fixture'}}]})
        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        with patch('api.services.runtime.httpx.AsyncClient', return_value=client):
            self.assertEqual(await manager.complete([ChatMessage(role='user', text='hello')], None), 'real response fixture')

    async def test_concurrent_generation_rejected(self):
        manager = RuntimeManager()
        manager.state, manager.model_id = 'ready', 'small'
        manager.process = SimpleNamespace(returncode=None)
        async with manager._generation:
            with self.assertRaises(RuntimeFailure) as error:
                await manager.complete([], None)
            self.assertEqual(error.exception.status_code, 429)

    async def test_bad_upstream_unloads_instead_of_fake_success(self):
        manager = RuntimeManager()
        manager.state, manager.model_id, manager.port = 'ready', 'small', 12345
        manager.process = SimpleNamespace(returncode=None)
        manager.unload = AsyncMock()
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
        with patch('api.services.runtime.httpx.AsyncClient', return_value=client):
            with self.assertRaises(RuntimeFailure):
                await manager.complete([], None)
        manager.unload.assert_awaited_once()
        self.assertEqual(manager.state, 'error')

    async def test_process_missing_sets_error_and_releases_lease(self):
        manager = RuntimeManager()
        manager.model_id, manager.state = 'small', 'loading'
        manager._release = Mock()
        with patch('asyncio.create_subprocess_exec', side_effect=FileNotFoundError('ft missing')):
            await manager._run(Path('/models/small'))
        self.assertEqual(manager.state, 'error')
        self.assertIn('ft missing', manager.error)
        manager._release.assert_called_once()

    async def test_unload_cancels_loading_and_kills_owned_process_group(self):
        manager = RuntimeManager()
        process = SimpleNamespace(pid=87654, wait=AsyncMock(return_value=0))
        manager.process = process
        manager.task = asyncio.create_task(asyncio.sleep(3600))
        with patch('os.killpg') as kill:
            await manager.unload()
        self.assertEqual(kill.call_count, 2)
        self.assertTrue(manager.task is None)
        self.assertEqual(manager.status(), dict(state='unloaded', model_id=None, error=None))

    async def test_ready_only_after_matching_model_then_process_failure_cleans_up(self):
        manager = RuntimeManager()
        manager.state, manager.model_id = 'loading', 'small'
        process = SimpleNamespace(pid=87654, returncode=None, wait=AsyncMock(return_value=1))
        def handle(request):
            if request.url.path == '/health':
                return httpx.Response(200, json={'status': 'ok'})
            return httpx.Response(200, json={'data': [{'id': 'small'}]})
        client = httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url='http://127.0.0.1')
        async def tick(seconds):
            self.assertEqual(manager.state, 'ready')
            process.returncode = 1
        with patch('asyncio.create_subprocess_exec', AsyncMock(return_value=process)), \
             patch('api.services.runtime.httpx.AsyncClient', return_value=client), \
             patch('asyncio.sleep', tick), patch('os.killpg'):
            await manager._run(Path('/models/small'))
        self.assertEqual(manager.state, 'error')
        self.assertIsNone(manager.process)

    async def test_load_blocked_until_failed_worker_cleanup_finishes(self):
        manager = RuntimeManager()
        manager.state = 'error'
        manager.task = asyncio.create_task(asyncio.sleep(3600))
        # Module may not yet be present in this independent worktree.
        import sys
        stub = SimpleNamespace(model_manager=Mock())
        try:
            with patch.dict(sys.modules, {'api.services.model_downloads': stub}):
                with self.assertRaises(RuntimeFailure) as error:
                    await manager.load()
            self.assertEqual(error.exception.status_code, 409)
            stub.model_manager.acquire_runtime_model.assert_not_called()
        finally:
            manager.task.cancel()
            try:
                await manager.task
            except asyncio.CancelledError:
                pass

    async def test_cancellation_during_spawn_still_reaps_process(self):
        manager = RuntimeManager()
        manager.state, manager.model_id = 'loading', 'small'
        entered, finish = asyncio.Event(), asyncio.Event()
        process = SimpleNamespace(pid=87654, returncode=None, wait=AsyncMock(return_value=0))
        async def spawn(*args, **kwargs):
            entered.set()
            await finish.wait()
            return process
        with patch('asyncio.create_subprocess_exec', spawn), patch('os.killpg') as kill:
            task = asyncio.create_task(manager._run(Path('/models/small')))
            await entered.wait()
            task.cancel()
            finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(kill.call_count, 2)
        self.assertIsNone(manager.process)
