import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from fastapi import HTTPException, Request

from api.server import app
from api.services.model_downloads import ModelManager
from api.services.runtime import RuntimeManager
from api.routes.v1.chat.completions import CompletionRequest, create_completion


class ModelLifecycleRouteTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.models = ModelManager(Path(temporary.name))
        self.runtime = RuntimeManager()
        for target, value in (
            ('api.routes.model_lifecycle.runtime_manager', self.runtime),
            ('api.routes.v1.chat.completions.runtime_manager', self.runtime),
            ('api.services.model_downloads.model_manager', self.models),
            ('api.server.runtime_manager', self.runtime),
            ('api.server.model_manager', self.models),
        ):
            patched = patch(target, value)
            patched.start()
            self.addCleanup(patched.stop)
        self.client = TestClient(app)

    def test_chat_never_falls_back_to_mock_when_unloaded(self):
        response = self.client.post('/v1/chat/completions', json={
            'messages': [{'role': 'user', 'text': 'Hello'}],
        })
        self.assertEqual(response.status_code, 503)
        self.assertIn('No model is ready', response.json()['detail'])
        self.assertNotIn('message', response.json())

    def test_control_routes_are_outside_versioned_inference_api(self):
        paths = self.client.get('/openapi.json').json()['paths']
        self.assertIn('/model-lifecycle', paths)
        self.assertIn('/model-lifecycle/load', paths)
        self.assertIn('/model-lifecycle/unload', paths)
        self.assertNotIn('/v1/runtime', paths)
        self.assertEqual(self.client.get('/v1/runtime').status_code, 404)

    def test_load_requires_selection_and_unload_is_idempotent(self):
        self.assertEqual(self.client.get('/model-lifecycle').json()['state'], 'unloaded')
        response = self.client.post('/model-lifecycle/load')
        self.assertEqual(response.status_code, 409)
        self.assertIn('select', response.json()['detail'])
        for _ in range(2):
            self.assertEqual(self.client.post('/model-lifecycle/unload').json()['state'], 'unloaded')

    def test_completion_contract_and_limits(self):
        self.runtime.complete = AsyncMock(return_value='Controlled runtime reply')
        response = self.client.post('/v1/chat/completions', json={
            'messages': [{'role': 'user', 'text': 'Hello'}],
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['message']['text'], 'Controlled runtime reply')
        args = self.runtime.complete.await_args.args
        self.assertEqual(args[0][0].text, 'Hello')
        self.assertIsNone(args[1])
        oversized = self.client.post('/v1/chat/completions', json={
            'messages': [{'role': 'user', 'text': 'x' * 32769}],
        })
        self.assertEqual(oversized.status_code, 200)
        self.assertEqual(len(self.runtime.complete.await_args.args[0][0].text), 32769)
        many = self.client.post('/v1/chat/completions', json={
            'messages': [{'role': 'user', 'text': 'Hi'}] * 129,
        })
        self.assertEqual(many.status_code, 200)
        self.assertEqual(len(self.runtime.complete.await_args.args[0]), 129)

    def test_api_lifespan_closes_runtime_and_download_manager(self):
        self.runtime.close = AsyncMock()
        with patch.object(self.models, 'close') as close, TestClient(app) as client:
            self.assertEqual(client.get('/health').status_code, 200)
        self.runtime.close.assert_awaited_once()
        close.assert_called_once()


class ChatDisconnectTests(unittest.IsolatedAsyncioTestCase):
    async def test_disconnected_client_cancels_and_awaits_generation_cleanup(self):
        started, cleaned = asyncio.Event(), asyncio.Event()

        async def complete(*args):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cleaned.set()

        async def receive():
            await started.wait()
            return {'type': 'http.disconnect'}

        request = Request({'type': 'http'}, receive=receive)
        body = CompletionRequest(messages=[{'role': 'user', 'text': 'Hello'}])
        with patch('api.routes.v1.chat.completions.runtime_manager.complete', complete):
            with self.assertRaises(HTTPException) as error:
                await asyncio.wait_for(create_completion(body, request), timeout=2)
        self.assertEqual(error.exception.status_code, 499)
        self.assertTrue(cleaned.is_set())
