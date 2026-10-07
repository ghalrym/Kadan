import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI

from api.routes.v1.chat.completions import CompletionRequest, OwnedStreamResponse, chunks, router
from api.services.runtime import RuntimeFailure


class CompletionTests(unittest.IsolatedAsyncioTestCase):
    async def test_json_legacy_and_content_input(self):
        app = FastAPI()
        app.include_router(router)
        manager = SimpleNamespace(runtime=SimpleNamespace(model_id='small'),
            submit=AsyncMock(return_value={'text': 'Hello', 'finish_reason': 'length',
                'cache': {'hit':True,'reused_tokens':20,'stored_tokens':20,'host_bytes':70000,'device_bytes':{},'reason':'hit'}}))
        with patch('api.routes.v1.chat.completions.memory_manager', manager):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
                for key in ('text', 'content'):
                    response = await client.post('/v1/chat/completions', json={'messages': [{'role': 'user', key: 'Hi'}]})
                    self.assertEqual(response.status_code, 200)
                    data = response.json()
                    self.assertEqual(data['message']['text'], 'Hello')
                    self.assertEqual(data['choices'][0]['message']['content'], 'Hello')
                    self.assertEqual(data['choices'][0]['finish_reason'], 'length')
                    self.assertEqual(data['object'], 'chat.completion')
                    self.assertEqual(data['cache']['reused_tokens'],20)

    async def test_chunks_are_incremental_and_terminal_waits_for_cleanup(self):
        release = asyncio.Event()
        async def source():
            yield {'content': 'First '}
            await release.wait()
            yield {'content': 'second'}
            yield {'cache': {'hit':True,'reused_tokens':20}}
            yield {'finish_reason': 'stop'}
        stream = chunks(source(), {'id': 'chatcmpl-test', 'created': 1, 'model': 'small'})
        self.assertIn('assistant', await anext(stream))
        self.assertIn('First ', await anext(stream))
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(.01)
        self.assertFalse(pending.done())
        release.set()
        self.assertIn('second', await pending)
        terminal = json.loads((await anext(stream)).removeprefix('data: '))
        self.assertEqual(terminal['cache']['reused_tokens'],20)
        self.assertEqual(terminal['choices'][0], {'index': 0, 'delta': {}, 'finish_reason': 'stop'})
        self.assertEqual(await anext(stream), 'data: [DONE]\n\n')

    async def test_error_after_content_is_not_success(self):
        async def source():
            yield {'content': 'Partial'}
            raise RuntimeFailure('failed', 503)
        events = [event async for event in chunks(source(), {'id': 'test'})]
        self.assertIn('"error"', events[-2])
        self.assertNotIn('"finish_reason": "stop"', ''.join(events))
        self.assertEqual(events[-1], 'data: [DONE]\n\n')

    async def test_disconnect_awaits_owner_cleanup(self):
        disconnected = asyncio.Event()
        cleaned = asyncio.Event()
        async def source():
            yield 'data: partial\n\n'
            await asyncio.Event().wait()
        async def close():
            await asyncio.sleep(.03)
            cleaned.set()
        owner = SimpleNamespace(aclose=close)
        async def receive():
            await disconnected.wait()
            return {'type': 'http.disconnect'}
        async def send(message):
            if message['type'] == 'http.response.body':
                disconnected.set()
        response = OwnedStreamResponse(source(), owner)
        await asyncio.wait_for(response({'type': 'http', 'asgi': {'spec_version': '2.0'}}, receive, send), 1)
        self.assertTrue(cleaned.is_set())

    def test_defaults_and_validation(self):
        self.assertFalse(CompletionRequest(messages=[{'role':'user','content':'Hi'}]).stream)
        with self.assertRaises(ValueError):
            CompletionRequest(messages=[])
