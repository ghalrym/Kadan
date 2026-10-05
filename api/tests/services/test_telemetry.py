import asyncio
import json
import unittest
from unittest.mock import mock_open, patch

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.services.telemetry import TelemetryMiddleware, TelemetryStore, MAX_BODY_SUMMARY_BYTES, summarize
from api.routes.v1 import requests, metrics
from api.tests.routes.v1.test_requests import record


class Body(BaseModel):
    messages: list[dict]


class TelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = TelemetryStore(3)
        self.app = FastAPI()
        self.app.add_middleware(TelemetryMiddleware, store=self.store)
        self.app.include_router(requests.router)
        self.app.include_router(metrics.router)
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.mode = 'success'

        @self.app.post('/v1/chat/completions')
        async def chat(body: Body):
            if self.mode == 'unavailable': raise HTTPException(503, 'PRIVATE FAILURE DETAIL')
            if self.mode == 'exception': raise RuntimeError('PRIVATE TRACEBACK')
            if self.mode == 'wait':
                self.entered.set()
                await self.release.wait()
            return {'message': {'role': 'assistant', 'text': 'PRIVATE OUTPUT'}}

        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://test')
        self.addAsyncCleanup(self.client.aclose)
        for target in ('api.routes.v1.requests.telemetry', 'api.routes.v1.metrics.telemetry'):
            patcher = patch(target, self.store)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_real_statuses_and_private_payloads_not_retained(self):
        body = {'messages': [{'text': 'PRIVATE PROMPT', 'role': 'user'}], 'password': 'SECRET'}
        response = await self.client.post('/v1/chat/completions', json=body)
        self.assertEqual(response.status_code, 200)
        self.mode = 'unavailable'
        self.assertEqual((await self.client.post('/v1/chat/completions', json=body)).status_code, 503)
        self.assertEqual((await self.client.post('/v1/chat/completions', json={'password': 'SECRET'})).status_code, 422)
        records = self.store.records()
        self.assertEqual([r.status for r in records], [422, 503, 200])
        encoded = json.dumps([r.model_dump() for r in records])
        for secret in ('PRIVATE', 'SECRET', 'password'):
            self.assertNotIn(secret, encoded)
        self.assertTrue(all(r.latency_ms >= 0 and r.request_bytes > 0 and r.response_bytes > 0 for r in records))
        self.assertIn('1 messages', records[-1].prompt)

    async def test_unhandled_error_is_observed_and_propagated(self):
        self.mode = 'exception'
        with self.assertRaisesRegex(RuntimeError, 'PRIVATE TRACEBACK'):
            await self.client.post('/v1/chat/completions', json={'messages': []})
        self.assertEqual(self.store.records()[0].status, 500)
        self.assertEqual(self.store.metrics()['active_requests'], 0)

    async def test_active_and_cancelled_request_cleanup(self):
        self.mode = 'wait'
        task = asyncio.create_task(self.client.post('/v1/chat/completions', json={'messages': []}))
        await self.entered.wait()
        self.assertEqual(self.store.metrics()['active_requests'], 1)
        self.assertEqual(self.store.records(), [])
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertEqual(self.store.records()[0].status, 499)
        self.assertEqual(self.store.metrics()['active_requests'], 0)

    async def test_monitoring_reads_never_log_themselves(self):
        for _ in range(3):
            await self.client.get('/v1/requests')
            with patch('api.routes.v1.metrics.memory_meters', return_value=([], [])):
                await self.client.get('/v1/metrics')
            await self.client.get('/v1/requests/missing')
        self.assertEqual(self.store.records(), [])
        self.assertEqual(self.store.metrics()['completed_requests'], 0)

    async def test_http_filter_detail_and_validation(self):
        for _ in range(2): await self.client.post('/v1/chat/completions', json={'messages': []})
        data = (await self.client.get('/v1/requests?status=200&search=CHAT&offset=1&limit=1')).json()
        self.assertEqual(data['total'], 2)
        self.assertEqual(len(data['requests']), 1)
        item = data['requests'][0]
        self.assertEqual((await self.client.get('/v1/requests/' + item['id'])).json()['request'], item)
        self.assertEqual((await self.client.get('/v1/requests?status=600')).status_code, 422)
        self.assertEqual(self.store.metrics()['completed_requests'], 2)

    async def test_large_body_is_forwarded_but_never_retained(self):
        response = await self.client.post('/v1/chat/completions', json={'messages': [{'text': 'x' * 100_000}]})
        self.assertEqual(response.status_code, 200)
        item = self.store.records()[0]
        self.assertGreater(item.request_bytes, MAX_BODY_SUMMARY_BYTES)
        self.assertIn('summary limit', item.prompt)
        self.assertLess(len(item.model_dump_json()), 1000)


class SummaryAndMetricsTests(unittest.TestCase):
    def test_structural_summary_only_and_model_allowlist(self):
        body = json.dumps({'messages': [{'text': 'secret'}], 'model': 'small'}).encode()
        summary, model = summarize(body, len(body), True)
        self.assertEqual(model, 'small')
        self.assertNotIn('secret', summary)
        self.assertIsNone(summarize(b'{"model":"private-custom-model"}', 32, True)[1])
        self.assertIn('malformed', summarize(b'{', 1, True)[0])
        self.assertIn('incomplete', summarize(b'', 0, False)[0])
        self.assertIn('malformed', summarize(b'[' * 2000, 2000, True)[0])

    def test_retention_window_and_no_empty_fake_statistics(self):
        store = TelemetryStore(2)
        empty = store.metrics(now=100)
        self.assertIsNone(empty['p50_latency_seconds'])
        self.assertIsNone(empty['error_rate_percent'])
        for i, status in enumerate((200, 422, 503)):
            store.begin()
            store.finish(record(str(i), status=status), now=100 + i)
        snapshot = store.metrics(now=103)
        self.assertTrue(snapshot['window_truncated'])
        self.assertEqual(snapshot['completed_requests'], 3)
        self.assertEqual(snapshot['requests_per_minute'], 2)
        self.assertEqual(snapshot['error_rate_percent'], 100)
        self.assertEqual(snapshot['p50_latency_seconds'], .1)
        self.assertFalse(store.metrics(now=163)['window_truncated'])
        self.assertEqual(store.metrics(now=163)['requests_per_minute'], 0)
        self.assertEqual(TelemetryStore(2).records(), [])

    def test_host_meter_and_absent_optional_torch(self):
        with patch('builtins.open', mock_open(read_data='MemTotal: 2097152 kB\nMemAvailable: 524288 kB\n')), patch.dict('sys.modules', {'torch': None}):
            meters, errors = metrics.memory_meters()
        self.assertEqual(meters[0].label, 'Host RAM')
        self.assertEqual(meters[0].total, 2)
        self.assertEqual(meters[0].used, 1.5)
        self.assertTrue(errors)
