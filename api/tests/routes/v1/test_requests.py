import unittest
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError
from api.pydantic_models.requests import RequestRecord
from api.routes.v1.requests import RequestFilters, get_request, list_requests
from api.services.telemetry import TelemetryStore


def record(identifier, status=200, kind='LLM', endpoint='/v1/chat/completions'):
    return RequestRecord(id=identifier, time='2026-10-04T12:00:00+00:00', type=kind,
                         status=status, latency='0.100 s', latency_ms=100, endpoint=endpoint,
                         prompt='JSON object; 2 messages', output=f'HTTP {status}',
                         request_bytes=20, response_bytes=10)


class RequestFilteringTests(unittest.TestCase):
    def setUp(self):
        self.store = TelemetryStore(capacity=3)
        patcher = patch('api.routes.v1.requests.telemetry', self.store)
        patcher.start()
        self.addCleanup(patcher.stop)
        for item in [record('old'), record('ok'), record('bad', 422), record('latest', 503)]:
            self.store.begin()
            self.store.finish(item)

    def test_any_valid_http_status_and_bounds(self):
        self.assertEqual(RequestFilters.model_validate({'status': '201'}).status, 201)
        self.assertEqual(RequestFilters.model_validate({'status': '503'}).status, 503)
        for fields in ({'status': 600}, {'status': 99}, {'limit': 101}, {'offset': -1}, {'search': 'x' * 129}):
            with self.assertRaises(ValidationError): RequestFilters.model_validate(fields)

    def test_filters_are_combined_before_pagination(self):
        result = list_requests(RequestFilters(type='LLM', search='CHAT', limit=1, offset=1))
        self.assertEqual(result.total, 3)
        self.assertEqual([r.id for r in result.requests], ['bad'])
        self.assertEqual(result.retention_limit, 3)
        self.assertEqual([r.id for r in list_requests(RequestFilters(status=503)).requests], ['latest'])

    def test_detail_and_eviction(self):
        self.assertEqual(get_request('bad').request.status, 422)
        with self.assertRaises(HTTPException) as caught: get_request('old')
        self.assertEqual(caught.exception.status_code, 404)

    def test_empty_is_not_fixture_data(self):
        with patch('api.routes.v1.requests.telemetry', TelemetryStore()):
            result = list_requests(RequestFilters())
        self.assertEqual(result.requests, [])
        self.assertEqual(result.total, 0)
