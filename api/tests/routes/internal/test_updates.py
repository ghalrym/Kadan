import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import httpx

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.internal import updates
from api.routes.v1 import updates as public
from api.services.maintenance import Admission, MaintenanceMiddleware


class UpdateRoutesTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(updates.router)
        self.app.include_router(public.router)
        self.client = TestClient(self.app)

    def test_unconfigured_install_and_internal_control_fail_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(self.client.get('/v1/updates').json()['configured'])
            self.assertEqual(self.client.post('/v1/updates/install', json={'commit': 'a' * 40}).status_code, 503)
            self.assertEqual(self.client.post('/internal/updates/drain').status_code, 403)
            self.assertEqual(self.client.get('/internal/updates/ready').status_code, 403)
        self.assertEqual(self.client.post('/v1/updates/install', json={'commit': 'a' * 40, 'image': 'evil'}).status_code, 422)

    def test_maintenance_blocks_mutations_but_not_status(self):
        @self.app.post('/v1/example')
        def mutate():
            return {'ok': True}
        self.app.add_middleware(MaintenanceMiddleware)
        gate = Admission()
        gate.seal()
        with patch('api.services.maintenance.admission', gate):
            self.assertEqual(self.client.post('/v1/example').status_code, 503)
            self.assertEqual(self.client.get('/v1/updates').status_code, 200)

    def test_readiness_checks_database_instead_of_unconditional_health(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'host-key'
            path.write_text('test-only-host-key')
            with patch.dict(os.environ, KADAN_HOST_KEY_FILE=str(path)), \
                 patch.object(updates, 'get_engine', side_effect=RuntimeError('database down')):
                response = self.client.get('/internal/updates/ready', headers={'X-Kadan-Host': 'test-only-host-key'})
                self.assertEqual(response.status_code, 503)

    def test_cancel_bridge_keeps_json_csrf_and_cookie_headers_without_execution_input(self):
        response = httpx.Response(200, json={'current': 'a' * 40, 'configured': True, 'phase': 'draining'})
        transport = Mock()
        transport.__enter__ = Mock(return_value=transport)
        transport.__exit__ = Mock(return_value=False)
        transport.request.return_value = response
        with patch.dict(os.environ, KADAN_UPDATER_SOCKET='/unused/test.sock'), \
             patch.object(public.httpx, 'Client', return_value=transport):
            result = self.client.post('/v1/updates/cancel', json=None,
                headers={'Origin': 'http://localhost:5173', 'X-Kadan-Update': '1', 'Cookie': 'kadan_updater=fixture'})
        self.assertEqual(result.status_code, 200)
        arguments = transport.request.call_args.kwargs
        self.assertEqual(arguments['headers']['content-type'], 'application/json')
        self.assertEqual(arguments['headers']['x-kadan-update'], '1')
        self.assertEqual(arguments['headers']['cookie'], 'kadan_updater=fixture')
        self.assertIsNone(arguments['json'])
