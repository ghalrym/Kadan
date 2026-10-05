import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from updater.__main__ import LocalConnection
from updater.server import Access, UnixServer, handler


class AccessTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.now = 1000
        self.access = Access(self.root, 'http://localhost:5173', clock=lambda: self.now)

    def test_explicit_origin_only_no_arbitrary_host_or_http_remote(self):
        for origin in ('http://evil.test', 'https://kadan.test/path', 'https://user@kadan.test', '*'):
            with self.assertRaises(ValueError): Access(self.root, origin)
        for headers in ({}, {'Host': 'localhost:5173'}, {'Origin': 'http://evil.test', 'X-Kadan-Update': '1'},
                        {'Origin': 'http://localhost:5173'},
                        {'Origin': 'http://localhost:5173', 'X-Kadan-Update': '1', 'Sec-Fetch-Site': 'cross-site'}):
            self.assertFalse(self.access.origin_ok(headers))
        self.assertTrue(self.access.origin_ok({'Origin': 'http://localhost:5173', 'X-Kadan-Update': '1'}))

    def test_one_time_pair_expiry_http_only_cookie_and_revoke(self):
        with patch('updater.server.secrets.token_urlsafe', side_effect=['test-code', 'test-session', 'next-code', 'next-session']):
            code = self.access.issue_code()
            cookie = self.access.pair(code)
            self.assertIn('HttpOnly; SameSite=Strict; Path=/v1/updates', cookie)
            self.assertTrue(self.access.authorized({'Cookie': cookie}))
            with self.assertRaises(ValueError): self.access.pair(code)
            self.access.pair(self.access.issue_code())
            self.assertFalse(self.access.authorized({'Cookie': cookie}))
            self.now += 43201
            self.assertFalse(self.access.authorized({'Cookie': 'kadan_updater=next-session'}))
        self.assertNotIn('test-session', (self.root / 'access.json').read_text())

    def test_code_timeout_and_attempt_limit(self):
        with patch('updater.server.secrets.token_urlsafe', return_value='test-code'):
            code = self.access.issue_code()
            self.now += 301
            with self.assertRaises(ValueError): self.access.pair(code)
            self.access.issue_code()
            for _ in range(8):
                with self.assertRaises(ValueError): self.access.pair('wrong')
            with self.assertRaisesRegex(ValueError, 'locked'): self.access.pair(code)

    def test_real_unix_http_protocol_rejects_unauthenticated_and_csrf_install(self):
        engine = Mock(status=Mock(return_value={'current': 'a' * 40}))
        path = self.root / 'control.sock'
        server = UnixServer(str(path), handler(engine, self.access))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            def post(path, body, headers=None):
                with LocalConnection(self.root / 'control.sock') as connection:
                    connection.request('POST', path, json.dumps(body),
                        {'Content-Type': 'application/json', **(headers or {})})
                    response = connection.getresponse()
                    return response.status, dict(response.getheaders()), json.load(response)
            trusted = {'Origin': self.access.origin, 'X-Kadan-Update': '1'}
            self.assertEqual(post('/install', {'commit': 'a' * 40})[0], 403)
            self.assertEqual(post('/install', {'commit': 'a' * 40}, trusted)[0], 401)
            with patch('updater.server.secrets.token_urlsafe', side_effect=['test-code', 'test-session']):
                code = self.access.issue_code()
                status, headers, body = post('/pair', {'code': code}, trusted)
            self.assertEqual(status, 200)
            paired = {**trusted, 'Cookie': headers['Set-Cookie']}
            self.assertEqual(post('/install', {'commit': 'a' * 40}, paired)[0], 202)
            engine.start.assert_called_once_with('a' * 40)
            self.assertEqual(post('/install', {'commit': 'a' * 40}, {**paired, 'Origin': 'https://evil.test'})[0], 403)
            self.assertEqual(post('/shell', {'command': 'docker ps'}, paired)[0], 404)
            self.assertEqual(post('/recover', None, paired)[0], 404)
            self.assertEqual(post('/install', {'commit': 'a' * 40, 'image': 'evil'}, paired)[0], 409)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)
