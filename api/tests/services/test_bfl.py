import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx

from api.inference.resources import ResourceCancelled
from api.services.bfl import BflClient, BflFailure, polling_url


class BflTests(unittest.TestCase):
    def test_submit_once_poll_and_never_forward_key_to_media(self):
        requests = []
        def handler(request):
            requests.append(request)
            if request.method == 'POST':
                return httpx.Response(200, json={'polling_url': 'https://api.bfl.ai/v1/get_result?id=test'})
            if request.url.host == 'api.bfl.ai':
                return httpx.Response(200, json={'status': 'Ready', 'result': {'sample': 'https://media.example.test/file'}})
            self.assertNotIn('x-key', request.headers)
            return httpx.Response(200, content=b'result')
        with patch.dict(os.environ, {'BFL_API_KEY': 'test-only'}), httpx.Client(transport=httpx.MockTransport(handler)) as http:
            client = BflClient(http)
            cancel = threading.Event()
            url = client.generate('flux-3-image', {'prompt': 'test'}, cancel)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'result'
                client.download(url, path, cancel, 20)
                self.assertEqual(path.read_bytes(), b'result')
        self.assertEqual(sum(r.method == 'POST' for r in requests), 1)

    def test_bad_poll_hosts_are_rejected(self):
        for url in ('http://api.bfl.ai/v1/get_result', 'https://api.bfl.ai.evil.test/v1/get_result',
                    'https://user:secret@api.bfl.ai/v1/get_result', 'https://api.bfl.ai:8443/v1/get_result'):
            with self.assertRaises(BflFailure):
                polling_url(url)

    def test_failure_does_not_retry_paid_submission_or_leak_details(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(500, text='secret-upstream-message')
        with patch.dict(os.environ, {'BFL_API_KEY': 'test-secret'}), httpx.Client(transport=httpx.MockTransport(handler)) as http:
            with self.assertRaises(BflFailure) as error:
                BflClient(http).generate('flux-3-image', {}, threading.Event())
            self.assertNotIn('secret', str(error.exception))
            self.assertEqual(len(calls), 1)

    def test_cancellation_before_submission(self):
        cancel = threading.Event()
        cancel.set()
        with patch.dict(os.environ, {'BFL_API_KEY': 'test'}):
            with self.assertRaises(ResourceCancelled):
                BflClient().generate('flux-3-image', {}, cancel)

    def test_moderation_failure_and_download_limit(self):
        def handler(request):
            if request.method == 'POST':
                return httpx.Response(200, json={'polling_url': 'https://api.bfl.ai/v1/get_result?id=test'})
            return httpx.Response(200, json={'status': 'Content Moderated'})
        with patch.dict(os.environ, {'BFL_API_KEY': 'test'}), httpx.Client(transport=httpx.MockTransport(handler)) as http:
            with self.assertRaisesRegex(BflFailure, 'Content Moderated'):
                BflClient(http).generate('flux-3-image', {}, threading.Event())
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b'1234'))) as http:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'result'
                with self.assertRaises(BflFailure):
                    BflClient(http).download('https://media.example.test/test', path, threading.Event(), 3)
                self.assertFalse(path.exists())
