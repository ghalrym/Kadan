"""Existing tokens stay on the initial HTTPS Hub request, including gated downloads."""
import unittest
from unittest.mock import patch
from urllib.request import HTTPRedirectHandler
from api.services.model_downloads import hub_open


class HubAuthTests(unittest.TestCase):
    def test_token_not_forwarded_to_redirect(self):
        with patch.dict('os.environ', {'HF_TOKEN': 'fixture-token'}), patch('api.services.model_downloads.urlopen') as opened:
            hub_open('https://huggingface.co/example/resolve/revision/model.safetensors')
        request = opened.call_args.args[0]
        self.assertEqual(request.get_header('Authorization'), 'Bearer fixture-token')
        redirected = HTTPRedirectHandler().redirect_request(request, None, 302, '', {}, 'https://cdn.example/weights')
        self.assertIsNone(redirected.get_header('Authorization'))

    def test_other_hosts_never_receive_token(self):
        with patch.dict('os.environ', {'HF_TOKEN': 'fixture-token'}), patch('api.services.model_downloads.urlopen') as opened:
            hub_open('https://example.com/model')
        self.assertEqual(opened.call_args.args[0], 'https://example.com/model')
