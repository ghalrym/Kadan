import os
from urllib.error import HTTPError
from urllib.request import Request
import unittest
from unittest.mock import Mock, patch

from api.services.huggingface_access import OriginAuthorizationRedirect, open_gated_checkpoint


class HubAccessTests(unittest.TestCase):
    def test_absent_token_and_untrusted_origin_fail_without_request(self):
        with patch.dict(os.environ, {'HF_TOKEN': ''}), patch('api.services.huggingface_access.build_opener') as opener:
            with self.assertRaisesRegex(ValueError, 'existing HF_TOKEN'):
                open_gated_checkpoint('https://huggingface.co/repository/resolve/revision/file')
            with self.assertRaises(ValueError): open_gated_checkpoint('https://example.com/file')
            opener.assert_not_called()

    def test_configured_token_used_only_on_initial_hub_request(self):
        with patch.dict(os.environ, {'HF_TOKEN': 'fixture-not-a-real-token'}), patch('api.services.huggingface_access.build_opener') as opener:
            open_gated_checkpoint('https://huggingface.co/repository/resolve/revision/file')
            request = opener.return_value.open.call_args.args[0]
            self.assertEqual(request.get_header('Authorization'), 'Bearer fixture-not-a-real-token')
            redirect = OriginAuthorizationRedirect().redirect_request(request, None, 302, 'Found', {}, 'https://cdn.example/file')
            self.assertIsNone(redirect.get_header('Authorization'))
            second = OriginAuthorizationRedirect().redirect_request(redirect, None, 302, 'Found', {}, 'https://huggingface.co/file')
            self.assertIsNone(second.get_header('Authorization'))
            with self.assertRaises(ValueError):
                OriginAuthorizationRedirect().redirect_request(request, None, 302, 'Found', {}, 'http://cdn.example/file')

    def test_access_denial_does_not_expose_token(self):
        with patch.dict(os.environ, {'HF_TOKEN': 'fixture-secret'}), patch('api.services.huggingface_access.build_opener') as opener:
            opener.return_value.open.side_effect = HTTPError('https://huggingface.co/file', 403, 'Forbidden', {}, None)
            with self.assertRaisesRegex(ValueError, 'denied checkpoint access') as failure:
                open_gated_checkpoint('https://huggingface.co/file')
            self.assertNotIn('fixture-secret', str(failure.exception))
