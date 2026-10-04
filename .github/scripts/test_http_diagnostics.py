"""Diagnostic regression tests use no network, tokens, or environment secrets."""
from email.message import Message
import io
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

import publish_pr as publisher
import probe_publisher_preflight as probe


class HTTPDiagnosticTests(unittest.TestCase):
    def test_reports_fixed_endpoint_and_rate_reset(self):
        headers = Message()
        for name, value in [('X-RateLimit-Limit', '60'), ('X-RateLimit-Remaining', '0'),
                            ('X-RateLimit-Used', '60'), ('X-RateLimit-Reset', '1791146400'),
                            ('Retry-After', '120'), ('Authorization', 'SECRET'),
                            ('Set-Cookie', 'SECRET')]:
            headers[name] = value
        text = publisher.http_diagnostic(publisher.ROOT, 403, headers)
        self.assertEqual(text, 'HTTP 403; path=/repos/ghalrym/Kadan; x-ratelimit-limit=60; '
            'x-ratelimit-remaining=0; x-ratelimit-used=60; x-ratelimit-reset=1791146400; retry-after=120')

    def test_unknown_paths_and_query_values_are_redacted(self):
        self.assertEqual(publisher.http_diagnostic('/SECRET?token=SECRET', 403, None),
                         'HTTP 403; path=<redacted>')
        self.assertEqual(publisher.http_diagnostic(publisher.ROOT + '?token=SECRET', 403, None),
                         'HTTP 403; path=/repos/ghalrym/Kadan')

    def test_untrusted_header_values_are_not_logged(self):
        for value in ('SECRET', '60\nSECRET', '1' * 13, '-1', '1.0', 'Wed, 01 Jan 2030 00:00:00 GMT'):
            with self.subTest(value=value):
                self.assertEqual(publisher.http_diagnostic(publisher.ROOT, 403,
                    {'x-ratelimit-reset': value, 'retry-after': value}),
                    'HTTP 403; path=/repos/ghalrym/Kadan')

    def test_api_error_excludes_credentials_and_response_body(self):
        api = publisher.API('SECRET_TOKEN')
        error = urllib.error.HTTPError('https://api.github.com' + publisher.ROOT,
            403, 'SECRET_REASON', {'x-ratelimit-remaining': '0'}, io.BytesIO(b'SECRET_BODY'))
        with patch.object(api.opener, 'open', side_effect=error) as opened:
            with self.assertRaises(publisher.Failure) as raised:
                api.request('GET', publisher.ROOT)
        self.assertEqual(str(raised.exception),
            'GitHub API request failed (HTTP 403; path=/repos/ghalrym/Kadan; x-ratelimit-remaining=0)')
        self.assertEqual(opened.call_count, 1)

    def test_probe_is_three_fixed_unauthenticated_gets_without_retry(self):
        response = MagicMock(status=200, headers={'x-ratelimit-remaining': '59'})
        response.__enter__.return_value = response
        error = urllib.error.HTTPError('https://api.github.com', 403, 'SECRET',
            {'x-ratelimit-remaining': '0'}, io.BytesIO(b'SECRET_BODY'))
        opener = MagicMock()
        opener.open.side_effect = [response, error, response]
        with patch.object(probe.urllib.request, 'build_opener', return_value=opener), \
                patch('sys.stdout', new_callable=io.StringIO) as output:
            probe.main()
        self.assertEqual(opener.open.call_count, 3)
        paths = []
        for call in opener.open.call_args_list:
            request = call.args[0]
            self.assertEqual(request.get_method(), 'GET')
            self.assertIsNone(request.data)
            self.assertFalse(any(k.lower() == 'authorization' for k in request.headers))
            paths.append(request.full_url)
        self.assertEqual(paths, [
            'https://api.github.com/repos/ghalrym/Kadan',
            'https://api.github.com/repos/ghalrym/Kadan/environments/kadan-pr-publishing',
            'https://api.github.com/repos/ghalrym/Kadan/environments/kadan-pr-publishing/deployment-branch-policies?per_page=100&page=1'])
        self.assertIn('HTTP 403; path=/repos/ghalrym/Kadan/environments/kadan-pr-publishing', output.getvalue())
        self.assertNotIn('SECRET', output.getvalue())
        response.read.assert_not_called()

    def test_probe_transport_error_is_sanitized(self):
        opener = MagicMock()
        opener.open.side_effect = urllib.error.URLError('SECRET')
        with patch.object(probe.urllib.request, 'build_opener', return_value=opener), \
                patch('sys.stdout', new_callable=io.StringIO) as output:
            probe.main()
        self.assertEqual(output.getvalue().count('Transport failure; no HTTP response available'), 3)
        self.assertNotIn('SECRET', output.getvalue())


if __name__ == '__main__':
    unittest.main()
