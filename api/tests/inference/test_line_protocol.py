"""Terminal worker failures keep local context out of public API error text."""
import sys
import unittest
from api.inference.line_protocol import LineProtocolError, LineProtocolProcess


class ErrorTests(unittest.TestCase):
    def failure(self, frame):
        child = LineProtocolProcess()
        code = ('import os; os.write(2, b"text_layer 30\\nprivate-path-and-token\\n"); '
                f'os.write(1, {frame!r})')
        try:
            child.start([sys.executable, '-c', code])
            with self.assertRaises(LineProtocolError) as raised:
                child.read(5)
            error = raised.exception
        finally:
            child.stop()
        self.assertTrue(child.closed)
        self.assertIn(b'text_layer 30', error.diagnostics)
        self.assertIn(b'private-path-and-token', error.diagnostics)
        self.assertNotIn('private-path-and-token', str(error))
        return error

    def test_bare_error_is_failure_not_invalid_ready(self):
        self.assertEqual(str(self.failure(b'error\n')), 'Inference subprocess failed')

    def test_image_stage_is_preserved(self):
        for stage in ('startup', 'load', 'generate', 'park', 'resume', 'cleanup'):
            with self.subTest(stage=stage):
                self.assertIn(f'image_{stage}_failed', str(self.failure(f'error image_{stage}_failed\n'.encode())))

    def test_untrusted_error_payload_is_not_public(self):
        self.assertEqual(str(self.failure(b'error private-path-and-token\n')), 'Inference subprocess failed')
