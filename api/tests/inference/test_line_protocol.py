"""Terminal worker failures keep local context out of public API error text."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from api.inference.line_protocol import LineProtocolError, LineProtocolProcess


class ErrorTests(unittest.TestCase):
    def child(self):
        child = LineProtocolProcess()
        child.process = Mock(pid=123, returncode=0)
        child.process.poll.return_value = 0
        child.selector = Mock()
        child.selector.get_map.return_value = {}
        self.enterContext(patch('api.inference.line_protocol.os.read', side_effect=BlockingIOError))
        child._capture_diagnostics(b'text_layer 30\nprivate-path-and-token\n')
        return child

    def failure(self, frame):
        child = self.child()
        child.buffer.extend(frame)
        with self.assertRaises(LineProtocolError) as raised:
            child.read(5)
        child.stop()
        self.assertTrue(child.closed)
        child.process.wait.assert_called_once_with(timeout=2)
        self.assertIn(b'text_layer 30', raised.exception.diagnostics)
        self.assertIn(b'private-path-and-token', raised.exception.diagnostics)
        self.assertNotIn('private-path-and-token', str(raised.exception))
        return raised.exception

    def test_bare_error_is_failure_not_invalid_ready(self):
        self.assertEqual(str(self.failure(b'error\n')), 'Inference subprocess failed')

    def test_image_stage_is_preserved(self):
        for stage in ('startup', 'load', 'generate', 'park', 'resume', 'cleanup'):
            with self.subTest(stage=stage):
                self.assertIn(f'image_{stage}_failed', str(self.failure(f'error image_{stage}_failed\n'.encode())))

    def test_untrusted_error_payload_is_not_public(self):
        self.assertEqual(str(self.failure(b'error private-path-and-token\n')), 'Inference subprocess failed')

    def test_exited_worker_without_reply_retains_private_stderr(self):
        child = self.child()
        with self.assertRaisesRegex(LineProtocolError, 'complete reply') as raised:
            child.read(1)
        self.assertIn(b'private-path-and-token', raised.exception.diagnostics)
        self.assertNotIn('private-path-and-token', str(raised.exception))

    def test_deadline_uses_clock_without_sleeping(self):
        child = self.child()
        with patch('api.inference.line_protocol.time.monotonic', return_value=2):
            with self.assertRaisesRegex(TimeoutError, 'deadline') as raised:
                child._pump(1, None)
        self.assertIn(b'private-path-and-token', raised.exception.diagnostics)

    def test_non_ascii_reply_is_rejected(self):
        self.assertIn('Non-ASCII', str(self.failure(b'\xff\n')))

    def test_oversized_reply_is_rejected(self):
        child = self.child()
        child.selector.select.return_value = [(SimpleNamespace(fileobj=child.process.stdout, data='out'), 1)]
        with patch('api.inference.line_protocol.os.read', return_value=b'x' * 5000), patch('api.inference.line_protocol.time.monotonic', return_value=0):
            with self.assertRaisesRegex(LineProtocolError, 'frame too large'):
                child._pump(1, None)

    def test_unsuccessful_exit_retains_private_stderr(self):
        child = self.child()
        child.process.returncode = 1
        with self.assertRaisesRegex(LineProtocolError, 'exit was not successful') as raised:
            child.finish(1)
        self.assertIn(b'private-path-and-token', raised.exception.diagnostics)

    def test_stage_receipts_have_pid_and_timestamp_without_private_text(self):
        child = LineProtocolProcess()
        child.process = SimpleNamespace(pid=123)
        with self.assertLogs('api.inference.line_protocol', level='INFO') as captured:
            child._capture_diagnostics(b'text_la')
            child._capture_diagnostics(b'yer 30\nprivate-secret\nblock 2\n')
        self.assertEqual(len(captured.output), 2)
        self.assertRegex(captured.output[0], r'pid=123 observed_unix_ns=[0-9]+ stage=text_layer index=30')
        self.assertNotIn('private-secret', '\n'.join(captured.output))
        child._capture_diagnostics(b'x' * 10000)
        self.assertLessEqual(len(child.diagnostics), 8192)
        self.assertLessEqual(len(child.stage_buffer), 8192)

    def test_exact_weight_bytes_are_allowlisted_and_not_truncated(self):
        child = LineProtocolProcess()
        child.process = SimpleNamespace(pid=600)
        records = [
            ('checkpoint_tensor_bytes', 33115078673),
            ('expanded_f32_inventory_bytes', 64879591457),
            ('nonfloating_inventory_bytes', 0),
            ('gpu_1_weight_planned_bytes', 9000000000),
            ('gpu_1_weight_allocated_bytes', 1024),
            ('gpu_0_weight_planned_bytes', 5000000000),
            ('gpu_0_weight_allocated_bytes', 0),
            ('text_checkpoint_read_bytes', 15136194560),
            ('vae_checkpoint_read_bytes', 12345),
        ]
        with patch('api.inference.line_protocol.report') as report, self.assertLogs('api.inference.line_protocol', level='INFO') as logs:
            for stage, value in records:
                line = f'{stage} {value}\n'.encode()
                child._capture_diagnostics(line[:7])
                child._capture_diagnostics(line[7:])
            child._capture_diagnostics(b'weight_retention_bounded_mib 61874\ngpu_2_weight_planned_bytes 123\ngpu_0_weight_allocated_bytes -1\ngpu_0_weight_allocated_bytes 1234567890123\nprivate-secret 123\n')
        self.assertEqual([call.args for call in report.call_args_list], records)
        self.assertEqual(len(logs.output), len(records))
        self.assertIn('stage=expanded_f32_inventory_bytes index=64879591457', logs.output[1])
