"""CPU-only boundary tests; no model, driver, GPU, database or API imports."""
import subprocess
import sys
import unittest

WORKER = sys.argv.pop()


class ProtocolTest(unittest.TestCase):
    def run_worker(self, commands, budgets=('100', '24', '24')):
        return subprocess.run([WORKER, *budgets], input=commands, text=isinstance(commands, str),
                              capture_output=True, timeout=5)

    def test_lifecycle(self):
        result = self.run_worker('hello\nreserve llm 60 20 10\nloaded 1\npin 1\n'
                                 'evict 1\nunpin 1\nevict 1\nsnapshot\nreleased 1\nsnapshot\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [
            'ok kadan-worker 1 accounting-only', 'ok 1', 'ok', 'ok', 'error busy',
            'ok', 'ok', 'ok 1 60 20 10', 'ok', 'ok 0 0 0 0'])

    def test_invalid_frames_do_not_mutate_or_desynchronize(self):
        result = self.run_worker('reserve llm -1 0 0\nreserve llm 18446744073709551616 0 0\n'
                                 'reserve unknown 1 0 0\nreserve llm 1 0\n'
                                 'reserve llm 1 0 0 extra\n' + 'x' * 5000 + '\n'
                                 'snapshot\nreserve llm 1 0 0\nreleased 1\nreleased 1\nhello')
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertTrue(all(line.startswith('error ') for line in lines[:6]))
        self.assertEqual(lines[6:], ['ok 0 0 0 0', 'ok 1', 'ok', 'error stale_handle',
                                     'ok kadan-worker 1 accounting-only'])

    def test_exact_frame_limit_and_recovery(self):
        # The newline is framing, not part of the 4096-byte payload limit.
        exact = 'hello' + ' ' * (4096 - len('hello'))
        oversized = 'reserve llm 1 0 0' + ' ' * (4097 - len('reserve llm 1 0 0'))
        for ending in ('\n', ''):
            with self.subTest(ending=ending):
                result = self.run_worker(exact + ending)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, 'ok kadan-worker 1 accounting-only\n')
                result = self.run_worker(oversized + ending)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, 'error frame_too_large\n')
        result = self.run_worker(exact + '\n' + oversized + '\nsnapshot\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [
            'ok kadan-worker 1 accounting-only', 'error frame_too_large', 'ok 0 0 0 0'])

    def test_host_only_and_maximum_devices(self):
        for devices in (0, 64):
            with self.subTest(devices=devices):
                footprint = ' '.join(['1'] * (devices + 1))
                result = self.run_worker(f'reserve llm {footprint}\nsnapshot\nreleased 1\n',
                                         ('1',) * (devices + 1))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), ['ok 1', f'ok 1 {footprint}', 'ok'])
        result = self.run_worker('hello\n', ('1',) * 66)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, '')

    def test_empty_and_binary_frames(self):
        result = self.run_worker(b'\n \t\r\nhello\x00\nreserve llm 1 0 0\x00\n'
                                 b'\xff\xfe\nhello\nsnapshot\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [
            b'error invalid_command', b'error invalid_command', b'error invalid_command',
            b'error invalid_integer', b'error invalid_command',
            b'ok kadan-worker 1 accounting-only', b'ok 0 0 0 0'])

    def test_wrong_state_and_stale_handles(self):
        result = self.run_worker('reserve llm 1 0 0\npin 1\nevict 1\neviction_failed 1\n'
                                 'unpin 1\nloaded 1\nloaded 1\nreleased 1\nevict 1\n'
                                 'loaded 1\npin 1\nevict 1\nreleased 1\n'
                                 'loaded 1\npin 1\nevict 1\nsnapshot\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [
            'ok 1', 'error not_resident', 'error busy', 'error not_evicting',
            'error not_pinned', 'ok', 'error not_loading', 'error busy', 'ok',
            'error not_loading', 'error not_resident', 'error busy', 'ok',
            'error stale_handle', 'error stale_handle', 'error stale_handle', 'ok 0 0 0 0'])

    def test_zero_byte_resident_limit(self):
        result = self.run_worker('reserve llm 0\n' * 1025 +
                                 'snapshot\nreleased 1\nreserve llm 0\nreleased 1\nsnapshot\n', ('0',))
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(lines[:1024], [f'ok {i}' for i in range(1, 1025)])
        self.assertEqual(lines[1024:], ['error resident_limit', 'ok 1024 0', 'ok',
                                       'ok 1025', 'error stale_handle', 'ok 1024 0'])

    def test_explicit_valid_budgets_required(self):
        for budgets in ((), ('-1',), ('1x',), ('18446744073709551616',)):
            with self.subTest(budgets=budgets):
                result = self.run_worker('', budgets)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, '')


if __name__ == '__main__':
    unittest.main()
