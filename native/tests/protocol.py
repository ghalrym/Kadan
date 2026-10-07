"""CPU-only boundary tests; no model, driver, GPU, database or API imports."""
import subprocess
import sys
import unittest

WORKER = sys.argv.pop()


class ProtocolTest(unittest.TestCase):
    def run_worker(self, commands, budgets=('100', '24', '24')):
        return subprocess.run([WORKER, *budgets], input=commands, text=True,
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

    def test_explicit_valid_budgets_required(self):
        for budgets in ((), ('-1',), ('1x',), ('18446744073709551616',)):
            with self.subTest(budgets=budgets):
                result = self.run_worker('', budgets)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, '')


if __name__ == '__main__':
    unittest.main()
