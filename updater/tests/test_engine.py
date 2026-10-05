import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from updater.engine import Engine
from updater.tests.test_releases import manifest


class FakeDocker:
    def __init__(self):
        self.calls = []
        self.running = manifest()
        self.fail = None
        self.engine = None

    def pull(self, release):
        self.calls.append('pull')
        if self.fail == 'pull': raise OSError('registry unavailable')
        if self.fail == 'cancel': self.engine.request_cancel()

    def check_volumes(self):
        pass

    def api_stopped(self):
        return False

    def internal(self, operation, method='GET'):
        self.calls.append(operation + ':' + method)
        if operation == 'drain':
            return {'state': 'busy' if self.fail == 'drain' else 'drained'}
        return {}

    def wait(self, predicate, timeout=120):
        if not predicate(): raise TimeoutError('not ready')

    def backup(self):
        self.calls.append('backup')
        if self.fail == 'backup': raise OSError('disk full')

    def stop_pair(self):
        self.calls.append('stop')
        if self.fail == 'stop': raise TimeoutError('container still running')

    def start_pair(self, release):
        self.calls.append('start:' + release['commit'][0])
        self.running = release

    def ready(self, release, models=False):
        return not (self.fail in ('readiness', 'model') and release['commit'][0] == 'b'
                    and (self.fail == 'readiness' or models))


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'control').mkdir()
        self.docker = FakeDocker()
        self.releases = Mock(verified=Mock(return_value=manifest('b' * 40)), upgrade=Mock(return_value=True),
                             latest=Mock(return_value=manifest('b' * 40)))
        self.state = {'current': manifest(), 'phase': 'idle', 'message': 'Ready'}
        self.engine = Engine(self.root, self.docker, self.releases, self.state)
        self.docker.engine = self.engine

    def execute(self):
        self.engine.operation.acquire()
        self.engine.save('checking', 'Checking')
        self.engine._run('b' * 40)

    def test_pulls_before_drain_backups_before_restart_and_pairs_versions(self):
        self.execute()
        self.assertEqual(self.state['phase'], 'complete')
        self.assertLess(self.docker.calls.index('pull'), self.docker.calls.index('drain:POST'))
        self.assertLess(self.docker.calls.index('backup'), self.docker.calls.index('stop'))
        self.assertEqual(self.state['previous']['commit'], 'a' * 40)
        self.assertEqual(self.state['current']['commit'], 'b' * 40)
        self.assertFalse((self.root / 'control/maintenance').exists())

    def test_checks_never_install(self):
        self.engine.check()
        self.assertEqual(self.engine.status()['available'], 'b' * 40)
        self.assertEqual(self.docker.calls, [])

    def test_pull_cancel_drain_and_backup_failure_do_not_restart(self):
        for failure in ('pull', 'cancel', 'drain', 'backup'):
            with self.subTest(failure=failure):
                self.docker.fail, self.docker.calls = failure, []
                self.engine.cancel.clear()
                self.execute()
                self.assertNotIn('stop', self.docker.calls)
                self.assertEqual(self.state['current']['commit'], 'a' * 40)
                self.assertEqual(self.state['phase'], 'failed')
                if failure == 'drain':
                    self.assertNotIn('prepare:POST', self.docker.calls)

    def test_bad_application_or_model_readiness_restores_previous_pair(self):
        for failure in ('readiness', 'model'):
            with self.subTest(failure=failure):
                self.docker.fail, self.docker.calls = failure, []
                self.execute()
                self.assertEqual(self.state['phase'], 'rolled_back')
                self.assertEqual(self.docker.calls.count('stop'), 2)
                self.assertIn('start:a', self.docker.calls)
                self.assertEqual(self.state['current']['commit'], 'a' * 40)

    def test_stop_timeout_never_recreates_or_forces_container(self):
        self.docker.fail = 'stop'
        self.execute()
        self.assertEqual(self.state['phase'], 'recovery_required')
        self.assertFalse(any(item.startswith('start:') for item in self.docker.calls))
        self.assertTrue((self.root / 'control/maintenance').exists())

    def test_migration_boundary_rejected_before_pull_and_restart(self):
        self.releases.verified.return_value = {**manifest('b' * 40), 'data_epoch': 2}
        self.execute()
        self.assertEqual(self.docker.calls, [])
        self.assertEqual(self.state['phase'], 'failed')

    def test_interrupted_process_requires_explicit_local_recovery(self):
        self.state['phase'] = 'restarting'
        recovered = Engine(self.root, self.docker, self.releases, self.state)
        self.assertEqual(recovered.status()['phase'], 'recovery_required')
        self.assertEqual(self.docker.calls, [])
        self.assertEqual(json.loads((self.root / 'state.json').read_text())['current']['commit'], 'a' * 40)

    def test_arbitrary_or_stale_browser_commit_is_rejected(self):
        self.engine.available = manifest('b' * 40)
        for commit in ('master', 'c' * 40, '$(touch /tmp/pwn)'):
            with self.assertRaises(ValueError): self.engine.start(commit)
        self.assertFalse(self.engine.operation.locked())

    def test_lost_resume_response_requires_another_drain_before_rollback(self):
        original = self.docker.internal
        def internal(operation, method='GET'):
            if operation == 'resume' and self.docker.running['commit'][0] == 'b':
                raise OSError('response lost after admission reopened')
            return original(operation, method)
        self.docker.internal = internal
        self.execute()
        self.assertEqual(self.state['phase'], 'rolled_back')
        self.assertEqual(self.docker.calls.count('drain:POST'), 2)

    def test_unreachable_running_api_is_not_evidence_of_quiescence(self):
        self.docker.internal = Mock(side_effect=OSError('unreachable'))
        with self.assertRaisesRegex(RuntimeError, 'still running'):
            self.engine.safe_stop()
        self.assertNotIn('stop', self.docker.calls)
