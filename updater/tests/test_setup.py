import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from updater.__main__ import bootstrap_install
from updater.tests.test_releases import manifest


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.release = manifest()
        self.releases = Mock(verified=Mock(return_value=self.release))
        self.docker = Mock(bootstrap_prepare=Mock(return_value='preserved-backup.dump'),
                           bootstrap_migration_complete=Mock(return_value=True))
        self.secret = patch('updater.__main__.secrets.token_urlsafe', return_value='test-only-host-key')
        self.key_generator = self.secret.start()
        self.addCleanup(self.secret.stop)

    def install(self, commit=None, allow=True):
        bootstrap_install(self.root, commit or self.release['commit'], allow, self.docker, self.releases)

    def state(self):
        return json.loads((self.root / 'state.json').read_text())

    def test_failed_prepare_retries_only_same_verified_release_and_preserves_evidence(self):
        with self.assertRaises(ValueError): self.install(allow=False)
        self.assertFalse((self.root / 'state.json').exists())
        self.docker.bootstrap_prepare.side_effect = OSError('registry unavailable')
        with self.assertRaises(OSError): self.install()
        self.assertEqual(self.state()['bootstrap']['stage'], 'preparing')
        with self.assertRaises(ValueError): self.install('b' * 40)
        self.docker.bootstrap_prepare.side_effect = None
        self.install()
        self.assertEqual(self.state()['phase'], 'idle')
        self.assertEqual(self.state()['bootstrap_history']['failures'], [{'stage': 'preparing', 'type': 'OSError'}])
        self.assertEqual(self.state()['bootstrap_history']['backup'], 'preserved-backup.dump')
        self.assertEqual((self.root / 'control/host-key').read_text(), 'test-only-host-key')
        self.key_generator.assert_called_once()
        with self.assertRaises(ValueError): self.install()

    def test_keyboard_interrupt_before_render_can_resume(self):
        def interrupt(release):
            self.assertEqual(self.state()['bootstrap']['stage'], 'preparing')
            self.assertFalse((self.root / 'compose.json').exists())
            raise KeyboardInterrupt()
        self.docker.bootstrap_prepare.side_effect = interrupt
        with self.assertRaises(KeyboardInterrupt): self.install()
        self.assertEqual(self.state()['bootstrap']['failures'][0]['type'], 'KeyboardInterrupt')
        self.docker.bootstrap_prepare.side_effect = None
        self.install()
        self.assertEqual(self.state()['phase'], 'idle')
        self.docker.bootstrap_migrate.assert_called_once()

    def test_process_exit_before_render_leaves_durable_resumable_checkpoint(self):
        # os._exit bypasses all Python handlers, matching the relevant power-loss
        # boundary without touching Docker or generating real host credentials.
        script = '''
import os, sys
from pathlib import Path
from unittest.mock import Mock, patch
from updater.__main__ import bootstrap_install
from updater.tests.test_releases import manifest
root=Path(sys.argv[1])
def exit_before_render(release): os._exit(73)
with patch('updater.__main__.secrets.token_urlsafe', return_value='test-only-host-key'):
 bootstrap_install(root, 'a'*40, True, Mock(bootstrap_prepare=exit_before_render), Mock(verified=Mock(return_value=manifest())))
'''
        result = subprocess.run([sys.executable, '-c', script, str(self.root)], timeout=10)
        self.assertEqual(result.returncode, 73)
        self.assertEqual(self.state()['bootstrap']['stage'], 'preparing')
        self.assertFalse((self.root / 'compose.json').exists())
        self.install()
        self.assertEqual(self.state()['phase'], 'idle')
        self.docker.bootstrap_prepare.assert_called_once()
        self.docker.bootstrap_migrate.assert_called_once()
        self.key_generator.assert_not_called()

    def test_uncertain_migration_never_repeats_without_verified_completion(self):
        self.docker.bootstrap_migrate.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt): self.install()
        self.assertEqual(self.state()['bootstrap']['stage'], 'migrating')
        self.docker.bootstrap_migration_complete.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'uncertain'): self.install()
        self.docker.bootstrap_migrate.assert_called_once()
        self.docker.start_pair.assert_not_called()
        self.assertEqual(self.state()['bootstrap']['backup'], 'preserved-backup.dump')
        # Once the owner has completed/verified the exact migration, resume reads
        # its revision and advances without calling upgrade a second time.
        self.docker.bootstrap_migration_complete.return_value = True
        self.install()
        self.docker.bootstrap_migrate.assert_called_once()
        self.docker.start_pair.assert_called_once_with(self.release)
        self.assertEqual(self.state()['phase'], 'idle')

    def test_exit_after_migration_before_checkpoint_uses_read_only_reconciliation(self):
        script = '''
import os, sys
from pathlib import Path
from unittest.mock import Mock, patch
from updater.__main__ import bootstrap_install
from updater.tests.test_releases import manifest
with patch('updater.__main__.secrets.token_urlsafe', return_value='test-only-host-key'):
 bootstrap_install(Path(sys.argv[1]), 'a'*40, True,
  Mock(bootstrap_prepare=Mock(return_value='retained.dump'), bootstrap_migrate=lambda: os._exit(74)),
  Mock(verified=Mock(return_value=manifest())))
'''
        result = subprocess.run([sys.executable, '-c', script, str(self.root)], timeout=10)
        self.assertEqual(result.returncode, 74)
        self.assertEqual(self.state()['bootstrap']['stage'], 'migrating')
        self.install()
        self.docker.bootstrap_migration_complete.assert_called_once_with(self.release)
        self.docker.bootstrap_prepare.assert_not_called()
        self.docker.bootstrap_migrate.assert_not_called()

    def test_start_and_activation_resume_never_rerun_migrations(self):
        self.docker.start_pair.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt): self.install()
        self.assertEqual(self.state()['bootstrap']['stage'], 'starting')
        self.docker.start_pair.side_effect = None
        self.docker.internal.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt): self.install()
        self.assertEqual(self.state()['bootstrap']['stage'], 'activating')
        self.docker.internal.side_effect = None
        self.install()
        self.docker.bootstrap_prepare.assert_called_once()
        self.docker.bootstrap_migrate.assert_called_once()
        self.assertEqual(self.docker.start_pair.call_count, 2)
        self.assertEqual(self.state()['phase'], 'idle')

    def test_changed_manifest_for_same_commit_is_not_a_resume(self):
        self.docker.bootstrap_prepare.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt): self.install()
        self.releases.verified.return_value = {**self.release, 'api': 'ghcr.io/ghalrym/kadan-api@sha256:' + '9' * 64}
        with self.assertRaisesRegex(ValueError, 'release changed'): self.install()
        self.assertEqual(self.state()['current'], self.release)
