import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from updater.__main__ import bootstrap_install
from updater.tests.test_releases import manifest


class SetupTests(unittest.TestCase):
    def test_failed_setup_retries_only_same_release_and_preserves_host_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = manifest()
            releases = Mock(verified=Mock(return_value=release))
            docker = Mock(bootstrap=Mock(side_effect=OSError('registry unavailable')))
            with patch('updater.__main__.secrets.token_urlsafe', return_value='test-only-host-key') as secret:
                with self.assertRaises(ValueError): bootstrap_install(root, release['commit'], False, docker, releases)
                self.assertFalse((root / 'state.json').exists())
                with self.assertRaises(OSError): bootstrap_install(root, release['commit'], True, docker, releases)
                self.assertEqual(json.loads((root / 'state.json').read_text())['phase'], 'bootstrap_failed')
                with self.assertRaises(ValueError): bootstrap_install(root, 'b' * 40, True, docker, releases)
                docker.bootstrap.side_effect = None
                bootstrap_install(root, release['commit'], True, docker, releases)
                self.assertEqual(json.loads((root / 'state.json').read_text())['phase'], 'idle')
                self.assertEqual((root / 'control/host-key').read_text(), 'test-only-host-key')
                secret.assert_called_once()
                with self.assertRaises(ValueError): bootstrap_install(root, release['commit'], True, docker, releases)
