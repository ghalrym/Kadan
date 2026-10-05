import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from updater.docker import Docker
from updater.tests.test_releases import manifest


class DockerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = dict(project='existing-home', model_volume='existing_models', postgres_volume='existing_db',
                           hf_volume='persistent_hf', POSTGRES_DB='kadan', POSTGRES_USER='kadan', POSTGRES_PASSWORD='fixture')
        self.docker = Docker(self.root, self.config)

    def test_fixed_project_external_volumes_and_no_socket_or_privileged_configuration(self):
        self.docker.render(manifest())
        value = json.loads((self.root / 'compose.json').read_text())
        self.assertEqual(value['volumes']['model_data'], {'external': True, 'name': 'existing_models'})
        self.assertEqual(value['volumes']['postgres_data']['name'], 'existing_db')
        self.assertEqual(value['volumes']['hf_data']['name'], 'persistent_hf')
        api = value['services']['api']
        self.assertIn('hf_data:/var/lib/kadan/huggingface', api['volumes'])
        self.assertIn(str(self.root / 'control') + ':/run/kadan-updater:ro', api['volumes'])
        self.assertNotIn('docker.sock', json.dumps(value))
        self.assertEqual(api['stop_grace_period'], '120s')
        self.assertIn('existing-home', self.docker.compose)

    def test_stop_timeout_never_uses_sigkill_or_compose_stop(self):
        calls = []
        def run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout=b'a' * 64 + b'\n')
        with patch.object(self.docker, 'run', side_effect=run), \
             patch.object(self.docker, 'wait', side_effect=TimeoutError):
            with self.assertRaises(TimeoutError): self.docker.stop_pair()
        self.assertTrue(any('--signal=SIGTERM' in args for args in calls))
        self.assertFalse(any('SIGKILL' in ' '.join(args) or 'stop' in args or 'down' in args for args in calls))

    def test_routine_replacement_only_targets_pair_and_never_migrates_or_restarts_postgres(self):
        with patch.object(self.docker, 'run') as run:
            self.docker.start_pair(manifest())
        args = run.call_args.args[0]
        self.assertEqual(args[-2:], ['api', 'frontend'])
        self.assertIn('--no-deps', args)
        self.assertNotIn('postgres', args)
        self.assertNotIn('alembic', args)

    def test_revision_mismatch_rejected_even_when_digest_is_valid(self):
        def run(args, **kwargs):
            value = b'[ {"Config": {"Labels": {"org.opencontainers.image.revision": "wrong"}}} ]'
            return subprocess.CompletedProcess(args, 0, stdout=value)
        with patch.object(self.docker, 'run', side_effect=run):
            with self.assertRaisesRegex(RuntimeError, 'revision'): self.docker.pull(manifest())

    def test_existing_password_is_not_compose_interpolation(self):
        self.config['POSTGRES_PASSWORD'] = 'existing$PASSWORD${SECRET}'
        self.docker.render(manifest())
        value = json.loads((self.root / 'compose.json').read_text())
        self.assertEqual(value['services']['api']['environment']['POSTGRES_PASSWORD'], 'existing$$PASSWORD$${SECRET}')

    def test_interrupted_migration_reconciliation_only_reads_revision(self):
        calls = []
        def run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout=b'true' if 'python' in args else b'')
        with patch.object(self.docker, 'check_volumes'), patch.object(self.docker, 'bootstrap_database'), \
             patch.object(self.docker, 'run', side_effect=run):
            self.assertTrue(self.docker.bootstrap_migration_complete(manifest()))
        inspection = next(args for args in calls if 'python' in args)
        self.assertIn('SELECT version_num FROM alembic_version', inspection[-1])
        self.assertFalse(any('upgrade' in args for args in calls))

    def test_migration_reconciliation_refuses_an_existing_running_setup_container(self):
        with patch.object(self.docker, 'check_volumes'), patch.object(self.docker, 'run',
                return_value=subprocess.CompletedProcess([], 0, stdout=b'running-id')) as run:
            with self.assertRaisesRegex(RuntimeError, 'still running'):
                self.docker.bootstrap_migration_complete(manifest())
        self.assertEqual(run.call_count, 1)
