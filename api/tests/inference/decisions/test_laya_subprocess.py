"""Laya command preparation only; no worker or model execution."""
import asyncio
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from api.inference.decisions.laya_subprocess import resolve_laya_command, LayaSubprocessEvaluator
from api.services.model_catalog import CATALOG, allowed_asset, validate_assets
from api.routes.v1.models import ModelStatus


class LayaPreparationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.binary = self.root / 'worker'
        self.binary.touch()
        self.binary.chmod(0o700)
        self.checkpoint = self.root / 'laya-checkpoint'
        self.checkpoint.mkdir()
        for name in CATALOG['laya'].required_files:
            path = self.checkpoint / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture, never executed')
        self.enterContext(patch.dict(os.environ, {'KADAN_NATIVE_DECISION_WORKER': str(self.binary)}, clear=True))
        self.manager = self.enterContext(patch('api.inference.decisions.laya_subprocess.model_manager'))
        self.manager.ensure_checkpoint.return_value = (CATALOG['laya'], self.checkpoint)

    def test_default_uses_shared_published_directory_and_request_cancellation(self):
        cancel = threading.Event()
        self.assertEqual(resolve_laya_command(cancel), [str(self.binary), str(self.checkpoint)])
        self.manager.ensure_checkpoint.assert_called_once_with(CATALOG['laya'], cancel)

    def test_custom_pinned_repository_preserved_with_isolated_identity(self):
        with patch.dict(os.environ, {'KADAN_LAYA_MODEL': 'owner/custom', 'KADAN_LAYA_REVISION': '1'*40}):
            resolve_laya_command(threading.Event())
        entry = self.manager.ensure_checkpoint.call_args.args[0]
        self.assertEqual((entry.repo_id, entry.revision), ('owner/custom', '1'*40))
        self.assertNotEqual(entry.id, 'laya')
        self.assertEqual(entry.required_files, CATALOG['laya'].required_files)

    def test_local_checkpoint_does_not_download(self):
        with patch.dict(os.environ, {'KADAN_LAYA_MODEL': str(self.checkpoint)}):
            resolve_laya_command(threading.Event())
        self.manager.ensure_checkpoint.assert_not_called()

    def test_cancelled_preflight_never_starts_download(self):
        cancel = threading.Event(); cancel.set()
        with self.assertRaises(InterruptedError): resolve_laya_command(cancel)
        self.manager.ensure_checkpoint.assert_not_called()

    def test_catalog_contract_and_api_kind_agree(self):
        entry = CATALOG['laya']
        validate_assets(entry, set(entry.required_files))
        self.assertTrue(all(allowed_asset(name, entry) for name in entry.required_files))
        self.assertFalse(allowed_asset('encoder/model.safetensors', entry))
        self.assertIn('decision', ModelStatus.model_json_schema()['properties']['kind']['enum'])

    def test_preflight_joins_cancellable_preparation(self):
        evaluator = LayaSubprocessEvaluator(resolve=Mock(return_value=['worker', 'checkpoint']))
        asyncio.run(evaluator.preflight())
        evaluator.resolve.assert_called_once()
