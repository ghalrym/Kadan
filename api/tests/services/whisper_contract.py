"""Shared lightweight contract for independently registered Whisper checkpoints."""
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from api.inference.resources import ResourceManager
from api.server import app
from api.services.transcription import TranscriptionManager
from api.services.whisper_catalog import CHECKPOINTS, checkpoint
from api.tests.services.test_transcription import audio_url


def assert_checkpoint_contract(case, name, digest):
    """Exercise catalog exposure, persisted selection and native dispatch without weights."""
    entry = checkpoint(name)
    case.assertEqual(entry.sha256, digest)
    case.assertIn(name, CHECKPOINTS)
    response = TestClient(app).get('/v1/audio/transcriptions/models')
    case.assertIn(name, response.json()['models'])
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / f'{name}.pt').write_bytes(b'fixture')
        store = Mock(root=root)
        store.get_checkpoint.return_value = (None, root)
        resources = ResourceManager(64 * 1024**3, {})
        native = Mock()
        native.transcribe.return_value = {'text': 'controlled native output', 'language': 'en'}
        factory = Mock(return_value=native)
        manager = TranscriptionManager(factory, resources, store)
        manager.select(name)
        case.assertEqual(manager.selected(), name)
        fixture = replace(entry, sha256=hashlib.sha256(b'fixture').hexdigest())
        with patch('api.services.transcription.checkpoint', return_value=fixture):
            result = manager.transcribe(audio_url())
        case.assertEqual(result['model'], name)
        case.assertEqual(result['raw_text'], 'controlled native output')
        store.get_checkpoint.assert_called_once_with(f'whisper-{name}')
        factory.assert_called_once_with(str(root / f'{name}.pt'), device='cpu')
        case.assertEqual(resources.snapshot()['reservations'], {})
