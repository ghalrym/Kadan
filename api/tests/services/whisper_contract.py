"""Full lightweight download-to-inference contract for each Whisper checkpoint."""
from dataclasses import replace
import hashlib
import io
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from api.inference.resources import ResourceManager
from api.server import app
from api.services.model_catalog import CATALOG
from api.services.model_downloads import ModelManager
from api.services.transcription import TranscriptionManager
from api.services.whisper_catalog import CHECKPOINTS, checkpoint
from api.tests.services.test_transcription import audio_url


def assert_checkpoint_contract(case, name, digest):
    """Exercise real download publication, selection and native dispatch using fixture bytes."""
    entry = checkpoint(name)
    case.assertEqual(entry.sha256, digest)
    case.assertIn(name, CHECKPOINTS)
    model_id = f'whisper-{name}'
    case.assertEqual(CATALOG[model_id].revision, digest)
    payload = b'controlled model fixture'
    fixture_digest = hashlib.sha256(payload).hexdigest()
    fixture_entry = replace(CATALOG[model_id], revision=fixture_digest)
    def source(request, **kwargs):
        response = io.BytesIO(payload)
        response.headers = {'Content-Length': str(len(payload))}
        return response
    with tempfile.TemporaryDirectory() as temporary:
        store = ModelManager(Path(temporary))
        resources = ResourceManager(64 * 1024**3, {})
        native = Mock()
        native.transcribe.return_value = {'text': 'controlled native output', 'language': 'en'}
        factory = Mock(return_value=native)
        manager = TranscriptionManager(factory, resources, store)
        with patch.dict(CATALOG, {model_id: fixture_entry}), patch(
                'api.routes.v1.models.model_manager', store), patch(
                'api.routes.v1.audio.transcriptions.transcription_manager', manager), patch(
                'api.services.transcription.checkpoint', return_value=replace(entry, sha256=fixture_digest)), patch(
                'api.services.model_downloads.urlopen', side_effect=source):
            client = TestClient(app)
            case.assertIn(name, client.get('/v1/audio/transcriptions/models').json()['models'])
            case.assertEqual(client.post(f'/v1/models/{model_id}/download').status_code, 202)
            store._thread.join(5)
            case.assertFalse(store._thread.is_alive())
            _, directory = store.get_checkpoint(model_id)
            case.assertEqual((directory / f'{name}.pt').read_bytes(), payload)
            case.assertEqual(client.put('/v1/audio/transcriptions/models', json={'model': name}).status_code, 200)
            case.assertEqual(manager.selected(), name)
            response = client.post('/v1/audio/transcriptions', json={'audio': audio_url(), 'formatting': False})
            case.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            case.assertEqual(result['model'], name)
            case.assertEqual(result['raw_text'], 'controlled native output')
            factory.assert_called_once_with(str(directory / f'{name}.pt'), device='cpu')
            case.assertEqual(resources.snapshot()['reservations'], {})
