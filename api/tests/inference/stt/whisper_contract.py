"""Full lightweight download-to-inference contract for each Whisper checkpoint."""
from dataclasses import replace
import hashlib
import io
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from api.inference.resources import ResourceManager
from api.memory_manager import MemoryManager
from api.services.chat_runtime import ChatRuntime
from api.tests.memory_manager.helpers import direct_feature
from api.server import app
from api.services.model_catalog import CATALOG
from api.services.model_downloads import ModelManager
from api.inference.stt.whisper_transcriber import WhisperTranscriber
from api.inference.stt.native_worker import NativeWhisper
from api.tests.inference.stt.test_native_worker import Child
from api.inference.stt.catalog import get_whisper_checkpoints, checkpoint
from api.tests.inference.stt.test_whisper_transcriber import audio_url


def assert_checkpoint_contract(case, name, digest):
    """Exercise real download publication, selection and native dispatch using fixture bytes."""
    entry = checkpoint(name)
    case.assertEqual(entry.sha256, digest)
    case.assertIn(name, get_whisper_checkpoints())
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
        child = Child()
        resolver = Mock(return_value=['/native', '/model', 'weights', 'dimensions', 'assets'])
        native = NativeWhisper(resources, resolver, lambda: child)
        manager = WhisperTranscriber(worker=native, resources=resources, store=store)
        memory = MemoryManager(runtime=ChatRuntime(resources=resources), transcription=manager)
        with patch('api.routes.v1.audio.transcriptions.memory_manager.submit', direct_feature(memory, 'stt')), patch.dict(CATALOG, {model_id: fixture_entry}), patch(
                'api.routes.v1.models.model_manager', store), patch(
                'api.routes.v1.audio.transcriptions.get_whisper_transcriber', return_value=manager), patch(
                'api.inference.stt.whisper_transcriber.checkpoint', return_value=replace(entry, sha256=fixture_digest)), patch(
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
            case.assertEqual(result['raw_text'], ' Hello Andrew!')
            resolver.assert_called_once_with(name)
            case.assertEqual(child.starts, 1)
            case.assertEqual(len(resources.snapshot()['reservations']), 1)
            manager.close()
            case.assertEqual(resources.snapshot()['reservations'], {})
