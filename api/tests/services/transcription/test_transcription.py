import base64
from dataclasses import replace
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

from api.inference.resources import ResourceManager
from api.services.runtime import RuntimeFailure
from api.services.transcription.transcription import TranscriptionManager, get_transcription_manager, _cached_transcription_manager
from api.services.transcription.whisper_catalog import OFFICIAL_CHECKPOINTS, checkpoint


def audio_url():
    data = io.BytesIO()
    with wave.open(data, 'wb') as wav:
        wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        wav.writeframes(b'\x00\x00' * 160)
    return 'data:audio/wav;base64,' + base64.b64encode(data.getvalue()).decode()


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root / 'tiny.pt').write_bytes(b'fixture')
        self.entry = replace(checkpoint('tiny'), sha256=hashlib.sha256(b'fixture').hexdigest())
        self.store = Mock(root=root)
        self.store.get_checkpoint.return_value = (None, root)
        self.resources = ResourceManager(8 * 1024**3, {0: 8 * 1024**3})
        self.native = Mock()
        self.native.transcribe.return_value = {'text': 'hello', 'language': 'en'}
        self.factory = Mock(return_value=self.native)
        self.manager = TranscriptionManager(self.factory, self.resources, self.store)
        patcher = patch('api.services.transcription.transcription.checkpoint', return_value=self.entry)
        patcher.start()
        self.addCleanup(patcher.stop)
        enabled = patch("api.services.transcription.transcription.get_whisper_checkpoints", return_value={"tiny": self.entry})
        enabled.start()
        self.addCleanup(enabled.stop)

    def test_all_official_choices_and_aliases(self):
        self.assertEqual(len(OFFICIAL_CHECKPOINTS), 12)
        self.assertEqual(checkpoint('large'), checkpoint('large-v3'))
        self.assertEqual(checkpoint('turbo'), checkpoint('large-v3-turbo'))

    def test_native_input_and_resource_lease(self):
        def infer(samples, **kwargs):
            self.assertEqual(self.resources.snapshot()['reservations']['whisper']['active_leases'], 1)
            self.assertEqual(samples.shape, (160,))
            self.assertEqual(kwargs['task'], 'transcribe')
            self.assertFalse(kwargs['fp16'])
            return {'text': '', 'language': 'en'}
        self.native.transcribe.side_effect = infer
        result = self.manager.transcribe(audio_url(), 'tiny')
        self.assertEqual(result['raw_text'], '')
        self.factory.assert_called_once_with(str(Path(self.temp.name) / 'tiny.pt'), device='cpu')
        self.assertEqual(self.resources.snapshot()['reservations'], {})

    def test_integrity_failure_never_loads(self):
        (Path(self.temp.name) / 'tiny.pt').write_bytes(b'corrupted')
        with self.assertRaisesRegex(RuntimeFailure, 'integrity'):
            self.manager.transcribe(audio_url(), 'tiny')
        self.factory.assert_not_called()

    def test_reference_never_fetches(self):
        with self.assertRaises(RuntimeFailure):
            self.manager.transcribe('https://example.com/audio.wav')
        self.store.get_checkpoint.assert_not_called()

    def test_failure_releases_and_allows_retry(self):
        self.native.transcribe.side_effect = RuntimeError('fixture failure')
        with self.assertRaisesRegex(RuntimeFailure, 'fixture failure'):
            self.manager.transcribe(audio_url())
        self.assertEqual(self.resources.snapshot()['reservations'], {})
        self.native.transcribe.side_effect = None
        self.assertEqual(self.manager.transcribe(audio_url())['text'], 'hello')

    def test_malformed_audio_and_budget_reject_before_load(self):
        with self.assertRaisesRegex(RuntimeFailure, 'Invalid audio'):
            self.manager.transcribe('data:audio/wav;base64,bad')
        self.manager.resources = ResourceManager(1, {})
        with self.assertRaises(RuntimeFailure):
            self.manager.transcribe(audio_url())
        self.factory.assert_not_called()
        self.assertEqual(self.manager.resources.snapshot()['reservations'], {})

    def test_busy_does_not_interrupt_owner(self):
        self.manager.lock.acquire()
        try:
            with self.assertRaises(RuntimeFailure) as caught:
                self.manager.transcribe(audio_url())
            self.assertEqual(caught.exception.status_code, 409)
        finally:
            self.manager.lock.release()

    def test_gpu_lease_covers_native_call(self):
        with patch.dict('os.environ', {'KADAN_WHISPER_DEVICE': 'cuda:0'}):
            self.manager.transcribe(audio_url())
        self.assertTrue(self.native.transcribe.call_args.kwargs['fp16'])
        self.assertEqual(self.factory.call_args.kwargs['device'], 'cuda:0')
        self.assertEqual(self.resources.snapshot()['reservations'], {})

    def test_cached_manager_retains_ownership_through_failure_and_retry(self):
        _cached_transcription_manager.cache_clear()
        self.addCleanup(_cached_transcription_manager.cache_clear)
        with patch('api.services.transcription.transcription.TranscriptionManager', return_value=self.manager):
            owner = get_transcription_manager()
            def fail(samples, **kwargs):
                self.assertIs(get_transcription_manager(), owner)
                with self.assertRaises(RuntimeFailure) as busy:
                    get_transcription_manager().transcribe(audio_url())
                self.assertEqual(busy.exception.status_code, 409)
                self.assertEqual(self.resources.snapshot()['reservations']['whisper']['active_leases'], 1)
                raise RuntimeError('native failure')
            self.native.transcribe.side_effect = fail
            with self.assertRaisesRegex(RuntimeFailure, 'native failure'):
                owner.transcribe(audio_url())
            self.assertEqual(self.resources.snapshot()['reservations'], {})
            self.native.transcribe.side_effect = None
            self.assertEqual(get_transcription_manager().transcribe(audio_url())['text'], 'hello')
            self.assertEqual(self.resources.snapshot()['reservations'], {})
