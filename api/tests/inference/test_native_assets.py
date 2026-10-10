"""Fresh-volume native exports use verified downloaded inputs, not server fixtures."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from api.inference.native_assets import ensure_assets
from api.inference.resources import ResourceManager, ResourceCancelled, ResourceExhausted
from api.inference.tts.native_worker import verify_tokenizer
from api.inference.tts.speech_runtime import SpeechUnavailable
from api.inference.stt.native_worker import resolve, HOST_BUDGET, MAX_PCM
from api.inference.stt.catalog import checkpoint
from dataclasses import replace


class NativeAssetsTests(unittest.TestCase):
    def setUp(self):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup);self.root=Path(folder.name)
        self.resources=ResourceManager(8*1024**3,{})
    def tts_source(self):
        root=self.root/'downloaded-tts';root.mkdir()
        for name,data in {'vocab.json':{'a':0,'b':1,'<|endoftext|>':2},'config.json':{'model_type':'gpt2'},
                'tokenizer_config.json':{'tokenizer_class':'GPT2Tokenizer','bos_token':'<|endoftext|>','eos_token':'<|endoftext|>','unk_token':'<|endoftext|>'}}.items():
            (root/name).write_text(json.dumps(data))
        (root/'merges.txt').write_text('#version: 0.2\n')
        return root
    def test_fresh_tts_export_verified_and_reused(self):
        source=self.tts_source();destination=self.root/'native/tts'
        ensure_assets('tts',source,destination,self.resources)
        verify_tokenizer(source,destination/'tokenizer.json')
        before=(destination/'manifest.json').stat().st_mtime_ns
        ensure_assets('tts',source,destination,self.resources)
        self.assertEqual((destination/'manifest.json').stat().st_mtime_ns,before)
        self.assertEqual(self.resources.snapshot()['reservations'],{})
        (source/'vocab.json').write_text('{}')
        with self.assertRaises(SpeechUnavailable):verify_tokenizer(source,destination/'tokenizer.json')
    def test_fresh_whisper_export_and_default_resolver(self):
        source=self.root/'downloaded-whisper';source.mkdir();weight=source/'tiny.pt'
        dims=dict(n_mels=80,n_audio_ctx=2,n_audio_state=4,n_audio_head=1,n_audio_layer=1,n_vocab=51865,n_text_ctx=8,n_text_state=4,n_text_head=1,n_text_layer=1)
        torch.save({'dims':dims,'model_state_dict':{'encoder.positional_embedding':torch.zeros(2,4)}},weight)
        digest=hashlib.sha256(weight.read_bytes()).hexdigest();entry=replace(checkpoint('tiny'),sha256=digest)
        store=SimpleNamespace(root=self.root,get_checkpoint=lambda model:(None,source))
        with patch.dict('os.environ',{},clear=True),patch('api.inference.stt.native_worker.model_manager',store),patch(
                'api.inference.stt.native_worker.checkpoint',return_value=entry),patch(
                'api.inference.stt.native_worker.subprocess.run',return_value=SimpleNamespace(stdout=f'whisper 1 cpu en {MAX_PCM} {HOST_BUDGET}\n'.encode())):
            command=resolve('tiny',resources=self.resources)
        destination=self.root/'native/whisper/tiny'
        self.assertEqual(command[1],str(destination));self.assertTrue((destination/'assets/english.tokens').is_file())
        self.assertEqual(json.loads((destination/'export.json').read_text())['source_sha256'],digest)
        self.assertEqual(self.resources.snapshot()['reservations'],{})
    def test_cancel_and_admission_precede_conversion(self):
        source=self.tts_source();destination=self.root/'native/tts';cancel=threading.Event();cancel.set()
        with self.assertRaises(ResourceCancelled):ensure_assets('tts',source,destination,self.resources,cancel)
        self.assertFalse(destination.exists())
        with self.assertRaises(ResourceExhausted):ensure_assets('tts',source,destination,ResourceManager(1,{}))
        self.assertFalse(destination.exists())
    def test_failed_integrity_leaves_no_partial_export_or_reservation(self):
        source=self.root/'tiny.pt';source.write_bytes(b'corrupt download');destination=self.root/'native/whisper/tiny'
        with self.assertRaisesRegex(ValueError,'SHA-256'):ensure_assets('whisper',source,destination,self.resources,expected='0'*64)
        self.assertFalse(destination.exists());self.assertEqual(list(destination.parent.glob('.native-assets-*')),[])
        self.assertEqual(self.resources.snapshot()['reservations'],{})

    def test_disk_cleanup_error_does_not_lose_host_reservation(self):
        source=self.tts_source();destination=self.root/'native/tts'
        with patch('api.inference.native_assets.shutil.rmtree',side_effect=OSError('disk cleanup failed')):
            with self.assertRaisesRegex(OSError,'disk cleanup failed'):
                ensure_assets('tts',source,destination,self.resources)
        self.assertEqual(self.resources.snapshot()['reservations'],{})
        verify_tokenizer(source,destination/'tokenizer.json')
