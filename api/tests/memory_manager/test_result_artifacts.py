import json
import os
from pathlib import Path
import tempfile
import time
import unittest

from api.inference.errors import InferenceFailure
from api.memory_manager.result_artifacts import ResultArtifacts


class ResultArtifactTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)/'results'
        self.store = ResultArtifacts(self.root, 10, max_file=32, max_bytes=48)
        self.first, self.second = 'a'*32, 'b'*32

    def test_bounded_storage_retention_and_round_trip(self):
        data=json.dumps({'value':'x'*10})
        metadata=self.store.store(self.first,data)
        self.assertEqual(self.store.read(self.first,metadata),json.loads(data))
        with self.assertRaisesRegex(ValueError,'full'):self.store.store(self.second,data+' ' * 9)
        os.utime(self.store.path(self.first),(time.time()-20,)*2)
        self.store.store(self.second,data)
        self.assertFalse(self.store.path(self.first).exists())
        with self.assertRaisesRegex(ValueError,'size limit'):self.store.store('c'*32,'x'*33)

    def test_tamper_symlink_and_identity_fail_closed(self):
        metadata=self.store.store(self.first,'{"ok":true}')
        path=self.store.path(self.first);path.write_text('{"ok":null}')
        with self.assertRaises(InferenceFailure):self.store.read(self.first,metadata)
        path.unlink();path.symlink_to('/dev/zero')
        with self.assertRaises(InferenceFailure):self.store.read(self.first,metadata)
        with self.assertRaises(ValueError):self.store.path('../outside')
