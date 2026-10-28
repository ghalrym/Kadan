import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experimental_worker.store import ModelStore


class ModelStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.metadata = {
            "id": "lfm3-test",
            "repo_id": "LiquidAI/Test",
            "revision": "abc",
            "license": "test",
            "license_url": "https://example.test/license",
            "kind": "llm",
            "context_limit": 8192,
        }
        self.content = b"checkpoint unit fixture"
        self.artifact = {
            "name": "config.json",
            "size": len(self.content),
            "algorithm": "sha256",
            "digest": hashlib.sha256(self.content).hexdigest(),
            "url": "https://huggingface.co/LiquidAI/Test/resolve/abc/config.json",
        }

    def store(self, **artifact_fields):
        artifact = {**self.artifact, **artifact_fields}
        return ModelStore(
            Path(self.directory.name), self.metadata, {"artifacts": [artifact]}
        )

    def test_acknowledgement_precedes_all_network_work(self):
        with patch("urllib.request.urlopen") as network:
            with self.assertRaisesRegex(ValueError, "acknowledge"):
                self.store().download(False, lambda _: None)
        network.assert_not_called()

    def test_verified_download_becomes_complete(self):
        store = self.store()
        progress = []
        with patch("urllib.request.urlopen", return_value=io.BytesIO(self.content)):
            result = store.download(True, progress.append)
        self.assertTrue(store.complete())
        self.assertEqual(result["models"][0]["status"], "complete")
        self.assertEqual(progress[-1]["value"], len(self.content))
        with patch("urllib.request.urlopen") as network:
            store.download(True, progress.append)
        network.assert_not_called()
        (store.root / "config.json").unlink()
        self.assertFalse(store.complete())

    def test_corruption_cannot_publish_completion(self):
        store = self.store(digest="0" * 64)
        with patch("urllib.request.urlopen", return_value=io.BytesIO(self.content)):
            with self.assertRaisesRegex(ValueError, "verification"):
                store.download(True, lambda _: None)
        self.assertFalse(store.marker.exists())

    def test_plain_git_blob_hash_includes_its_header(self):
        digest = hashlib.sha1(
            f"blob {len(self.content)}\0".encode() + self.content
        ).hexdigest()
        store = self.store(algorithm="git-sha1", digest=digest)
        with patch("urllib.request.urlopen", return_value=io.BytesIO(self.content)):
            store.download(True, lambda _: None)
        self.assertTrue(store.complete())

    def test_mutable_or_traversal_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "pinned revision"):
            self.store(
                url="https://huggingface.co/LiquidAI/Test/resolve/main/config.json"
            )
        with self.assertRaisesRegex(ValueError, "artifact path"):
            self.store(name="../config.json")

    def test_context_is_model_specific_and_persists_without_downloading(self):
        store = self.store()
        store.set_context(4096)
        self.assertEqual(self.store().context(), 4096)
        self.assertFalse(store.complete())
        store.set_context(None)
        self.assertIsNone(self.store().status()["models"][0]["context_limit"])
        self.assertEqual(
            self.store().status()["models"][0]["architecture_context_limit"], 8192
        )

    def test_invalid_context_does_not_replace_the_saved_setting(self):
        store = self.store()
        store.set_context(4096)
        for value in (0, False, 8193):
            with self.subTest(value=value), self.assertRaises(ValueError):
                store.set_context(value)
        self.assertEqual(store.context(), 4096)
