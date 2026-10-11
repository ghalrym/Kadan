import unittest
from types import SimpleNamespace

from api.inference.llm.context import ContextLimitError, resolve_context


class ContextTests(unittest.TestCase):
    def test_default_is_architecture_max(self):
        self.assertEqual(resolve_context({'max_position_embeddings': 131072}), (131072, 131072))

    def test_override_is_explicit(self):
        self.assertEqual(resolve_context({'max_position_embeddings': 131072}, 20000), (131072, 20000))

    def test_nested_text_configuration_is_supported(self):
        config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=32768))
        self.assertEqual(resolve_context(config, 4096), (32768, 4096))

    def test_invalid_limits_are_rejected(self):
        for value in (0, -1, True, 131073, '4096'):
            with self.subTest(value=value), self.assertRaises(ContextLimitError):
                resolve_context({'max_position_embeddings': 131072}, value)

    def test_checkpoint_must_declare_a_positive_limit(self):
        for value in (None, 0, -1, True, '4096'):
            with self.subTest(value=value), self.assertRaises(ContextLimitError):
                resolve_context({'max_position_embeddings': value})
