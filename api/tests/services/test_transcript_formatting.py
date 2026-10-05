import unittest
import torch
from unittest.mock import Mock

from api.inference.resources import ResourceManager
from api.services.model_catalog import CATALOG, allowed_asset, validate_assets
from api.services.transcript_formatting import (
    ATTRIBUTION, CONTROL, RAM_BYTES, SYSTEM, NativeNormalizer,
    TranscriptFormatter, transcript_chunks,
)


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text)


class FormatterTests(unittest.TestCase):
    def setUp(self):
        self.resources = ResourceManager(RAM_BYTES, {})
        self.native = Mock()
        self.native.normalize.return_value = 'The answer is 43.'
        self.loader = Mock(return_value=self.native)
        self.formatter = TranscriptFormatter(self.loader, self.resources)

    def test_preserves_raw_and_releases_allocations_before_accounting(self):
        self.native.close.side_effect = lambda: self.assertTrue(self.resources.snapshot()['reservations'])
        result = self.formatter.format('uh forty two no forty three', 'en')
        self.assertEqual(result.raw_text, 'uh forty two no forty three')
        self.assertEqual(result.text, 'The answer is 43.')
        self.assertEqual(result.formatting_model, ATTRIBUTION)
        self.assertEqual(result.formatting_status, 'formatted')
        self.assertFalse(self.resources.snapshot()['reservations'])
        self.native.close.assert_called_once()

    def test_bypass_does_not_load(self):
        for language, enabled, status in [('en', False, 'disabled'), ('fr', True, 'unsupported_language'),
                                          (None, True, 'unsupported_language')]:
            result = self.formatter.format('bonjour', language, enabled)
            self.assertEqual((result.text, result.formatting_status), ('bonjour', status))
        self.loader.assert_not_called()

    def test_empty_output_valid_and_failure_retains_raw(self):
        self.native.normalize.return_value = ''
        self.assertEqual(self.formatter.format('um', 'en').text, '')
        self.native.normalize.side_effect = RuntimeError('failure')
        result = self.formatter.format('keep this', 'en')
        self.assertEqual((result.text, result.formatting_status), ('keep this', 'unavailable'))
        self.assertFalse(self.resources.snapshot()['reservations'])

    def test_load_failure_and_insufficient_ram_release_budget(self):
        self.loader.side_effect = RuntimeError('failed load')
        self.assertEqual(self.formatter.format('raw', 'en').formatting_status, 'unavailable')
        self.assertFalse(self.resources.snapshot()['reservations'])
        formatter = TranscriptFormatter(self.loader, ResourceManager(1, {}))
        self.loader.reset_mock()
        self.assertEqual(formatter.format('raw', 'en').text, 'raw')
        self.loader.assert_not_called()

    def test_busy_request_does_not_allocate_or_change_raw(self):
        self.formatter._lock.acquire()
        try:
            result = self.formatter.format('raw', 'en')
            self.assertEqual(result.formatting_status, 'busy')
            self.loader.assert_not_called()
        finally:
            self.formatter._lock.release()

    def test_long_transcript_keeps_every_character_with_bounded_chunks(self):
        raw = 'First sentence. Second sentence! ' + 'unpunctuated ' * 300 + '🦊' * 1200
        chunks = list(transcript_chunks(raw, CharacterTokenizer()))
        self.assertEqual(''.join(chunks), raw)
        self.assertTrue(all(0 < len(chunk) <= 1000 for chunk in chunks))
        self.assertEqual(chunks[0], 'First sentence. Second sentence!')

    def test_native_uses_exact_prompt_and_disables_thinking(self):
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda text, **kwargs: list(text)
        tokenizer.apply_chat_template.return_value = 'prompt'
        tokenizer.return_value = {'input_ids': torch.tensor([[1, 2]])}
        tokenizer.decode.return_value = ''
        model = Mock()
        model.generation_config.eos_token_id = 9
        model.generate.return_value = torch.tensor([[1, 2, 9]])
        normalizer = NativeNormalizer(tokenizer, model, torch)
        self.assertEqual(normalizer.normalize('um'), '')
        tokenizer.apply_chat_template.assert_called_once_with([
            {'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': CONTROL + '\num'}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        self.assertFalse(model.generate.call_args.kwargs['do_sample'])
        self.assertEqual(model.generate.call_args.kwargs['max_new_tokens'], 35)

    def test_catalog_requires_weights_tokenizer_and_attribution_without_shard_index(self):
        entry = CATALOG['s1-mini']
        validate_assets(entry, set(entry.required_files))
        self.assertEqual(entry.kind, 'formatting')
        for missing in ('model.safetensors', 'NOTICE', 'chat_template.jinja'):
            with self.assertRaises(ValueError):
                validate_assets(entry, set(entry.required_files) - {missing})
        self.assertFalse(allowed_asset('model.py', entry))
        self.assertFalse(allowed_asset('../model.safetensors', entry))
