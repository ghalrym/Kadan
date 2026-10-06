"""Exercise the pinned Laya library with a tiny synthetic local CPU checkpoint.

No downloaded weights or accuracy/performance claim: this checks the real loader,
tokenizer, typed head and Kadan preflight against the installed dependency.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from api.inference.decisions.model import load_laya, preflight, parse_answer, translate_questions, clear_tokenizer_cache
from api.services.runtime import RuntimeFailure
from api.tests.inference.decisions.test_model import questions


@unittest.skipUnless(importlib.util.find_spec('laya'), 'Optional pinned Laya runtime is not installed')
class LayaContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import ModernBertConfig, AutoModel, PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models, pre_tokenizers
        from safetensors.torch import save_file
        from laya.common import DecisionModel
        cls.directory = tempfile.TemporaryDirectory()
        path = Path(cls.directory.name)
        cfg = ModernBertConfig(vocab_size=16, hidden_size=64, intermediate_size=128,
                               num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=128,
                               pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2)
        cfg.save_pretrained(path / 'encoder')
        tokenizer = Tokenizer(models.WordLevel({'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3, '[UNK]': 4}, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='[PAD]', cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]', unk_token='[UNK]')
        fast.save_pretrained(path / 'tokenizer')
        config = dict(encoder=str(path / 'encoder'), head_layers=1, max_len=128, head_max_len=64, act_costs={'act': 0})
        (path / 'rl_agent_config.json').write_text(json.dumps(config))
        torch.manual_seed(0)
        model = DecisionModel(AutoModel.from_config(cfg, attn_implementation='sdpa'), head_layers=1, n_act=2)
        save_file(model.state_dict(), path / 'model.safetensors')
        with patch.dict('os.environ', {'KADAN_LAYA_MODEL': str(path)}):
            cls.agent = load_laya()

    @classmethod
    def tearDownClass(cls):
        clear_tokenizer_cache(tokenizer=cls.agent.tok)
        del cls.agent
        cls.directory.cleanup()

    def test_cpu_native_typed_outputs(self):
        self.assertEqual(self.agent.device.type, 'cpu')
        definitions = translate_questions(questions())
        preflight(self.agent, 'Service failed', definitions)
        result = self.agent.predict('Service failed', definitions)
        for question in questions():
            answer = parse_answer(question, result['answers'][question.key])
            self.assertEqual(answer.key, question.key)
            self.assertIsNotNone(answer.confidence)

    def test_pinned_hub_download_and_cpu_only_loader_arguments(self):
        path = Path(self.directory.name)
        with patch.dict('os.environ', {}, clear=True), \
                patch('huggingface_hub.snapshot_download', return_value=str(path)) as download, \
                patch('laya.load', return_value=self.agent) as loader:
            self.assertIs(load_laya(), self.agent)
        self.assertEqual(download.call_args.kwargs['revision'], '7b928d828b7b0e022f929d9bd2e44165aa270148')
        self.assertEqual(loader.call_args.kwargs, dict(device='cpu', fast=False, compile=False))

    def test_oversized_configuration_rejected_before_model_allocation(self):
        path = Path(self.directory.name)
        encoder_path = path / 'encoder/config.json'
        original = encoder_path.read_text()
        config = json.loads(original)
        config['hidden_size'] = 1000000
        encoder_path.write_text(json.dumps(config))
        try:
            with patch.dict('os.environ', {'KADAN_LAYA_MODEL': str(path)}), patch('laya.load') as loader:
                with self.assertRaises(RuntimeFailure):
                    load_laya()
                loader.assert_not_called()
        finally:
            encoder_path.write_text(original)
        with patch.dict('os.environ', {'KADAN_LAYA_MODEL': str(path), 'KADAN_LAYA_RAM_BYTES': '1'}), patch('laya.load') as loader:
            with self.assertRaises(RuntimeFailure):
                load_laya()
            loader.assert_not_called()

    def test_exact_preflight_rejects_each_truncation_path(self):
        definitions = translate_questions(questions())
        cases = [('state ' * 200, definitions)]
        for field in ('instructions', 'criteria'):
            definition = dict(definitions['cause'])
            definition[field] = 'word ' * 100 if field == 'instructions' else {'load': 'word ' * 100}
            cases.append(('state', {'cause': definition}))
        cases.append(('[MASK]', definitions))
        for state, question_defs in cases:
            with self.subTest(state=state[:20], questions=question_defs), self.assertRaises(RuntimeFailure) as caught:
                preflight(self.agent, state, question_defs)
            self.assertEqual(caught.exception.status_code, 422)
