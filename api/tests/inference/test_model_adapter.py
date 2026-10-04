import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from transformers import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

from api.inference.model_adapter import GptOssOffloadedExperts
from api.inference.offload import ExpertBank, ExpertCache, tensor_bytes
from api.inference.quantization import dequantize_mxfp4
from api.inference.generation import autoregressive_generate


class ModelAdapterTests(unittest.TestCase):
    def test_offloaded_experts_match_transformers_equation_with_eviction(self):
        torch.manual_seed(12)
        config = GptOssConfig(hidden_size=32, intermediate_size=32, num_local_experts=3)
        reference = GptOssExperts(config)
        banks = {}
        for expert in range(3):
            tensors = {
                'gate_up_blocks': torch.randint(0, 255, (64, 1, 16), dtype=torch.uint8),
                'gate_up_scales': torch.full((64, 1), 123, dtype=torch.uint8),
                'gate_up_bias': torch.randn(64),
                'down_blocks': torch.randint(0, 255, (32, 1, 16), dtype=torch.uint8),
                'down_scales': torch.full((32, 1), 123, dtype=torch.uint8),
                'down_bias': torch.randn(32),
            }
            banks[0, expert] = tensors
            with torch.no_grad():
                reference.gate_up_proj[expert].copy_(dequantize_mxfp4(tensors['gate_up_blocks'], tensors['gate_up_scales']).T)
                reference.down_proj[expert].copy_(dequantize_mxfp4(tensors['down_blocks'], tensors['down_scales']).T)
                reference.gate_up_proj_bias[expert].copy_(tensors['gate_up_bias'])
                reference.down_proj_bias[expert].copy_(tensors['down_bias'])
        cache = ExpertCache(ExpertBank(banks), tensor_bytes(banks[0, 0]), device='cpu')
        ours = GptOssOffloadedExperts(0, cache)
        hidden = torch.randn(4, 32)
        indices = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2]])
        weights = torch.tensor([[.2, .8], [.3, .7], [.4, .6], [.5, .5]])
        with torch.inference_mode():
            expected = reference(hidden, indices, weights)
            actual = ours(hidden, indices, weights)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(cache.evictions, 2)
        self.assertLessEqual(cache.resident_bytes, cache.capacity_bytes)

    def test_generation_uses_prefill_then_cached_single_token_and_stops(self):
        calls = []
        class Model:
            config = SimpleNamespace(eos_token_id=9)
            def __call__(self, **kwargs):
                calls.append(kwargs)
                logits = torch.zeros(1, 1, 10)
                logits[0, 0, 3 if len(calls) == 1 else 9] = 10
                return SimpleNamespace(logits=logits, past_key_values='cache')
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2]])
        tokenizer.decode.return_value = 'answer'
        result = autoregressive_generate(Model(), tokenizer, [{'role': 'user', 'text': 'hi'}], 'cpu', max_new_tokens=3)
        self.assertEqual(result, 'answer')
        self.assertIsNone(calls[0]['past_key_values'])
        self.assertEqual(calls[1]['past_key_values'], 'cache')
        self.assertEqual(calls[1]['input_ids'].tolist(), [[3]])
        tokenizer.decode.assert_called_once_with([3], skip_special_tokens=True)

    def test_cancel_and_context_limit_fail_before_forward(self):
        cancel = threading.Event()
        cancel.set()
        model = Mock()
        with self.assertRaises(InterruptedError):
            autoregressive_generate(model, Mock(), [], 'cpu', cancel_event=cancel)
        model.assert_not_called()
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = torch.ones(1, 10, dtype=torch.long)
        with self.assertRaises(ValueError):
            autoregressive_generate(model, tokenizer, [], 'cpu', max_new_tokens=5, context_limit=12)
        model.assert_not_called()

    def test_tiny_full_model_logits_and_kv_cache_match_resident_reference(self):
        from copy import deepcopy
        from transformers import GptOssForCausalLM
        torch.manual_seed(7)
        config = GptOssConfig(hidden_size=32, intermediate_size=32, num_local_experts=2,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, num_experts_per_tok=2, vocab_size=64)
        config._attn_implementation = 'eager'
        config._experts_implementation = 'eager'
        reference = GptOssForCausalLM(config).eval()
        banks = {}
        for index, layer in enumerate(reference.model.layers):
            for expert in range(2):
                tensors = {
                    'gate_up_blocks': torch.randint(0, 255, (64, 1, 16), dtype=torch.uint8),
                    'gate_up_scales': torch.full((64, 1), 120, dtype=torch.uint8),
                    'gate_up_bias': torch.randn(64) * .01,
                    'down_blocks': torch.randint(0, 255, (32, 1, 16), dtype=torch.uint8),
                    'down_scales': torch.full((32, 1), 120, dtype=torch.uint8),
                    'down_bias': torch.randn(32) * .01,
                }
                banks[index, expert] = tensors
                with torch.no_grad():
                    layer.mlp.experts.gate_up_proj[expert].copy_(dequantize_mxfp4(tensors['gate_up_blocks'], tensors['gate_up_scales']).T)
                    layer.mlp.experts.down_proj[expert].copy_(dequantize_mxfp4(tensors['down_blocks'], tensors['down_scales']).T)
                    layer.mlp.experts.gate_up_proj_bias[expert].copy_(tensors['gate_up_bias'])
                    layer.mlp.experts.down_proj_bias[expert].copy_(tensors['down_bias'])
        ours = deepcopy(reference)
        cache = ExpertCache(ExpertBank(banks), tensor_bytes(banks[0, 0]), device='cpu')
        for index, layer in enumerate(ours.model.layers):
            layer.mlp.experts = GptOssOffloadedExperts(index, cache)
        with torch.inference_mode():
            ref = reference(input_ids=torch.tensor([[1, 5, 2]]), use_cache=True)
            actual = ours(input_ids=torch.tensor([[1, 5, 2]]), use_cache=True)
            torch.testing.assert_close(actual.logits, ref.logits)
            ref_next = reference(input_ids=torch.tensor([[3]]), past_key_values=ref.past_key_values, use_cache=True)
            actual_next = ours(input_ids=torch.tensor([[3]]), past_key_values=actual.past_key_values, use_cache=True)
            torch.testing.assert_close(actual_next.logits, ref_next.logits)
        self.assertGreater(cache.evictions, 4)
