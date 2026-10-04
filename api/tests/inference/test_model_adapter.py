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
        config._experts_implementation = 'eager'
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
            config = SimpleNamespace(eos_token_id=9, max_position_embeddings=4096)
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

    def test_checkpoint_loader_streams_tiny_fixture_without_resident_expert_parameters(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from safetensors.torch import save_file
        from transformers import GptOssForCausalLM
        from api.inference.model_adapter import GptOssAdapter
        config = GptOssConfig(hidden_size=32, intermediate_size=32, num_local_experts=2,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, num_experts_per_tok=2, vocab_size=64)
        config._attn_implementation = 'eager'
        source = GptOssForCausalLM(config).eval()
        tensors = {name: value.contiguous() for name, value in source.state_dict().items() if '.experts.' not in name}
        prefix = 'model.layers.0.mlp.experts.'
        tensors.update({
            prefix + 'gate_up_proj_blocks': torch.zeros((2, 64, 1, 16), dtype=torch.uint8),
            prefix + 'gate_up_proj_scales': torch.full((2, 64, 1), 127, dtype=torch.uint8),
            prefix + 'gate_up_proj_bias': torch.zeros((2, 64)),
            prefix + 'down_proj_blocks': torch.zeros((2, 32, 1, 16), dtype=torch.uint8),
            prefix + 'down_proj_scales': torch.full((2, 32, 1), 127, dtype=torch.uint8),
            prefix + 'down_proj_bias': torch.zeros((2, 32)),
        })
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            save_file(tensors, path / 'model.safetensors')
            (path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {name: 'model.safetensors' for name in tensors}}))
            data = config.to_dict()
            data['quantization_config'] = {'quant_method': 'mxfp4'}
            (path / 'config.json').write_text(json.dumps(data))
            resources = Mock()
            resources.capacity = SimpleNamespace(device_bytes={0: 2**30})
            with patch('api.inference.model_adapter.ExpertCache', side_effect=lambda bank, size, device, **kwargs: ExpertCache(bank, size, 'cpu')) as cache_factory, \
                 patch.object(GptOssAdapter, '_restore'), patch('transformers.AutoTokenizer.from_pretrained', return_value=Mock()):
                adapter = GptOssAdapter(SimpleNamespace(id='medium'), path, resources, 'cuda:0')
            try:
                self.assertIs(cache_factory.call_args.kwargs['resources'], resources)
                self.assertEqual(cache_factory.call_args.args[1], resources.capacity.device_bytes[0])
                self.assertIn(':experts', cache_factory.call_args.kwargs['owner'])
                resident = sum(t.numel() * t.element_size() for t in
                    list(adapter.model.parameters()) + list(adapter.model.buffers()))
                self.assertEqual(adapter.device_budget, resident + 16 * 1024**2)
                self.assertFalse(any(tensor.is_meta for tensor in adapter.model.parameters()))
                self.assertFalse(any(tensor.is_meta for tensor in adapter.model.buffers()))
                self.assertFalse(any('.experts.' in name for name, _ in adapter.model.named_parameters()))
                with torch.inference_mode():
                    logits = adapter.model(input_ids=torch.tensor([[1, 2]])).logits
                self.assertTrue(torch.isfinite(logits).all())
            finally:
                adapter.close()

    def test_chunked_prefill_matches_full_prompt_greedy_tokens(self):
        from transformers import GptOssForCausalLM
        torch.manual_seed(11)
        config = GptOssConfig(hidden_size=32, intermediate_size=32, num_local_experts=2,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, num_experts_per_tok=2, vocab_size=64, eos_token_id=None)
        config._attn_implementation = 'eager'
        config._experts_implementation = 'eager'
        model = GptOssForCausalLM(config).eval()
        tokens = torch.randint(0, 64, (1, 65))
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = tokens
        tokenizer.eos_token_id = None
        tokenizer.decode.side_effect = lambda generated, **kw: str(generated)
        expected = []
        with torch.inference_mode():
            inputs, cache = tokens, None
            for _ in range(3):
                output = model(input_ids=inputs, past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                inputs = output.logits[:, -1, :].argmax(-1, keepdim=True)
                expected.append(inputs.item())
        actual = autoregressive_generate(model, tokenizer, [{'role': 'user', 'text': 'fixture'}],
                                         'cpu', max_new_tokens=3)
        self.assertEqual(actual, str(expected))

    def test_full_size_skeleton_has_no_materialized_experts_or_meta_rope(self):
        from api.inference.model_adapter import build_gptoss_skeleton
        from unittest.mock import patch
        # Default architecture dimensions are full-size, but every parameter and
        # every temporary factory allocation must be meta, not merely moved there
        # after nn.Parameter registration.
        config = GptOssConfig()
        config._attn_implementation = 'eager'
        native_empty = torch.empty
        def guarded_empty(*args, **kwargs):
            import math
            shape = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args
            if math.prod(shape) > 1_000_000:
                actual_device = torch.device(kwargs.get('device') or torch.get_default_device())
                self.assertEqual(actual_device.type, 'meta', 'Large constructor allocation escaped meta')
            return native_empty(*args, **kwargs)
        with patch('torch.empty', guarded_empty):
            model = build_gptoss_skeleton(config)
        self.assertTrue(all(parameter.is_meta for parameter in model.parameters()))
        self.assertTrue(all(not buffer.is_meta for buffer in model.buffers()))
        self.assertTrue(torch.isfinite(model.model.rotary_emb.inv_freq).all())
