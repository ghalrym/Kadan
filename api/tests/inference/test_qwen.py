import threading
import unittest

import torch
from torch import nn
from transformers import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

from api.inference.offload import ExpertBank, ExpertCache
from api.inference.qwen import HostLinear, QwenExperts, checkpoint_name, project, read_projection


def packed(rows, columns):
    return {'weight': torch.full((rows, columns // 2), 0x22, dtype=torch.uint8),
            'scale': torch.ones(rows, columns // 16).to(torch.float8_e4m3fn), 'global': torch.tensor(.1)}


class QwenTests(unittest.TestCase):
    def test_full_size_skeleton_never_allocates_expert_storage(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        from api.inference.qwen import qwen_skeleton
        devices = []
        class RecordEmpty(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                result = func(*args, **(kwargs or {}))
                if 'empty' in str(func) and isinstance(result, torch.Tensor):
                    devices.append(result.device.type)
                return result
        with RecordEmpty():
            model = qwen_skeleton(Qwen3_5MoeTextConfig())
        self.assertTrue(devices)
        self.assertEqual(set(devices), {'meta'})
        self.assertTrue(all(p.is_meta for p in model.parameters()))
        self.assertTrue(all(not b.is_meta for b in model.buffers()))

    def test_eviction_never_waits_on_adapter_lock(self):
        from api.inference.qwen import QwenAdapter
        from api.inference.resources import ResourceBusy
        adapter = QwenAdapter('cpu')
        locked, release = threading.Event(), threading.Event()
        def hold():
            with adapter._lock:
                locked.set()
                release.wait(5)
        thread = threading.Thread(target=hold)
        thread.start()
        self.assertTrue(locked.wait(2))
        try:
            with self.assertRaises(ResourceBusy):
                adapter._offload()
        finally:
            release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())

    def test_checkpoint_mapping(self):
        self.assertEqual(checkpoint_name('model.layers.0.mlp.gate.weight'), 'model.language_model.layers.0.mlp.gate.weight')
        self.assertEqual(checkpoint_name('lm_head.weight'), 'lm_head.weight')

    def test_routed_experts_preserve_shared_gate_and_router(self):
        torch.manual_seed(2)
        config = Qwen3_5MoeTextConfig(hidden_size=16, moe_intermediate_size=16,
                                    shared_expert_intermediate_size=16, num_experts=3, num_experts_per_tok=2)
        expected = Qwen3_5MoeSparseMoeBlock(config)
        parts = {}
        with torch.no_grad():
            for expert in range(3):
                item = {}
                for name in ('gate', 'up', 'down'):
                    p = packed(16, 16)
                    p['global'] = torch.tensor(.01 * (expert + 1))
                    item.update({name + ':' + key: value for key, value in p.items()})
                parts[0, expert] = item
                expected.experts.gate_up_proj[expert].fill_(.01 * (expert + 1))
                expected.experts.down_proj[expert].fill_(.01 * (expert + 1))
        import copy
        actual = copy.deepcopy(expected)
        bank = ExpertBank(parts)
        cache = ExpertCache(bank, bank.max_expert_bytes, 'cpu')
        actual.experts = QwenExperts(0, cache, lambda: None)
        x = torch.randn(1, 3, 16)
        with torch.inference_mode():
            torch.testing.assert_close(actual(x), expected(x), rtol=1e-5, atol=1e-6)
        self.assertGreater(cache.evictions, 0)
        cache.close()

    def test_dense_fp8_tiles_and_cancellation(self):
        p = {'weight': torch.arange(64).reshape(4, 16).float().to(torch.float8_e4m3fn),
             'scale': torch.tensor([1., 2., 3., 4.])}
        bank = ExpertBank({i: {k: v[i:i+1] for k, v in p.items()} for i in range(4)})
        cache = ExpertCache(bank, bank.max_expert_bytes, 'cpu')
        event = threading.Event()
        linear = HostLinear(list(range(4)), list(range(5)), 4, cache, lambda: event)
        x = torch.randn(2, 16)
        with torch.inference_mode():
            torch.testing.assert_close(linear(x), project(x, p))
        self.assertEqual(cache.evictions, 3)
        event.set()
        with self.assertRaises(InterruptedError):
            linear(x)
        cache.close()

    def test_reject_quantized_dtype_without_layout(self):
        class Reader:
            def tensor(self, key):
                return torch.ones(2, 8) if key.endswith('.weight') else torch.ones(2, 1)
        with self.assertRaisesRegex(ValueError, 'unsupported quantized dtype'):
            read_projection(Reader(), 'x')

    def test_tiny_checkpoint_load_and_hybrid_attention_forward(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from safetensors.torch import save_file
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast, Qwen3_5MoeForCausalLM
        from api.inference.qwen import build_qwen
        from api.inference.resources import ResourceManager
        config = Qwen3_5MoeTextConfig(vocab_size=32, hidden_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=1,
            linear_num_value_heads=2, moe_intermediate_size=16, shared_expert_intermediate_size=16,
            num_experts=2, num_experts_per_tok=1, layer_types=['linear_attention', 'full_attention'],
            rope_parameters={'rope_type': 'default', 'rope_theta': 10000., 'partial_rotary_factor': 1., 'mrope_section': [2, 2, 4]})
        original = Qwen3_5MoeForCausalLM(config)
        tensors = {checkpoint_name(k): v.detach().bfloat16().contiguous()
                   for k, v in original.state_dict().items() if '.experts.' not in k}
        for layer in range(2):
            for expert in range(2):
                for proj, rows, cols in [('gate_proj', 16, 32), ('up_proj', 16, 32), ('down_proj', 32, 16)]:
                    prefix = f'model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}'
                    for key, value in packed(rows, cols).items():
                        suffix = {'weight': 'weight', 'scale': 'weight_scale', 'global': 'weight_scale_2'}[key]
                        tensors[prefix + '.' + suffix] = value
        # Exercise both dense mixed formats including the lm_head offload path.
        name = 'model.language_model.layers.1.self_attn.q_proj'
        tensors[name + '.weight'] = tensors[name + '.weight'].to(torch.float8_e4m3fn)
        tensors[name + '.weight_scale'] = torch.tensor(1.)
        for key, value in packed(32, 32).items():
            tensors['lm_head.' + {'weight': 'weight', 'scale': 'weight_scale', 'global': 'weight_scale_2'}[key]] = value
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_file(tensors, path / 'model.safetensors')
            (path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {key: 'model.safetensors' for key in tensors}}))
            (path / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5_moe', 'text_config': config.to_dict(), 'quantization_config': {'quant_method': 'modelopt'}}))
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'x': 0, '[UNK]': 1}, unk_token='[UNK]')), unk_token='[UNK]')
            tokenizer.save_pretrained(path)
            resources = ResourceManager(1024**3, {})
            adapter = build_qwen(SimpleNamespace(id='small'), path, resources, 'cpu')
            with torch.inference_mode():
                output = adapter.model(torch.tensor([[0, 1]]), use_cache=True)
                self.assertTrue(torch.isfinite(output.logits).all())
                continued = adapter.model(torch.tensor([[0]]), past_key_values=output.past_key_values, use_cache=True)
                self.assertTrue(torch.isfinite(continued.logits).all())
                chunk = adapter.model(torch.zeros((1, 32), dtype=torch.long), use_cache=True)
                chunk = adapter.model(torch.zeros((1, 8), dtype=torch.long), past_key_values=chunk.past_key_values, use_cache=True)
                self.assertTrue(torch.isfinite(chunk.logits).all())
            self.assertGreater(adapter.cache.misses, 0)
            adapter._offload()
            self.assertFalse(adapter.is_resident)
            self.assertIsNotNone(adapter.bank)
            adapter._restore()
            self.assertTrue(adapter.is_resident)
            adapter.close()
            self.assertFalse(resources.snapshot()['reservations'])
            self.assertFalse(adapter.is_resident)

            # A partial meta skeleton must not mask missing-tensor errors or
            # retain its host reservation on failed loading.
            index_path = path / 'model.safetensors.index.json'
            index = json.loads(index_path.read_text())
            del index['weight_map']['model.language_model.embed_tokens.weight']
            index_path.write_text(json.dumps(index))
            with self.assertRaises(KeyError):
                build_qwen(SimpleNamespace(id='small'), path, resources, 'cpu')
            self.assertFalse(resources.snapshot()['reservations'])
