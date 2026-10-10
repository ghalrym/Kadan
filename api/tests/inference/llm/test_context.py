import unittest
from types import SimpleNamespace

from api.inference.llm.context import ContextLimitError, ContextMemoryError, configure_context, estimate_request_memory, resolve_context


class ContextTests(unittest.TestCase):
    def config(self, **kwargs):
        return dict(model_type='gpt_oss', max_position_embeddings=131072,
                    num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
                    hidden_size=512, head_dim=64, **kwargs)

    def test_default_is_architecture_max_and_override_is_explicit(self):
        adapter = SimpleNamespace(model=SimpleNamespace(config=self.config()))
        configure_context(adapter, None)
        self.assertEqual(adapter.effective_context_limit, 131072)
        configure_context(adapter, 20000)
        self.assertEqual(adapter.effective_context_limit, 20000)
        self.assertEqual(adapter.supported_context_limit, 131072)
        for value in (0, -1, True, 131073, '4096'):
            with self.assertRaises(ContextLimitError):
                resolve_context(self.config(), value)

    def test_budget_depends_on_request_not_configured_max(self):
        small = estimate_request_memory(self.config(), 4096, 3840)
        large = estimate_request_memory(self.config(), 100000, 99744)
        self.assertGreater(large.cache_bytes, small.cache_bytes * 20)
        self.assertGreater(large.workspace_bytes, small.workspace_bytes)
        altered = self.config()
        altered['max_position_embeddings'] = 1000000
        self.assertEqual(small, estimate_request_memory(altered, 4096, 3840))

    def test_glm_latent_state_and_expanded_workspace_grow(self):
        c = dict(model_type='glm5_next_text', num_hidden_layers=2, num_attention_heads=8,
                 hidden_size=512, layer_types=['linear_attention', 'indexed_attention'],
                 kv_lora_rank=128, qk_rope_head_dim=0, qk_nope_head_dim=64, v_head_dim=64,
                 index_head_dim=32, index_n_heads=16, index_kpool=4, index_topk=2048, linear_num_heads=4, linear_head_dim=32)
        a, b = [estimate_request_memory(c, length, length - 256) for length in (4096, 200000)]
        self.assertGreater(b.cache_bytes, a.cache_bytes)
        self.assertGreater(b.workspace_bytes, a.workspace_bytes)
        c['qk_rope_head_dim'] = -1
        with self.assertRaises(ContextMemoryError):
            estimate_request_memory(c, 4096, 3840)

    def test_qwen_hybrid_budget_and_invalid_architectures(self):
        c = dict(model_type='qwen3_5_moe_text', num_hidden_layers=4, num_attention_heads=8,
                 num_key_value_heads=2, hidden_size=512, head_dim=64,
                 layer_types=['linear_attention', 'full_attention'] * 2,
                 linear_num_value_heads=8, linear_num_key_heads=4,
                 linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4)
        a = estimate_request_memory(c, 4096, 3840)
        b = estimate_request_memory(c, 100000, 99744)
        self.assertGreater(b.cache_bytes, a.cache_bytes)
        self.assertGreater(b.workspace_bytes, a.workspace_bytes)
        for change in ({'model_type': 'unknown'}, {'num_attention_heads': 0},
                       {'layer_types': ['indexed_attention'] * 4}):
            with self.assertRaises(ContextMemoryError):
                estimate_request_memory({**c, **change}, 4096, 3840)
        with self.assertRaises(ContextMemoryError):
            estimate_request_memory(self.config(layer_types=['linear_attention'] * 4), 4096, 3840)
