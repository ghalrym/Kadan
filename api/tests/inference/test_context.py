import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from api.inference.context import ContextLimitError, ContextMemoryError, configure_context, estimate_request_memory, resolve_context


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

    def test_long_prompt_not_truncated_and_admission_precedes_forward(self):
        import torch
        from api.inference.generation import autoregressive_generate
        from api.inference.resources import ResourceManager
        model = Mock()
        model.config = SimpleNamespace(**self.config(), eos_token_id=2)
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = torch.ones((1, 10000), dtype=torch.long)
        with self.assertRaises(ContextMemoryError):
            autoregressive_generate(model, tokenizer, [{'role': 'user', 'text': 'large'}], 'cuda:0',
                                    resources=ResourceManager(10**9, {0: 1}), max_new_tokens=256)
        model.assert_not_called()
        self.assertEqual(tokenizer.apply_chat_template.call_args.kwargs.get('truncation'), None)
        with self.assertRaises(ContextLimitError):
            autoregressive_generate(model, tokenizer, [], 'cpu', context_limit=10000)
        model.assert_not_called()

    def test_prefill_chunk_boundaries_keep_every_token(self):
        import torch
        from api.inference.generation import autoregressive_generate
        for length in (32, 33, 64, 65):
            seen = []
            class Model:
                config = SimpleNamespace(max_position_embeddings=1000, eos_token_id=1)
                def __call__(self, input_ids, **kwargs):
                    seen.extend(input_ids[0].tolist())
                    return SimpleNamespace(logits=torch.tensor([[[0., 1.]]]), past_key_values=None)
            tokenizer = Mock()
            tokenizer.apply_chat_template.return_value = torch.arange(length).reshape(1, -1)
            tokenizer.decode.return_value = ''
            autoregressive_generate(Model(), tokenizer, [], 'cpu')
            self.assertEqual(seen, list(range(length)))

    def test_failure_and_cancellation_release_request_reservation(self):
        import threading
        import torch
        from api.inference.generation import autoregressive_generate
        from api.inference.resources import ResourceManager
        for fails in (True, False):
            event = threading.Event()
            resources = ResourceManager(10**8, {})
            config = SimpleNamespace(**self.config(), eos_token_id=1)
            class Model:
                def __init__(self):
                    self.config = config
                def __call__(self, **kwargs):
                    self_test.assertTrue(resources.snapshot()['reservations'])
                    if fails:
                        raise RuntimeError('controlled forward failure')
                    event.set()
                    return SimpleNamespace(logits=torch.tensor([[[1., 0.]]]), past_key_values=None)
            self_test = self
            tokenizer = Mock()
            tokenizer.apply_chat_template.return_value = torch.ones((1, 65), dtype=torch.long)
            with self.assertRaises((RuntimeError, InterruptedError)):
                autoregressive_generate(Model(), tokenizer, [], 'cpu', resources=resources, cancel_event=event)
            self.assertEqual(resources.snapshot()['reservations'], {})

    def test_working_expert_must_fit_before_forward(self):
        import torch
        from api.inference.generation import autoregressive_generate
        from api.inference.resources import ResourceManager
        budget = estimate_request_memory(self.config(), 266, 10)
        resources = ResourceManager(10**9, {0: budget.device_bytes + 1023})
        model = Mock()
        model.config = SimpleNamespace(**self.config(), eos_token_id=1)
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = torch.ones((1, 10), dtype=torch.long)
        with self.assertRaisesRegex(ContextMemoryError, 'working expert'):
            autoregressive_generate(model, tokenizer, [], 'cuda:0', resources=resources, expert_headroom_bytes=1024)
        model.assert_not_called()
        self.assertEqual(resources.snapshot()['reservations'], {})

    def test_preflight_evicts_idle_expert_for_request_then_working_slot(self):
        from contextlib import nullcontext
        from unittest.mock import patch
        from api.inference.generation import request_memory
        from api.inference.resources import ResourceManager
        budget = estimate_request_memory(self.config(), 4096, 3840)
        resources = ResourceManager(10**9, {0: budget.device_bytes + 1024})
        evicted = []
        def evict():
            evicted.append(True)
            old.release()
        old = resources.reserve('idle-expert', 'llm', device_bytes={0: 1024}, evict=evict)
        with patch('torch.cuda.synchronize'), patch('torch.cuda.empty_cache'), patch('torch.cuda.device', return_value=nullcontext()):
            with request_memory(resources, 'model', self.config(), 4096, 3840, 'cuda:0', None, 1024):
                self.assertEqual(evicted, [True])
                self.assertEqual(sum(r['host_bytes'] for r in resources.snapshot()['reservations'].values()), budget.host_bytes + 2048)
                working = resources.reserve('working-expert', 'llm', device_bytes={0: 1024})
                working.release()
        self.assertEqual(resources.snapshot()['reservations'], {})

    def test_cache_race_failure_becomes_admission_error_and_releases_request(self):
        import torch
        from api.inference.generation import autoregressive_generate
        from api.inference.resources import ResourceManager, ResourceExhausted
        resources = ResourceManager(10**9, {})
        model = Mock(side_effect=ResourceExhausted('competing allocation'))
        model.config = SimpleNamespace(**self.config(), eos_token_id=1)
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = torch.ones((1, 10), dtype=torch.long)
        with self.assertRaisesRegex(ContextMemoryError, 'memory changed'):
            autoregressive_generate(model, tokenizer, [], 'cpu', resources=resources)
        self.assertEqual(resources.snapshot()['reservations'], {})

    def test_long_input_uses_all_chunks_without_fixed_fixture_limit(self):
        import torch
        from api.inference.generation import autoregressive_generate
        lengths = []
        class Model:
            config = SimpleNamespace(max_position_embeddings=20000, eos_token_id=1)
            def __call__(self, input_ids, **kwargs):
                lengths.append(input_ids.shape[-1])
                return SimpleNamespace(logits=torch.tensor([[[0., 1.]]]), past_key_values=None)
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = torch.ones((1, 10001), dtype=torch.long)
        tokenizer.decode.return_value = ''
        autoregressive_generate(Model(), tokenizer, [], 'cpu')
        self.assertEqual(sum(lengths), 10001)
        self.assertLessEqual(max(lengths), 32)


if __name__ == '__main__':
    unittest.main()
