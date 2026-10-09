"""CPU component/architecture tests, not checkpoint parity or GPU validation."""
import unittest
import threading

try:
    import torch
    from api.inference.llm.glm import GlmExperts, read_dense, source_names, TextLM, validate_expert, HostLinear, GlmAdapter
    from api.inference.llm.offload import ExpertBank, ExpertCache
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional inference dependencies unavailable')
class GlmTests(unittest.TestCase):
    def test_checkpoint_names(self):
        prefix = 'model.language_model.layers.3.'
        self.assertEqual(source_names('model.layers.3.attn_hc.fn'), (prefix + 'hc_attn_fn',))
        self.assertEqual(source_names('model.layers.3.self_attn.forget_gate.A_log'), (prefix + 'self_attn.A_log',))
        self.assertEqual(source_names('model.layers.3.self_attn.conv1d.weight'), tuple(
            prefix + f'self_attn.{x}_conv1d.weight' for x in ('q', 'k', 'v')))
        self.assertEqual(source_names('lm_head.weight'), ('lm_head.weight',))

    def test_fp8_block_scale_and_reject_missing(self):
        tensors = {'a.weight': torch.ones(129, 129).to(torch.float8_e4m3fn),
                   'a.weight_scale_inv': torch.tensor([[2., 3.], [4., 5.]])}
        class Reader:
            keys = set(tensors)
            def tensor(self, name):
                return tensors[name]
        value = read_dense(Reader(), ('a.weight',), torch.float32)
        self.assertEqual(value[0, 0], 2.)
        self.assertEqual(value[-1, -1], 5.)
        del tensors['a.weight_scale_inv']
        Reader.keys.remove('a.weight_scale_inv')
        with self.assertRaisesRegex(ValueError, 'missing block scales'):
            read_dense(Reader(), ('a.weight',), torch.float32)

    def test_dense_host_tiles(self):
        torch.manual_seed(9)
        weight = torch.randn(7, 16)
        bias = torch.randn(7)
        bank = ExpertBank({i: {'weight': weight[i:i + 1], 'bias': bias[i:i + 1]} for i in range(7)})
        cache = ExpertCache(bank, bank.max_expert_bytes, device='cpu')
        layer = HostLinear(weight, list(range(7)), cache, lambda: None)
        inputs = torch.randn(3, 16)
        torch.testing.assert_close(layer(inputs), torch.nn.functional.linear(inputs, weight, bias))
        self.assertEqual(list(layer.parameters()), [])
        self.assertGreater(cache.evictions, 0)
        cache.close()

    def test_handoff_preserves_host_bank_and_restores(self):
        from api.inference.resources import ResourceManager
        resources = ResourceManager(4096, {0: 4096})
        adapter = GlmAdapter()
        adapter.device = torch.device('cpu')
        adapter.resources, adapter.owner, adapter.gpu_bytes = resources, 'test-glm', 1024
        adapter.model = torch.nn.Linear(2, 2)
        adapter.bank = ExpertBank({0: {'weight': torch.ones(2, 2)}})
        adapter.cache = ExpertCache(adapter.bank, 128, 'cpu')
        adapter.reservation = resources.reserve('test-glm:host', 'llm', host_bytes=128)
        adapter._restore()
        original_bank = adapter.bank
        with resources.exclusive('image-request'):
            self.assertFalse(adapter.is_resident)
            self.assertIs(adapter.bank, original_bank)
            self.assertIn('test-glm:host', resources.snapshot()['reservations'])
        adapter._restore()
        self.assertTrue(adapter.is_resident)
        adapter.close()
        self.assertEqual(resources.snapshot()['reservations'], {})

    def test_full_size_meta_skeleton_has_no_allocated_parameters(self):
        try:
            from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
            from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel
        except ImportError:
            self.skipTest('Pinned Transformers source required')
        from accelerate import init_empty_weights
        with init_empty_weights(include_buffers=True):
            config = Glm5NextTextConfig()
            model = TextLM(Glm5NextTextModel(config), config)
        self.assertTrue(all(p.is_meta for p in model.parameters()))
        self.assertTrue(all(b.is_meta for b in model.buffers()))
        self.assertFalse(set(dict(model.named_buffers())) - set(model.state_dict()))

    def test_inference_glm_hybrid_with_host_experts(self):
        try:
            from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
            from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel
        except ImportError:
            self.skipTest('GLM needs pinned Transformers source, not release 5.16.0')
        torch.manual_seed(7)
        config = Glm5NextTextConfig(
            hidden_size=32, intermediate_size=64, moe_intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
            n_group=1, topk_group=1, layer_types=['linear_attention', 'deepseek_sparse_attention'],
            mlp_layer_types=['dense', 'sparse'], linear_num_heads=2, linear_head_dim=16,
            vocab_size=64, pad_token_id=0, eos_token_id=63, q_lora_rank=16,
            kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=0,
            v_head_dim=16, index_n_heads=2, index_head_dim=16,
            index_topk=4, index_kpool=2, swiglu_limit=10.,
        )
        config._attn_implementation = 'eager'
        model = TextLM(Glm5NextTextModel(config), config).eval()
        native = model.model.layers[1].mlp.experts
        # Four constant-weight experts with exactly representable NVFP4 values.
        with torch.no_grad():
            native.gate_up_proj.fill_(.5)
            native.down_proj.fill_(.5)
        values = {}
        for expert in range(4):
            values[(1, expert)] = {}
            for part in ('gate_proj', 'up_proj', 'down_proj'):
                values[(1, expert)].update({
                    f'{part}.weight_packed': torch.full((32, 16), 0x11, dtype=torch.uint8),
                    f'{part}.weight_scale': torch.ones((32, 2)).to(torch.float8_e4m3fn),
                    f'{part}.weight_global_scale': torch.ones(1),
                })
        for expert in values.values():
            validate_expert(expert, 32, 32)
        corrupt = dict(values[(1, 0)])
        corrupt['gate_proj.weight_global_scale'] = torch.tensor([0.])
        with self.assertRaisesRegex(ValueError, 'global scale value'):
            validate_expert(corrupt, 32, 32)
        bank = ExpertBank(values)
        cache = ExpertCache(bank, bank.max_expert_bytes, device='cpu')
        inputs = torch.tensor([[1, 2, 3]])
        with torch.inference_mode():
            reference = model(inputs).logits
            cache.close()
            linears = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, torch.nn.Linear)]
            for name, layer in linears:
                values[('linear', name)] = {'weight': layer.weight.detach()}
                if layer.bias is not None:
                    values[('linear', name)]['bias'] = layer.bias.detach()
            bank = ExpertBank(values)
            cache = ExpertCache(bank, bank.max_expert_bytes, device='cpu')
            for name, layer in linears:
                parent, _, leaf = name.rpartition('.')
                setattr(model.get_submodule(parent) if parent else model, leaf,
                        HostLinear(layer.weight, [('linear', name)], cache, lambda: None))
            model.model.layers[1].mlp.experts = GlmExperts(cache, 1, 10.)
            output = model(inputs)
            torch.testing.assert_close(output.logits, reference, rtol=1e-4, atol=1e-5)
            # Incremental decode exercises KDA recurrent state and DSA/indexer KV.
            next_output = model(torch.tensor([[4]]), past_key_values=output.past_key_values)
            self.assertTrue(torch.isfinite(next_output.logits).all())
            self.assertGreater(cache.misses, 0)
            self.assertLessEqual(cache.resident_bytes, bank.max_expert_bytes)
        cancelled = threading.Event()
        cancelled.set()
        model.model.layers[1].mlp.experts.cancel_event = cancelled
        with self.assertRaises(InterruptedError):
            model.model.layers[1].mlp.experts(torch.zeros(1, 32), torch.tensor([[0]]), torch.ones(1, 1))
        cache.close()
        self.assertEqual(cache.resident_bytes, 0)


if __name__ == '__main__':
    unittest.main()
