"""Tiny real Diffusers loading smoke: no pretrained weights, network, or image inference."""
import importlib
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest

import torch
from safetensors.torch import save_file

from api.inference.klein_quant import load_transformer

# The minimal numerical-test environment may omit the optional Diffusers runtime.
# The complete repository install must run this test, including CI.
DIFFUSERS = importlib.import_module('diffusers') if importlib.util.find_spec('diffusers') else None


@unittest.skipIf(DIFFUSERS is None, 'Install the repository image runtime for the real loader smoke')
class KleinRuntimeTests(unittest.TestCase):
    def test_original_quantized_keys_load_with_local_config_and_real_converter(self):
        model = DIFFUSERS.Flux2Transformer2DModel(
            in_channels=16, out_channels=16, num_layers=1, num_single_layers=1,
            attention_head_dim=16, num_attention_heads=1, joint_attention_dim=16,
            timestep_guidance_channels=16, mlp_ratio=2., axes_dims_rope=(4, 4, 4, 4),
            guidance_embeds=False)
        template = model.state_dict()
        # Explicit original-format names test the converter boundary as well as
        # quantized decode. Shapes come from a tiny randomly initialized model.
        names = {
            'img_in': 'x_embedder', 'txt_in': 'context_embedder',
            'time_in.in_layer': 'time_guidance_embed.timestep_embedder.linear_1',
            'time_in.out_layer': 'time_guidance_embed.timestep_embedder.linear_2',
            'double_stream_modulation_img.lin': 'double_stream_modulation_img.linear',
            'double_stream_modulation_txt.lin': 'double_stream_modulation_txt.linear',
            'single_stream_modulation.lin': 'single_stream_modulation.linear',
            'final_layer.adaLN_modulation.1': 'norm_out.linear', 'final_layer.linear': 'proj_out',
            'double_blocks.0.img_attn.proj': 'transformer_blocks.0.attn.to_out.0',
            'double_blocks.0.txt_attn.proj': 'transformer_blocks.0.attn.to_add_out',
            'double_blocks.0.img_mlp.0': 'transformer_blocks.0.ff.linear_in',
            'double_blocks.0.img_mlp.2': 'transformer_blocks.0.ff.linear_out',
            'double_blocks.0.txt_mlp.0': 'transformer_blocks.0.ff_context.linear_in',
            'double_blocks.0.txt_mlp.2': 'transformer_blocks.0.ff_context.linear_out',
            'single_blocks.0.linear1': 'single_transformer_blocks.0.attn.to_qkv_mlp_proj',
            'single_blocks.0.linear2': 'single_transformer_blocks.0.attn.to_out',
        }
        norms = {
            'double_blocks.0.img_attn.norm.query_norm': 'transformer_blocks.0.attn.norm_q',
            'double_blocks.0.img_attn.norm.key_norm': 'transformer_blocks.0.attn.norm_k',
            'double_blocks.0.txt_attn.norm.query_norm': 'transformer_blocks.0.attn.norm_added_q',
            'double_blocks.0.txt_attn.norm.key_norm': 'transformer_blocks.0.attn.norm_added_k',
            'single_blocks.0.norm.query_norm': 'single_transformer_blocks.0.attn.norm_q',
            'single_blocks.0.norm.key_norm': 'single_transformer_blocks.0.attn.norm_k',
        }
        dense = {original + '.weight': torch.ones_like(template[target + '.weight'])
                 for original, target in names.items()}
        dense.update({original + '.scale': torch.ones_like(template[target + '.weight'])
                      for original, target in norms.items()})
        for modality, projections in (('img', ('to_q', 'to_k', 'to_v')),
                                      ('txt', ('add_q_proj', 'add_k_proj', 'add_v_proj'))):
            dense[f'double_blocks.0.{modality}_attn.qkv.weight'] = torch.cat([
                torch.ones_like(template[f'transformer_blocks.0.attn.{name}.weight']) for name in projections])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model.save_config(root / 'transformer')
            for quantization in ('fp8', 'nvfp4'):
                with self.subTest(quantization=quantization):
                    state = {}
                    for key, tensor in dense.items():
                        if tensor.ndim != 2:
                            state[key] = tensor
                        elif quantization == 'fp8':
                            state[key] = tensor.to(torch.float8_e4m3fn)
                        else:
                            rows, columns = tensor.shape
                            self.assertEqual(columns % 16, 0)
                            prefix = key.removesuffix('weight')
                            state[key] = torch.full((rows, columns // 2), 0x22, dtype=torch.uint8)
                            state[prefix + 'comfy_quant'] = torch.tensor(list(b'{"format":"nvfp4"}'), dtype=torch.uint8)
                            padded_scales = ((rows + 127) // 128) * 128 * ((columns // 16 + 3) // 4) * 4
                            state[prefix + 'weight_scale'] = torch.ones(padded_scales).to(torch.float8_e4m3fn)
                            state[prefix + 'weight_scale_2'] = torch.tensor(1.)
                    # Also exercise the common original-checkpoint prefix.
                    save_file({'model.diffusion_model.' + key: value for key, value in state.items()}, str(root / 'tiny.safetensors'))
                    actual = load_transformer(root, 'tiny.safetensors', quantization, torch.float32,
                        DIFFUSERS, threading.Event())
                    self.assertEqual(set(actual.state_dict()), set(template))
                    self.assertLess(sum(parameter.numel() for parameter in actual.parameters()), 100_000)
                    for key, value in actual.state_dict().items():
                        self.assertEqual(value.device.type, 'cpu')
                        torch.testing.assert_close(value, torch.ones_like(template[key]), msg=key)
