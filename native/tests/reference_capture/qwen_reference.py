"""CPU-only test reference; never imported by the production API.

Preserves the reviewed packed-weight equations and installed HF architecture.
The capture entry points enforce synthetic or explicitly gated actual limits.
"""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .checkpoint import SafeTensorReader
from api.inference.resources import ResourceManager
from api.inference.llm.context import configure_context
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoTokenizer, Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding

from .offload import ExpertBank, ExpertCache
from .quantization import nvfp4_linear

check_cancel = ResourceManager._cancelled


def checkpoint_name(name):
    """Map text-only Transformers parameter names to the multimodal checkpoint prefix.

    The top-level lm_head name remains unchanged.
    """
    return name.replace('model.', 'model.language_model.', 1) if name.startswith('model.') else name


def read_projection(reader, prefix):
    """Read one CPU projection without expanding its quantized weights.

    NVFP4 uses uint8 [out, in/2], E4M3 [out, in/16] block scales
    and one global multiplier; FP8 uses [out, in] and scalar/row scales.
    Return the tensor mapping consumed by project(); reject other layouts.
    """
    weight = reader.tensor(prefix + '.weight')
    scale = reader.tensor(prefix + '.weight_scale')
    if weight.dtype == torch.uint8:
        global_scale = reader.tensor(prefix + '.weight_scale_2')
        if weight.ndim != 2 or weight.shape[1] % 8 or tuple(scale.shape) != (weight.shape[0], weight.shape[1] // 8):
            raise ValueError(f'{prefix}: unsupported NVFP4 shape')
        if scale.dtype != torch.float8_e4m3fn or global_scale.numel() != 1:
            raise ValueError(f'{prefix}: expected E4M3 block and scalar global scales')
        return {'weight': weight, 'scale': scale, 'global': global_scale}
    if weight.dtype == torch.float8_e4m3fn:
        if weight.ndim != 2 or scale.numel() not in (1, weight.shape[0]):
            raise ValueError(f'{prefix}: unsupported FP8 scale layout')
        return {'weight': weight, 'scale': scale}
    raise ValueError(f'{prefix}: unsupported quantized dtype {weight.dtype}')


def project(x, part):
    """Apply a cached projection to [..., in] inputs, returning [..., out].

    Weights and inputs must share a device. NVFP4 decoding uses the
    reference linear helper; FP8 weights multiply by scales in FP32 before casting
    to the input dtype. This is weight-only reference execution.
    """
    origin = x.device
    x = x.to(part['weight'].device)
    if 'global' in part:
        return nvfp4_linear(x, part['weight'], part['scale'], part['global']).to(origin)
    # Dense FP8 rows are tiled before transfer. Accumulate scaling in FP32.
    scale = part['scale'].float().reshape(-1, 1)
    return F.linear(x, (part['weight'].float() * scale).to(x.dtype)).to(origin)


class HostLinear(nn.Module):
    """Packed output-row tiles are copied only while their cache lease is held."""
    def __init__(self, keys, rows, out_features, cache, cancellation):
        """Bind ordered cache keys to adjacent output-row boundaries.

        The cancellation callback returns the current request event; this
        module owns no resident weight parameter.
        """
        super().__init__()
        self.keys, self.rows, self.out_features = keys, rows, out_features
        self.cache, self.cancellation = cache, cancellation

    def forward(self, x):
        """Assemble [..., out_features] from leased weight tiles on the input device.

        Check cancellation between tiles and release cached tensor references
        before another tile can trigger eviction.
        """
        result = torch.empty((*x.shape[:-1], self.out_features), dtype=x.dtype, device=x.device)
        for key, start, stop in zip(self.keys, self.rows[:-1], self.rows[1:]):
            check_cancel(self.cancellation())
            with self.cache.use(key) as part:
                result[..., start:stop] = project(x, part)
                del part
        return result


class QwenExperts(nn.Module):
    """Routing and shared experts remain in HF; only expert storage changes."""
    def __init__(self, layer, cache, cancellation):
        """Bind one layer to its packed expert cache and current-event callback."""
        super().__init__()
        self.layer, self.cache, self.cancellation = layer, cache, cancellation

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """Combine selected SiLU experts for flattened [tokens, hidden] inputs.

        Indices and weights are [tokens, top_k], supplied by the preserved
        router. One expert is leased at a time; the returned tensor matches
        the input shape. Shared-expert gating remains outside this module.
        """
        output = torch.zeros_like(hidden_states)
        for expert in top_k_index.unique().tolist():
            check_cancel(self.cancellation())
            token, slot = torch.where(top_k_index == expert)
            x = hidden_states[token]
            with self.cache.use((self.layer, expert)) as part:
                projections = {p: {k: v for name, v in part.items() if name.startswith(p + ':')
                                   for k in [name.split(':', 1)[1]]} for p in ('gate', 'up', 'down')}
                gated = F.silu(project(x, projections['gate'])) * project(x, projections['up'])
                y = project(gated, projections['down'])
                output.index_add_(0, token, (y * top_k_weights[token, slot, None]).to(output.dtype))
                del projections, part
        return output


class QwenAdapter:
    """Test-only CPU ownership for one-token numerical reference capture."""
    def __init__(self, device):
        if torch.device(device).type != 'cpu':
            raise ValueError('Reference capture is CPU-only')
        self.device = torch.device('cpu')
        self.model = self.tokenizer = self.cache = self.bank = self.reservation = None
        self._cancel = None

    def close(self):
        if self.cache is not None:
            self.cache.close()
        self.model = self.tokenizer = self.cache = self.bank = None
        if self.reservation is not None:
            self.reservation.release()
            self.reservation = None


def qwen_skeleton(config):
    """Build parameter storage on meta, then reconstruct CPU RoPE buffers.

    The device context intercepts allocation before Parameter creation,
    so construction does not temporarily materialize all expert weights.
    """
    with torch.device('meta'):
        model = Qwen3_5MoeForCausalLM(config)
    model.model.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config, device='cpu')
    return model


def build_qwen(entry, path, resources, device, cancel_event=None):
    """Load a local reviewed ModelOpt text checkpoint into a QwenAdapter.

    Keep routed experts and quantized dense row tiles packed in CPU RAM;
    replace their modules with Kadan cache consumers. Load remaining
    parameters and validate unresolved meta state under the host reservation.
    The tokenizer is local-only. Cancellation or malformed tensors release
    partial ownership and propagate an error; no GPU validation is implied.
    """

    check_cancel(cancel_event)
    path = Path(path)
    raw = json.loads((path / 'config.json').read_text())
    if raw.get('model_type') != 'qwen3_5_moe' or raw.get('quantization_config', {}).get('quant_method') != 'modelopt':
        raise ValueError('Qwen adapter requires the reviewed Qwen3.5 MoE ModelOpt checkpoint')
    config = Qwen3_5MoeTextConfig(**raw['text_config'])
    generation_path = path / 'generation_config.json'
    if generation_path.is_file():
        # The architecture config omits the chat end-of-turn token. Honor the
        # checkpoint's generation stops so a finished answer does not run to the budget.
        generation = json.loads(generation_path.read_text())
        config.eos_token_id = generation.get('eos_token_id', config.eos_token_id)
    config._attn_implementation = 'eager'
    if config.hidden_act != 'silu':
        raise ValueError('Qwen expert activation must be SiLU')
    reader = SafeTensorReader(path)
    adapter = QwenAdapter(device)
    bank_parts, dense_specs = {}, []
    # Admission precedes model/tensor allocations. Reserve all checkpoint bytes
    # conservatively (including ignored vision/MTP) and bounded compute scratch.
    host_bytes = sum(reader.nbytes(key) for key in reader.keys)
    dense_bytes = sum(reader.nbytes(key) * 2 for key in reader.keys
                      if (key.startswith('model.language_model.') or key.startswith('lm_head.')) and '.experts.' not in key)
    adapter.resources, adapter.owner = resources, 'qwen:' + entry.id
    adapter.reservation = resources.reserve(adapter.owner + ':host', 'llm',
        host_bytes=host_bytes + dense_bytes + 64 * 1024**2, cancel_event=cancel_event)
    try:
        with adapter.reservation.lease(cancel_event):
            # The device context intercepts torch.empty BEFORE Parameter wraps
            # it: no transient complete expert allocation in CPU RAM.
            model = qwen_skeleton(config)
            adapter.model = model
            for i, layer in enumerate(model.model.layers):
                for expert in range(config.num_experts):
                    check_cancel(cancel_event)
                    part = {}
                    for short, projection in (('gate', 'gate_proj'), ('up', 'up_proj'), ('down', 'down_proj')):
                        prefix = f'model.language_model.layers.{i}.mlp.experts.{expert}.{projection}'
                        part.update({short + ':' + key: tensor for key, tensor in read_projection(reader, prefix).items()})
                    bank_parts[i, expert] = part
                layer.mlp.experts = QwenExperts(i, None, lambda: adapter._cancel)
            for name, module in list(model.named_modules()):
                if not isinstance(module, nn.Linear):
                    continue
                prefix = checkpoint_name(name)
                if prefix + '.weight_scale' not in reader.keys:
                    continue
                if module.bias is not None:
                    raise ValueError(f'{prefix}: quantized bias is unsupported')
                part = read_projection(reader, prefix)
                # Bound each packed transfer while avoiding thousands of tiny cache
                # reservations for large dense matrices such as the vocabulary head.
                tile_rows = max(128, min(2048, (4 * 1024**2) // module.in_features))
                rows = list(range(0, module.out_features, tile_rows)) + [module.out_features]
                keys = []
                for start, stop in zip(rows[:-1], rows[1:]):
                    key = (name, start)
                    bank_parts[key] = {k: (v[start:stop] if v.numel() != 1 and v.shape[0] == module.out_features else v)
                                       for k, v in part.items()}
                    keys.append(key)
                dense_specs.append((name, keys, rows, module.out_features))
            adapter.bank = ExpertBank(bank_parts)
            adapter.cache = ExpertCache(adapter.bank, adapter.bank.max_expert_bytes, 'cpu')
            for name, keys, rows, out_features in dense_specs:
                model.set_submodule(name, HostLinear(keys, rows, out_features, adapter.cache, lambda: adapter._cancel))
            for layer in model.model.layers:
                layer.mlp.experts.cache = adapter.cache
            for name, parameter in list(model.named_parameters()):
                check_cancel(cancel_event)
                tensor = reader.tensor(checkpoint_name(name))
                if tuple(tensor.shape) != tuple(parameter.shape) or tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                    raise ValueError(f'{name}: missing, quantized or incompatible resident parameter')
                set_module_tensor_to_device(model, name, 'cpu', value=tensor,
                                            dtype=torch.float32 if name.endswith(('.A_log', '.dt_bias')) else torch.bfloat16)
            # Only explicitly reconstructed nonpersistent RoPE buffers are allowed.
            for name, buffer in list(model.named_buffers()):
                if buffer.is_meta:
                    raise ValueError(f'{name}: unresolved meta buffer')
                set_module_tensor_to_device(model, name, 'cpu', value=buffer)
            if any(p.is_meta for p in model.parameters()):
                raise ValueError('Unresolved Qwen meta parameters')
            model.eval()
            configure_context(adapter, None)
            adapter.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            check_cancel(cancel_event)
        return adapter
    except BaseException:
        adapter.close()
        raise
    finally:
        reader.close()
