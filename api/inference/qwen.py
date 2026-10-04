"""Kadan-owned Qwen3.5/3.6 text adapter with demand-loaded packed experts.

Transformers supplies Apache-2.0 attention/GatedDeltaNet architecture definitions;
Kadan owns checkpoint loading, packed host banks, byte-limited device caching and
execution lifecycle. No FreeToken imports or engine. NVIDIA's reviewed ModelOpt
checkpoint mixes FP8 dense projections and NVFP4 routed/shared experts/lm_head.
Reference W4A16 execution does not reproduce calibrated activation quantization.
"""
import json
import threading
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .checkpoint import SafeTensorReader
from .generation import autoregressive_generate, check_cancel
from .offload import ExpertBank, ExpertCache
from .quantization import nvfp4_linear
from .resources import ResourceBusy


def checkpoint_name(name):
    return name.replace('model.', 'model.language_model.', 1) if name.startswith('model.') else name


def read_projection(reader, prefix):
    """Accept only canonical ModelOpt packed bytes or scaled FP8 weights."""
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
    if 'global' in part:
        return nvfp4_linear(x, part['weight'], part['scale'], part['global'])
    # Dense FP8 rows are tiled before transfer. Accumulate scaling in FP32.
    scale = part['scale'].float().reshape(-1, 1)
    return F.linear(x, (part['weight'].float() * scale).to(x.dtype))


class HostLinear(nn.Module):
    """Packed output-row tiles are copied only while their cache lease is held."""
    def __init__(self, keys, rows, out_features, cache, cancellation):
        super().__init__()
        self.keys, self.rows, self.out_features = keys, rows, out_features
        self.cache, self.cancellation = cache, cancellation

    def forward(self, x):
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
        super().__init__()
        self.layer, self.cache, self.cancellation = layer, cache, cancellation

    def forward(self, hidden_states, top_k_index, top_k_weights):
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
    def __init__(self, device):
        self.device = torch.device(device)
        self.model = self.tokenizer = self.cache = self.bank = self.reservation = None
        self._lock = threading.RLock()
        self._cancel = None
        self.device_reservation = None

    @property
    def is_resident(self):
        return self.model is not None and self.device_reservation is not None

    def _restore(self, event=None):
        if self.is_resident:
            return
        if self.model is None:
            raise RuntimeError('Qwen runtime is closed')
        self.device_reservation = self.resources.reserve(self.owner + ':device', 'llm',
            device_bytes={self.device.index or 0: self.gpu_bytes} if self.device.type == 'cuda' else {},
            evict=self._offload, cancel_event=event)
        try:
            with self.device_reservation.lease(event):
                self.model.to(self.device)
        except BaseException:
            self._offload()
            raise

    def _offload(self):
        # Admission callbacks must never wait for an adapter lock: generation
        # can hold it while waiting for admission, creating lock inversion.
        if not self._lock.acquire(blocking=False):
            raise ResourceBusy('Qwen is busy; cannot evict during restore/generation')
        try:
            if self.cache is not None:
                self.cache.clear()
            if self.model is not None:
                self.model.to('cpu')
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
            if self.device_reservation is not None:
                self.device_reservation.release()
                self.device_reservation = None
        finally:
            self._lock.release()

    def generate(self, messages, max_new_tokens=256, cancel_event=None):
        with self._lock:
            with self.reservation.lease(cancel_event):
                self._restore(cancel_event)
                with self.device_reservation.lease(cancel_event):
                    self._cancel = cancel_event
                    try:
                        return autoregressive_generate(self.model, self.tokenizer, messages, self.device,
                                                       cancel_event, max_new_tokens, context_limit=self.effective_context_limit,
                                                       resources=self.resources, owner=self.owner)
                    finally:
                        self._cancel = None

    def close(self):
        with self._lock:
            # Failure can leave a partial meta skeleton: discard it, never try
            # copying unresolved meta tensors to CPU during cleanup.
            if self.cache is not None:
                self.cache.close()
            self.model = self.tokenizer = self.cache = self.bank = None
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
            if self.device_reservation is not None:
                self.device_reservation.release()
                self.device_reservation = None
            if self.reservation is not None:
                self.reservation.release()
                self.reservation = None


def qwen_skeleton(config):
    from transformers import Qwen3_5MoeForCausalLM
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
    with torch.device('meta'):
        model = Qwen3_5MoeForCausalLM(config)
    model.model.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config, device='cpu')
    return model


def build_qwen(entry, path, resources, device, cancel_event=None):
    from accelerate.utils import set_module_tensor_to_device
    from transformers import AutoTokenizer, Qwen3_5MoeTextConfig

    check_cancel(cancel_event)
    path = Path(path)
    raw = json.loads((path / 'config.json').read_text())
    if raw.get('model_type') != 'qwen3_5_moe' or raw.get('quantization_config', {}).get('quant_method') != 'modelopt':
        raise ValueError('Qwen adapter requires the reviewed Qwen3.5 MoE ModelOpt checkpoint')
    config = Qwen3_5MoeTextConfig(**raw['text_config'])
    config._attn_implementation = 'eager'
    if config.hidden_act != 'silu':
        raise ValueError('Qwen expert activation must be SiLU')
    reader = SafeTensorReader(path)
    adapter = QwenAdapter(device)
    bank_parts, dense_specs = {}, []
    # Admission precedes model/tensor allocations. Reserve all checkpoint bytes
    # conservatively (including ignored vision/MTP) and bounded compute scratch.
    host_bytes = sum(reader.nbytes(key) for key in reader.keys)
    cache_bytes = 512 * 1024**2
    dense_bytes = sum(reader.nbytes(key) * 2 for key in reader.keys
                      if (key.startswith('model.language_model.') or key.startswith('lm_head.')) and '.experts.' not in key)
    gpu_bytes = dense_bytes + cache_bytes + 2 * 1024**3
    adapter.resources, adapter.owner, adapter.gpu_bytes = resources, 'qwen:' + entry.id, gpu_bytes
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
                rows = list(range(0, module.out_features, 128)) + [module.out_features]
                keys = []
                for start, stop in zip(rows[:-1], rows[1:]):
                    key = (name, start)
                    bank_parts[key] = {k: (v[start:stop] if v.numel() != 1 and v.shape[0] == module.out_features else v)
                                       for k, v in part.items()}
                    keys.append(key)
                dense_specs.append((name, keys, rows, module.out_features))
            adapter.bank = ExpertBank(bank_parts)
            adapter.cache = ExpertCache(adapter.bank, cache_bytes, device)
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
            # Exact resident tensors plus cache and bounded request scratch.
            adapter.gpu_bytes = sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers())) + cache_bytes + 2 * 1024**3
            model.eval()
            from api.inference.context import configure_context
            configure_context(adapter, None)
            adapter.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            check_cancel(cancel_event)
            adapter._restore(cancel_event)
        return adapter
    except BaseException:
        adapter.close()
        raise
