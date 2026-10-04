"""Kadan model adapters. No external inference service or automatic quant loader."""
import gc
import json
import threading
from pathlib import Path

import torch
from torch import nn

from api.inference.checkpoint import SafeTensorReader
from api.inference.generation import autoregressive_generate, check_cancel
from api.inference.offload import ExpertBank, ExpertCache
from api.inference.quantization import mxfp4_linear


class GptOssOffloadedExperts(nn.Module):
    """The GPT-OSS expert equation; routing stays in Transformers' model layer."""
    def __init__(self, layer, cache, cancel_event=None):
        super().__init__()
        self.layer, self.cache, self.cancel_event = layer, cache, cancel_event

    def forward(self, hidden_states, router_indices=None, routing_weights=None):
        output = torch.zeros_like(hidden_states)
        for expert in torch.unique(router_indices).tolist():
            check_cancel(self.cancel_event)
            token_idx, topk_idx = torch.where(router_indices == expert)
            with self.cache.use((self.layer, expert)) as tensors:
                # Decode only the selected projection; never all experts or layers.
                projected = mxfp4_linear(hidden_states[token_idx], tensors['gate_up_blocks'],
                    tensors['gate_up_scales'], tensors['gate_up_bias'].to(hidden_states.dtype), scratch_bytes=16 * 1024**2)
                gate, up = projected[..., ::2], projected[..., 1::2]
                gate = gate.clamp(max=7.0)
                up = up.clamp(min=-7.0, max=7.0)
                activation = (up + 1) * gate * torch.sigmoid(1.702 * gate)
                result = mxfp4_linear(activation, tensors['down_blocks'],
                    tensors['down_scales'], tensors['down_bias'].to(hidden_states.dtype), scratch_bytes=16 * 1024**2)
                output.index_add_(0, token_idx, (result * routing_weights[token_idx, topk_idx, None]).to(output.dtype))
        return output


def build_gptoss_skeleton(config):
    from transformers import GptOssForCausalLM
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding
    with torch.device('meta'):
        model = GptOssForCausalLM(config)
    # RoPE buffers are deterministic configuration-derived state, not checkpoint
    # tensors. Rebuild them on CPU instead of leaving uninitialized to_empty data.
    model.model.rotary_emb = GptOssRotaryEmbedding(config)
    return model


class GptOssAdapter:
    def __init__(self, entry, path, resources, device, cancel_event=None):
        self.resources, self.device = resources, torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('Production inference requires one CUDA GPU; CPU is for tensor tests only')
        self.device_index = self.device.index or 0
        self._lock = threading.RLock()
        self.model = self.tokenizer = self.bank = self.cache = None
        self.host_reservation = self.device_reservation = None
        self.is_resident = False
        self.owner = f'llm:{id(self)}'
        self.cancel_event = cancel_event
        reader = SafeTensorReader(path)
        config_data = json.loads((Path(path) / 'config.json').read_text())
        if config_data.get('model_type') != 'gpt_oss':
            raise ValueError('Expected the native GPT-OSS checkpoint family')
        if config_data.get('quantization_config', {}).get('quant_method') != 'mxfp4':
            raise ValueError('Expected native MXFP4 GPT-OSS weights')
        config_data.pop('quantization_config', None)
        from accelerate.utils import set_module_tensor_to_device
        from transformers import AutoTokenizer, GptOssConfig
        config = GptOssConfig(**config_data)
        config._attn_implementation = 'eager'
        config._experts_implementation = 'eager'
        layers, experts = config.num_hidden_layers, config.num_local_experts
        expert_names = {f'model.layers.{layer}.mlp.experts.{name}'
                        for layer in range(layers)
                        for name in ('gate_up_proj_blocks', 'gate_up_proj_scales', 'gate_up_proj_bias',
                                     'down_proj_blocks', 'down_proj_scales', 'down_proj_bias')}
        if not expert_names <= reader.keys:
            raise ValueError('Native packed GPT-OSS expert tensors are missing')
        dense_bytes = sum(reader.nbytes(name) for name in reader.keys - expert_names)
        packed_bytes = sum(reader.nbytes(name) for name in expert_names)
        # Host preserves packed experts plus dense weights when GPU eviction occurs.
        self.host_reservation = resources.reserve(self.owner + ':host', 'llm',
            host_bytes=packed_bytes + 2 * dense_bytes + 512 * 1024**2)
        self.device_budget = dense_bytes + 16 * 1024**2
        self.cache_bytes = resources.capacity.device_bytes[self.device_index]
        try:
            check_cancel(cancel_event)
            self.model = build_gptoss_skeleton(config)
            banks = {}
            for layer in range(layers):
                check_cancel(cancel_event)
                prefix = f'model.layers.{layer}.mlp.experts.'
                tensors = {name: reader.tensor(prefix + name) for name in (
                    'gate_up_proj_blocks', 'gate_up_proj_scales', 'gate_up_proj_bias',
                    'down_proj_blocks', 'down_proj_scales', 'down_proj_bias')}
                expected = {
                    'gate_up_proj_blocks': (experts, 2 * config.intermediate_size, config.hidden_size // 32, 16),
                    'gate_up_proj_scales': (experts, 2 * config.intermediate_size, config.hidden_size // 32),
                    'gate_up_proj_bias': (experts, 2 * config.intermediate_size),
                    'down_proj_blocks': (experts, config.hidden_size, config.intermediate_size // 32, 16),
                    'down_proj_scales': (experts, config.hidden_size, config.intermediate_size // 32),
                    'down_proj_bias': (experts, config.hidden_size),
                }
                if any(tuple(tensors[name].shape) != shape for name, shape in expected.items()):
                    raise ValueError(f'Unsupported GPT-OSS packed tensor shape in layer {layer}')
                for expert in range(experts):
                    banks[layer, expert] = {name.replace('_proj', ''): tensor[expert]
                                             for name, tensor in tensors.items()}
            self.bank = ExpertBank(banks)
            self.cache = ExpertCache(self.bank, self.cache_bytes, device=str(self.device),
                                     resources=resources, owner=self.owner + ':experts')
            for layer in range(layers):
                self.model.model.layers[layer].mlp.experts = GptOssOffloadedExperts(layer, self.cache, cancel_event)
            required = dict(self.model.named_parameters())
            if set(required) != reader.keys - expert_names:
                missing = set(required) - reader.keys
                unknown = reader.keys - expert_names - set(required)
                raise ValueError(f'Unrecognized GPT-OSS dense checkpoint mapping: missing={sorted(missing)[:3]}, unknown={sorted(unknown)[:3]}')
            for name, parameter in required.items():
                check_cancel(cancel_event)
                tensor = reader.tensor(name)
                if tensor.shape != parameter.shape or tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                    raise ValueError(f'Unsupported dense tensor {name}')
                set_module_tensor_to_device(self.model, name, 'cpu', value=tensor, dtype=torch.bfloat16)
            if any(buffer.is_meta for buffer in self.model.buffers()):
                raise ValueError('Uninitialized model buffers after meta construction')
            self.model.eval()
            self.device_budget = sum(t.numel() * t.element_size() for t in
                list(self.model.parameters()) + list(self.model.buffers())) + 16 * 1024**2
            from api.inference.context import configure_context
            configure_context(self, None)
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            self._restore(cancel_event)
        except BaseException:
            self.close()
            raise

    def _restore(self, cancel_event=None):
        if self.is_resident:
            return
        check_cancel(cancel_event)
        self.device_reservation = self.resources.reserve(self.owner + ':device', 'llm',
            device_bytes={self.device_index: self.device_budget}, evict=self._evict, cancel_event=cancel_event)
        try:
            with self.device_reservation.lease(cancel_event=cancel_event):
                self.model.to(self.device)
                self.is_resident = True
        except BaseException:
            self._evict()
            if self.device_reservation is not None:
                self.device_reservation.release()
                self.device_reservation = None
            raise

    def _evict(self):
        with self._lock:
            if self.cache is not None:
                self.cache.clear()
            if self.model is not None:
                self.model.to('cpu')
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            self.is_resident = False

    def generate(self, messages, max_new_tokens=256, cancel_event=None):
        if self.model is None:
            raise RuntimeError('Model has been closed')
        # Admission can run eviction callbacks taking adapter locks. Never hold
        # this adapter lock while waiting for the resource admission lock.
        self._restore(cancel_event)
        with self.host_reservation.lease(cancel_event=cancel_event), self.device_reservation.lease(cancel_event=cancel_event):
            with self._lock:
                for layer in self.model.model.layers:
                    layer.mlp.experts.cancel_event = cancel_event
                return autoregressive_generate(self.model, self.tokenizer, messages, self.device,
                    cancel_event=cancel_event, max_new_tokens=max_new_tokens,
                    context_limit=self.effective_context_limit, resources=self.resources, owner=self.owner,
                    expert_headroom_bytes=self.bank.max_expert_bytes)

    def close(self):
        with self._lock:
            if self.cache is not None:
                self.cache.clear()
            self.model = self.tokenizer = self.bank = self.cache = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
            for reservation in (self.device_reservation, self.host_reservation):
                if reservation is not None:
                    reservation.release()
            self.device_reservation = self.host_reservation = None
            self.is_resident = False


def build_runtime(entry, path, resources, device='cuda:0', cancel_event=None):
    # All production model families share idle host eviction + lazy rebuilding.
    # Direct constructors remain useful for architecture-level tensor tests.
    from api.inference.reloadable import ReloadableAdapter
    if entry.id == 'medium':
        factory = GptOssAdapter
    elif entry.id == 'small':
        from api.inference.qwen import build_qwen
        factory = build_qwen
    elif entry.id == 'large':
        from api.inference.glm import build_glm
        factory = build_glm
    else:
        raise ValueError('No Kadan adapter exists for this model')
    return ReloadableAdapter(factory, entry, path, resources, device, cancel_event)
