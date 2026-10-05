"""Kadan GLM-5.3 text adapter: host-packed experts and demand-driven CUDA cache.

Uses Apache-2.0 Transformers glm5_next layer definitions, never FreeToken runtime.
Checkpoint renames follow Transformers conversion_mapping.py (469230357aab).
Text-only greedy decoding excludes the vision tower and MTP auxiliary layer.
W4A16 reference execution deliberately does not emulate calibrated W4A4 activations.
"""
import json
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from uuid import uuid4

import torch
from torch import nn
from torch.nn import functional as F

from .checkpoint import SafeTensorReader
from .generation import autoregressive_generate, check_cancel
from .offload import ExpertBank, ExpertCache
from .quantization import nvfp4_linear
from .resources import ResourceBusy


PREFIX = 'model.language_model.'
FP32_PARTS = ('forget_gate.A_log', 'forget_gate.dt_bias', '_hc.', 'conv1d',
              'e_score_correction_bias', 'index_kpool_compress_ape')


def dense_dtype(name):
    """Keep named sensitive gate, convolution and mixing tensors FP32; use BF16 otherwise."""
    return torch.float32 if any(part in name for part in FP32_PARTS) else torch.bfloat16


def source_names(name):
    """Return original checkpoint keys for one Transformers text-model tensor.

    Undo mHC and forget-gate renames. A fused convolution maps to three keys
    in q/k/v concatenation order; the language head keeps its root key.
    Reject parameters outside the supported text-model namespace."""
    if name == 'lm_head.weight':
        return (name,)
    if not name.startswith('model.'):
        raise ValueError(f'Unexpected GLM parameter {name}')
    name = PREFIX + name[len('model.'):]
    for destination, source in (('attn_hc.', 'hc_attn_'), ('ffn_hc.', 'hc_ffn_')):
        name = name.replace(destination, source)
    name = name.replace('self_attn.forget_gate.', 'self_attn.')
    if name.endswith('self_attn.conv1d.weight'):
        return tuple(name.replace('conv1d.weight', f'{part}_conv1d.weight') for part in ('q', 'k', 'v'))
    return (name,)


def read_dense(reader, names, dtype):
    """Read CPU nonexpert weights, cast them, and concatenate multiple keys by row.

    Accept BF16, FP16 and FP32 tensors, or 2-D E4M3FN weights with explicit
    128x128 weight_scale_inv blocks. FP8 arithmetic expands scales in FP32
    before casting. Unsupported dtypes or missing/mismatched scales raise
    ValueError; this may allocate a full dense host projection."""
    tensors = []
    for name in names:
        weight = reader.tensor(name)
        if weight.dtype == torch.float8_e4m3fn:
            scale_name = name.removesuffix('.weight') + '.weight_scale_inv'
            if weight.ndim != 2 or scale_name not in reader.keys:
                raise ValueError(f'Unsupported GLM FP8 layout: {name}; missing block scales')
            scale = reader.tensor(scale_name).float()
            expected = ((weight.shape[0] + 127) // 128, (weight.shape[1] + 127) // 128)
            if tuple(scale.shape) != expected:
                raise ValueError(f'Unexpected GLM FP8 block scale shape: {name}')
            weight = weight.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:weight.shape[0], :weight.shape[1]]
        elif weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f'Unsupported GLM nonexpert dtype: {name}: {weight.dtype}')
        tensors.append(weight.to(dtype))
    return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)


def validate_expert(tensors, hidden, intermediate):
    """Check the three packed projections of one compressed-tensors GLM expert.

    Gate/up matrices are [intermediate, hidden]; down is [hidden, intermediate].
    Accept eight FP4 values per int32 or two per uint8, one E4M3FN scale per
    16 columns, and finite positive scalar/per-row global quant scales.
    Return None on success; malformed dimensions, dtypes or globals raise."""
    for part, shape in (('gate_proj', (intermediate, hidden)),
                        ('up_proj', (intermediate, hidden)),
                        ('down_proj', (hidden, intermediate))):
        packed = tensors[f'{part}.weight_packed']
        factor = 8 if packed.dtype == torch.int32 else 2
        if packed.dtype not in (torch.int32, torch.uint8) or tuple(packed.shape) != (shape[0], shape[1] // factor):
            raise ValueError(f'Invalid GLM packed expert shape/dtype: {part}')
        scale = tensors[f'{part}.weight_scale']
        if tuple(scale.shape) != (shape[0], shape[1] // 16) or scale.dtype not in (torch.float8_e4m3fn, torch.uint8):
            raise ValueError(f'Invalid GLM expert block scales: {part}')
        global_scale = tensors[f'{part}.weight_global_scale']
        if global_scale.numel() != 1 and tuple(global_scale.shape) not in ((shape[0],), (shape[0], 1)):
            raise ValueError(f'Invalid GLM expert global scale shape: {part}')
        if not torch.isfinite(global_scale).all() or (global_scale <= 0).any():
            raise ValueError(f'Invalid GLM expert global scale value: {part}')


class GlmExperts(nn.Module):
    """Replace only routed expert matmuls; native GLM router/shared expert remain."""
    def __init__(self, cache, layer, limit):
        """Bind a layer's cache and SwiGLU clamp limit without allocating expert weights."""
        super().__init__()
        self.cache, self.layer, self.limit = cache, layer, limit
        self.cancel_event = None

    def forward(self, hidden, indices, weights):
        """Combine routed experts for hidden [tokens, hidden_size] using top-k routing.

        Indices and routing weights are [tokens, top_k]. Lease one packed expert
        at a time, apply GLM's clamped SwiGLU, then accumulate weighted outputs
        in the input dtype. Cancellation is checked between experts; admission
        and decoder errors propagate. Shared experts remain in the native layer."""
        result = torch.zeros_like(hidden)
        for expert in indices.unique().tolist():
            check_cancel(self.cancel_event)
            token, slot = torch.where(indices == expert)
            with self.cache.use((self.layer, expert)) as tensors:
                x = hidden[token]
                def linear(value, part):
                    """Apply a leased gate/up/down projection with reciprocal global-scale semantics."""
                    return nvfp4_linear(value, tensors[f'{part}.weight_packed'],
                                        tensors[f'{part}.weight_scale'],
                                        tensors[f'{part}.weight_global_scale'],
                                        layout='compressed-tensors')
                gate = linear(x, 'gate_proj').clamp(max=self.limit)
                up = linear(x, 'up_proj').clamp(-self.limit, self.limit)
                out = linear(F.silu(gate) * up, 'down_proj')
                result.index_add_(0, token, (out * weights[token, slot, None]).to(hidden.dtype))
        return result


class TextLM(nn.Module):
    """Expose a text decoder and language head through the owned generation-loop interface."""
    def __init__(self, model, config):
        """Attach the supplied decoder and create its untied, bias-free vocabulary projection."""
        super().__init__()
        self.model, self.config = model, config
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, past_key_values=None, use_cache=True, logits_to_keep=1, **kwargs):
        """Return trailing-token logits and the decoder's updated attention cache.

        Forward incoming cached KDA/DSA state to the native text decoder; project
        only the last logits_to_keep hidden states. Vision and MTP paths are absent.
        Extra generation-loop keywords are accepted but not forwarded."""
        output = self.model(input_ids=input_ids, past_key_values=past_key_values,
                            use_cache=use_cache, return_dict=True)
        return SimpleNamespace(logits=self.lm_head(output.last_hidden_state[:, -logits_to_keep:]),
                               past_key_values=output.past_key_values)


class HostLinear(nn.Module):
    """Ordinary GLM projections live in host RAM and execute bounded row tiles."""
    def __init__(self, weight, keys, cache, cancellation):
        """Retain detached CPU weights and ordered tile keys without registering GPU parameters.

        The plain weight attribute also supplies the dtype expected by native GLM
        layers. Cancellation is a callable returning the current request event."""
        super().__init__()
        self.weight = weight.detach()  # ordinary tensor: Module.to cannot move it
        self.keys, self.cache, self.cancellation = keys, cache, cancellation

    def forward(self, x):
        """Return [..., output_rows] by leasing CPU-backed weight tiles onto the input device.

        Each tile lease ends after its matmul; the entry may remain cached until
        eviction. Cancellation and cache admission failures propagate. Output storage is allocated separately from cache tiles.
        Module.to() does not relocate the retained full host weight."""
        result = torch.empty((*x.shape[:-1], self.weight.shape[0]), dtype=x.dtype, device=x.device)
        row = 0
        for key in self.keys:
            check_cancel(self.cancellation())
            with self.cache.use(key) as part:
                size = part['weight'].shape[0]
                result[..., row:row + size] = F.linear(x, part['weight'].to(x.dtype), part.get('bias'))
                row += size
        return result


class GlmAdapter:
    """Own GLM host storage, GPU residency and serialized generation lifecycle.

    GPU eviction retains host banks for restoration. Explicit close releases
    both tiers; request token/KV admission is delegated to the generation loop."""
    def __init__(self):
        """Initialize an empty ownership handle; build_glm supplies model, device and reservations."""
        self._lock = RLock()
        self.model = self.cache = self.bank = self.reservation = None
        self.device_reservation = None
        self._cancel = None

    @property
    def is_resident(self):
        """Report whether a model and device reservation exist, without probing CUDA health."""
        return self.model is not None and self.device_reservation is not None

    def _restore(self, event=None):
        """Admit resident GPU tensors and move registered model state to the selected device.

        Packed experts and plain host projection weights stay in RAM. A closed
        model raises RuntimeError; admission/copy errors propagate after rollback."""
        if self.is_resident:
            return
        if self.model is None:
            raise RuntimeError('GLM runtime is closed')
        self.device_reservation = self.resources.reserve(
            self.owner + ':device', 'llm', device_bytes={self.device.index or 0: self.gpu_bytes},
            evict=self._offload, cancel_event=event)
        try:
            with self.device_reservation.lease(event):
                self.model.to(self.device)
        except BaseException:
            self._offload()
            raise

    def _offload(self):
        """Evict GPU cache entries and move registered state to RAM before releasing VRAM.

        Use a nonblocking adapter lock because resource admission invokes this
        callback while holding its own lock. Busy adapters raise ResourceBusy;
        host banks and their reservation survive for a later restore."""
        if not self._lock.acquire(blocking=False):
            raise ResourceBusy('GLM is busy; cannot evict during restore/generation')
        try:
            if self.cache is not None:
                self.cache.clear()
            if self.model is not None:
                self.model.to('cpu')
            self._sync()
            if self.device_reservation is not None:
                self.device_reservation.release()
                self.device_reservation = None
        finally:
            self._lock.release()

    def _sync(self):
        """Finish selected-device CUDA work and release unused allocator blocks; CPU is a no-op."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    def generate(self, messages, max_new_tokens=256, cancel_event=None):
        """Return greedy response text while holding host and device ownership leases.

        Restore GPU residency if necessary and propagate cancellation to tiled and
        expert operations. The shared loop validates the effective token limit and
        admits request memory; limit/admission failures do not silently shorten input."""
        with self._lock:
            with self.reservation.lease(cancel_event):
                self._restore(cancel_event)
                with self.device_reservation.lease(cancel_event):
                    self._cancel = cancel_event
                    for layer in self.model.model.layers:
                        if isinstance(getattr(layer.mlp, 'experts', None), GlmExperts):
                            layer.mlp.experts.cancel_event = cancel_event
                    try:
                        return autoregressive_generate(self.model, self.tokenizer, messages, self.device,
                                                       cancel_event, max_new_tokens, context_limit=self.effective_context_limit,
                                                       resources=self.resources, owner=self.owner,
                                                       expert_headroom_bytes=self.bank.max_expert_bytes)
                    finally:
                        self._cancel = None

    def close(self):
        """Drop model/cache/bank references, synchronize CUDA, then release both reservations.

        Serialize against generation. Cleanup errors propagate so callers can retain
        this handle and retry instead of treating its memory as already released."""
        with self._lock:
            if self.cache is not None:
                self.cache.close()
            self.model = self.cache = self.bank = None
            self._sync()
            if self.device_reservation is not None:
                self.device_reservation.release()
                self.device_reservation = None
            if self.reservation is not None:
                self.reservation.release()
                self.reservation = None


def build_glm(entry, path, resources, device='cuda:0', cancel_event=None):
    """Build a locally loaded GLM text adapter with Kadan-owned memory and execution.

    Validate checkpoint names, shapes and compressed-tensors packing before
    publishing the adapter. Construct a meta skeleton, retain packed experts and
    dense projection tiles in host RAM, and admit resident state on one CUDA
    GPU. Vision/MTP and activation calibration tensors are explicitly excluded.

    Return a ready adapter; unsupported checkpoints, missing dependencies,
    cancellation or admission failures raise with partial ownership cleaned up.
    Resident tensors plus 16 MiB decode scratch are reserved separately from
    per-entry cache and request KV/workspace. CPU tests cover components only;
    full checkpoint/GPU parity and peak memory remain unvalidated."""
    check_cancel(cancel_event)
    try:
        from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel
    except ImportError as exc:
        raise RuntimeError('GLM requires the pinned Transformers source with glm5_next; released 5.16.0 lacks this architecture') from exc
    from accelerate import init_empty_weights
    from transformers import AutoTokenizer
    from accelerate.utils import set_module_tensor_to_device

    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('GLM production loading requires one CUDA device; CPU tests exercise components only')
    raw = json.loads((Path(path) / 'config.json').read_text())
    if raw.get('model_type') != 'glm5_next' or raw.get('quantization_config', {}).get('quant_method') != 'compressed-tensors':
        raise ValueError('Expected GLM glm5_next compressed-tensors checkpoint')
    text = dict(raw['text_config'])
    text.pop('model_type', None)
    text['layer_types'] = ['indexed_attention' if x == 'deepseek_sparse_attention' else x for x in text['layer_types']]
    if set(text['layer_types']) - {'indexed_attention', 'linear_attention'}:
        raise ValueError('Unknown GLM attention architecture')
    config = Glm5NextTextConfig(**text)
    config._attn_implementation = 'eager'
    reader = SafeTensorReader(path)
    with init_empty_weights(include_buffers=True):
        model = TextLM(Glm5NextTextModel(config), config)
    # Remove giant meta expert parameters before dense tensor enumeration.
    sparse = [i for i, kind in enumerate(config.mlp_layer_types) if kind == 'sparse']
    for layer in sparse:
        model.model.layers[layer].mlp.experts = nn.Identity()
    persistent = set(model.state_dict())
    target = list(model.named_parameters()) + [(n, b) for n, b in model.named_buffers() if n in persistent]
    dense_bytes = sum(p.numel() * (4 if dense_dtype(name) == torch.float32 else 2) for name, p in target)
    keys = []
    for layer in sparse:
        for expert in range(config.num_local_experts):
            for part in ('gate_proj', 'up_proj', 'down_proj'):
                for kind in ('weight_packed', 'weight_scale', 'weight_global_scale'):
                    keys.append(f'{PREFIX}layers.{layer}.mlp.experts.{expert}.{part}.{kind}')
    required = set(keys)
    for name, _ in target:
        required.update(source_names(name))
    missing = required - reader.keys
    if missing:
        raise ValueError(f'GLM checkpoint lacks required tensors: {sorted(missing)[:3]}')
    host_bytes = sum(reader.nbytes(name) for name in required) + dense_bytes + 2 * 1024**3
    # Every text-decoder key must be understood. Vision and post-decoder MTP
    # are the only excluded architectures; activation calibration is unused in W4A16.
    import re
    for key in reader.keys:
        if not key.startswith(PREFIX):
            if key == 'lm_head.weight' or key.startswith('model.visual.') or key.startswith('mtp.'):
                continue
            raise ValueError(f'Unrecognized GLM checkpoint tensor: {key}')
        match = re.match(r'model\.language_model\.layers\.(\d+)\.', key)
        if match and int(match.group(1)) >= config.num_hidden_layers:
            continue  # auxiliary MTP decoder; not part of autoregressive text model
        if key in required:
            continue
        if key.endswith('.input_global_scale') and key.removesuffix('.input_global_scale') + '.weight_packed' in required:
            continue
        if key.endswith('.weight_scale_inv') and key.removesuffix('.weight_scale_inv') + '.weight' in required:
            continue
        raise ValueError(f'Unrecognized GLM text-decoder tensor: {key}')
    adapter = GlmAdapter()
    adapter.device = device
    adapter.resources = resources
    adapter.owner = f'glm:{uuid4()}'
    experts = {}
    value = None
    try:
        adapter.reservation = resources.reserve(
            adapter.owner + ':host', 'llm', host_bytes=host_bytes, cancel_event=cancel_event,
        )
        with adapter.reservation.lease(cancel_event):
            experts = {}
            for layer in sparse:
                for expert in range(config.num_local_experts):
                    check_cancel(cancel_event)
                    prefix = f'{PREFIX}layers.{layer}.mlp.experts.{expert}.'
                    experts[(layer, expert)] = {
                        f'{part}.{kind}': reader.tensor(prefix + f'{part}.{kind}')
                        for part in ('gate_proj', 'up_proj', 'down_proj')
                        for kind in ('weight_packed', 'weight_scale', 'weight_global_scale')
                    }
                    validate_expert(experts[(layer, expert)], config.hidden_size, config.moe_intermediate_size)
            for name, parameter in target:
                check_cancel(cancel_event)
                # KDA gates, mHC, convolution, router correction and indexer APE
                # preserve FP32 arithmetic; ordinary matrix multiplies use BF16.
                dtype = dense_dtype(name)
                value = read_dense(reader, source_names(name), dtype)
                if tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(f'GLM tensor shape mismatch for {name}: {tuple(value.shape)} != {tuple(parameter.shape)}')
                set_module_tensor_to_device(model, name, 'cpu', value=value, dtype=dtype)
            if any(p.is_meta for p in model.parameters()) or any(b.is_meta for b in model.buffers()):
                raise ValueError('GLM has unloaded tensors')
            linears = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
            tiles = {}
            for name, module in linears:
                tile_keys = []
                for start in range(0, module.out_features, 1024):
                    key = ('dense', name, start)
                    tile_keys.append(key)
                    experts[key] = {'weight': module.weight.detach()[start:start + 1024]}
                    if module.bias is not None:
                        experts[key]['bias'] = module.bias.detach()[start:start + 1024]
                tiles[name] = tile_keys
            adapter.bank = ExpertBank(experts)
            adapter.cache = ExpertCache(adapter.bank, resources.capacity.device_bytes[device.index or 0],
                                        device=device, resources=resources, owner=adapter.owner + ':experts')
            for name, module in linears:
                parent_name, _, leaf = name.rpartition('.')
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, leaf, HostLinear(module.weight, tiles[name], adapter.cache, lambda: adapter._cancel))
            for layer in sparse:
                model.model.layers[layer].mlp.experts = GlmExperts(adapter.cache, layer, config.swiglu_limit)
            adapter.gpu_bytes = sum(p.numel() * p.element_size() for p in list(model.parameters()) + list(model.buffers())) + 16 * 1024**2
            adapter.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            adapter.model = model.eval()
            from api.inference.context import configure_context
            configure_context(adapter, None)
            adapter._restore(cancel_event)
            check_cancel(cancel_event)
        return adapter
    except BaseException:
        model = None
        value = None
        experts.clear()
        if 'linears' in locals():
            linears.clear()
            module = None
            parent = None
        adapter.close()
        raise
    finally:
        reader.close()
