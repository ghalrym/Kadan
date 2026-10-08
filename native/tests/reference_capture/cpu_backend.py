"""Pinned installed HF CPU fallback; imported only after synthetic preflight."""
import inspect
import struct
from types import SimpleNamespace

import torch
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf
from transformers.integrations import hub_kernels

from api.inference.llm.qwen import build_qwen
from api.inference.llm.context import configure_context
from api.inference.resources import ResourceManager
from .artifacts import require, sha256
from .cache_contract import validate_cache
from .limits import Limits


def cpu_tensor(tensor, dtype=None):
    require(tensor.device.type == 'cpu', 'non_cpu_tensor')
    if dtype is not None:
        require(tensor.dtype == dtype, 'tensor_dtype')
    require(bool(torch.isfinite(tensor).all()), 'tensor_nonfinite')


def tensor_record(tensor):
    cpu_tensor(tensor)
    values = tensor.detach().float().reshape(-1).tolist()
    data = struct.pack('<' + 'f' * len(values), *values)
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'float32_sha256': sha256(data)}


def select_backend():
    require(not hub_kernels._kernels_enabled, 'hub_kernels_enabled')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)
    selected = {}
    # USE_HUB_KERNELS=NO alone does not disable installed FLA/causal_conv1d.
    # Select the installed original PyTorch functions, without copying equations.
    for name in ('causal_conv1d_fn', 'causal_conv1d_update',
                 'torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'):
        function = inspect.unwrap(getattr(hf, name))
        require(function.__module__ == hf.__name__ and inspect.getsourcefile(function) == hf.__file__, 'fallback_provenance')
        setattr(hf, name, function)
        selected[name] = {'module': function.__module__, 'first_line': function.__code__.co_firstlineno}
    require(torch.get_num_threads() == 2 and torch.get_num_interop_threads() == 1, 'thread_settings')
    return {'torch_version': torch.__version__, 'torch_git': torch.version.git_version,
            'torch_cuda_build': torch.version.cuda, 'effective_threads': torch.get_num_threads(),
            'effective_interop_threads': torch.get_num_interop_threads(),
            'mkldnn_enabled': torch.backends.mkldnn.enabled,
            'matmul_precision': torch.get_float32_matmul_precision(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'torch_build': torch.__config__.show(), 'parallel_info': torch.__config__.parallel_info(),
            'selected_fallbacks': selected, 'device': 'cpu', 'hub_kernels': False}


def forward(root, token, limits=Limits()):
    report = select_backend()
    resources = ResourceManager(limits.host_bytes, {})
    adapter = None
    output = None
    hooks = []
    request = None
    layers, routes = [], []
    calls = {'model': 0, 'chunk': 0, 'recurrent': 0, 'conv': 0, 'conv_update': 0}
    originals = {}
    # Test-only call counters preserve the selected installed functions unchanged.
    for name, key in (('torch_chunk_gated_delta_rule','chunk'), ('torch_recurrent_gated_delta_rule','recurrent'),
                      ('causal_conv1d_fn','conv'), ('causal_conv1d_update','conv_update')):
        original = getattr(hf, name)
        originals[name] = original
        def counted(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)
        setattr(hf, name, counted)
    try:
        adapter = build_qwen(SimpleNamespace(id=limits.owner), root, resources, 'cpu')
        configure_context(adapter, 1)
        require(adapter.model.config._attn_implementation == 'eager' and not adapter.model.training, 'model_backend')
        require(not getattr(adapter.model, '_use_kernels', False), 'kernelized_model')
        for name, parameter in adapter.model.named_parameters():
            cpu_tensor(parameter, torch.float32 if name.endswith(('.A_log', '.dt_bias')) else torch.bfloat16)
        for tensor in adapter.model.buffers():
            cpu_tensor(tensor)
        for index, layer in enumerate(adapter.model.model.layers):
            def layer_hook(module, args, result, index=index):
                cpu_tensor(result, torch.bfloat16)
                layers.append({'layer': index, **tensor_record(result)})
            def route_hook(module, args, result, index=index):
                logits, weights, ids = result
                cpu_tensor(logits, torch.bfloat16);cpu_tensor(weights, torch.bfloat16)
                require(ids.device.type == 'cpu', 'router_device')
                probabilities = torch.softmax(logits.float(), dim=-1)[0]
                ordered = sorted(probabilities.tolist(), reverse=True)
                k = module.top_k
                routes.append({'layer': index, 'ids': ids[0].tolist(), 'weights': weights[0].tolist(),
                               'logits': logits[0].tolist(), 'probabilities': probabilities.tolist(),
                               'boundary_margin': ordered[k-1] - ordered[k] if k < len(ordered) else None,
                               'has_probability_ties': len(set(ordered)) != len(ordered),
                               'boundary_tie': k < len(ordered) and ordered[k-1] == ordered[k]})
            hooks.extend([layer.register_forward_hook(layer_hook), layer.mlp.gate.register_forward_hook(route_hook)])
        request = resources.reserve(limits.owner + ':request', 'llm', host_bytes=limits.request_bytes)
        with adapter.reservation.lease(), request.lease(), torch.inference_mode():
            calls['model'] += 1
            output = adapter.model(input_ids=torch.tensor([[token]], dtype=torch.long),
                                   attention_mask=torch.ones((1,1), dtype=torch.long),
                                   position_ids=torch.zeros((1,1), dtype=torch.long),
                                   use_cache=True, return_dict=True, logits_to_keep=1)
            cpu_tensor(output.logits, torch.bfloat16)
            require(list(output.logits.shape) == [1,1,adapter.model.config.vocab_size], 'logit_shape')
            cache = output.past_key_values
            cache_records = []
            for i, (kind, tensors) in enumerate(validate_cache(cache, adapter.model.config)):
                for tensor in tensors:cpu_tensor(tensor)
                cache_records.append({'layer': i, 'kind': kind, 'tensors': [tensor_record(t) for t in tensors]})
            logits = output.logits[0,0].float().tolist()
            linear = adapter.model.config.layer_types.count('linear_attention')
            require(calls == {'model':1,'chunk':linear,'recurrent':0,'conv':linear,'conv_update':0}, 'forward_path')
            require(len(layers) == len(routes) == len(adapter.model.model.layers), 'diagnostic_layer_count')
            report.update(calls=calls, layers=layers, routes=routes, cache=cache_records,
                          cache_length=1, final_dtype='torch.bfloat16', capacity=1,
                          final_float32_is_container_only=True,
                          layer_hash_scope='reference-only; cannot localize native divergence',
                          admission=resources.snapshot())
            del cache
            output = None
    finally:
        output = None
        for hook in hooks:hook.remove()
        for name, function in originals.items():setattr(hf, name, function)
        if adapter is not None:adapter.close()
        if request is not None:request.release()
    require(not resources.snapshot()['reservations'], 'cleanup_reservations')
    report['zero_reservations'] = True
    return logits, report
