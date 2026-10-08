"""Exact fresh one-token cache structure; no numerical-package dependency."""
from .artifacts import require


def validate_cache(cache, config):
    require(cache is not None and cache.get_seq_length() == 1, 'cache_length')
    require(len(config.layer_types) == config.num_hidden_layers == len(cache.layers), 'cache_layer_count')
    conv_channels = (2 * config.linear_num_key_heads * config.linear_key_head_dim
                     + config.linear_num_value_heads * config.linear_value_head_dim)
    expected = {
        'conv': ([1, conv_channels, config.linear_conv_kernel_dim], 'torch.bfloat16'),
        'recurrent': ([1, config.linear_num_value_heads, config.linear_key_head_dim,
                       config.linear_value_head_dim], 'torch.float32'),
        'key': ([1, config.num_key_value_heads, 1, config.head_dim], 'torch.bfloat16'),
        'value': ([1, config.num_key_value_heads, 1, config.head_dim], 'torch.bfloat16'),
    }
    result = []
    for index, kind in enumerate(config.layer_types):
        state = cache.layers[index]  # Count checked first: no zip truncation.
        if kind == 'linear_attention':
            for name in ('conv_states', 'recurrent_states'):
                slots = getattr(state, name, None)
                require(isinstance(slots, dict) and len(slots) == 1
                        and all(type(key) is int and key == 0 for key in slots), 'cache_state_slots:' + name)
            tensors = [('conv', state.conv_states[0]), ('recurrent', state.recurrent_states[0])]
        else:
            require(kind == 'full_attention', 'cache_layer_kind')
            tensors = [('key', getattr(state, 'keys', None)), ('value', getattr(state, 'values', None))]
        for name, tensor in tensors:
            shape, dtype = expected[name]
            require(list(getattr(tensor, 'shape', ())) == shape, 'cache_shape:' + name)
            require(str(getattr(tensor, 'dtype', None)) == dtype, 'cache_dtype:' + name)
        result.append((kind, [tensor for _, tensor in tensors]))
    return result
