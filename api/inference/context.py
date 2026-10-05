"""Explicit context policy and conservative, request-sized inference admission.

Estimates cover batch one BF16 caches, FP32 recurrent state and eager attention
with 32-token prefill chunks. GPU measurements remain required; no context value
is silently reduced to make a request fit. Static model/dequant budgets are held
separately by each adapter.
"""
from dataclasses import dataclass


class ContextLimitError(ValueError):
    pass


class ContextMemoryError(RuntimeError):
    pass


def _get(config, key, default=None):
    """Read a configuration key from either a mapping or an attribute-based object."""
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def text_config(config):
    """Return the nested text configuration when present, otherwise the supplied configuration."""
    return _get(config, 'text_config', None) or config


def resolve_context(config, configured=None):
    """Return (architecture maximum, effective limit). None selects the maximum; invalid or
    excessive limits raise ContextLimitError without truncation.
    """
    supported = _get(text_config(config), 'max_position_embeddings')
    if isinstance(supported, bool) or not isinstance(supported, int) or supported < 1:
        raise ContextLimitError('Checkpoint does not declare a positive architecture context limit')
    if configured is not None and (isinstance(configured, bool) or not isinstance(configured, int) or not 1 <= configured <= supported):
        raise ContextLimitError(f'Configured context must be between 1 and {supported} tokens, or null for the architecture maximum')
    return supported, supported if configured is None else configured


def configure_context(adapter, configured=None):
    """Validate a requested limit and set the adapter context metadata without allocating model or
    cache memory.
    """
    supported, effective = resolve_context(adapter.model.config, configured)
    adapter.configured_context_limit = configured
    adapter.supported_context_limit = supported
    adapter.effective_context_limit = effective


@dataclass(frozen=True)
class RequestMemory:
    cache_bytes: int
    workspace_bytes: int
    host_bytes: int

    @property
    def device_bytes(self):
        """Return estimated request cache plus workspace bytes, excluding resident model weights."""
        return self.cache_bytes + self.workspace_bytes


def estimate_request_memory(config, total_tokens, prompt_tokens):
    """Estimate batch-one cache, workspace and host bytes for actual prompt-plus-output tokens.
    Reject invalid dimensions or unsupported architectures; estimates are not measured GPU
    peaks.
    """
    c = text_config(config)
    if (any(isinstance(v, bool) or not isinstance(v, int) for v in (total_tokens, prompt_tokens))
            or not 0 < prompt_tokens <= total_tokens):
        raise ValueError('Invalid token counts for request memory estimate')
    def integer(key, default=None):
        """Read a positive integer architecture dimension or raise ContextMemoryError."""
        value = _get(c, key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ContextMemoryError(f'Cannot budget checkpoint: invalid {key}')
        return value
    layers, heads, hidden = integer('num_hidden_layers'), integer('num_attention_heads'), integer('hidden_size')
    family = _get(c, 'model_type', '')
    kinds = _get(c, 'layer_types') or ['full_attention'] * layers
    if len(kinds) != layers:
        raise ContextMemoryError('Cannot budget checkpoint: layer layout length mismatch')
    allowed_by_family = {
        'gpt_oss': {'full_attention', 'sliding_attention'},
        'glm5_next': {'linear_attention', 'indexed_attention', 'deepseek_sparse_attention'},
        'glm5_next_text': {'linear_attention', 'indexed_attention', 'deepseek_sparse_attention'},
    }
    for qwen in ('qwen3_5_moe', 'qwen3_5_moe_text', 'qwen3_5', 'qwen3_5_text'):
        allowed_by_family[qwen] = {'linear_attention', 'full_attention'}
    allowed = allowed_by_family.get(family)
    if allowed is None:
        raise ContextMemoryError(f'No request-memory estimator for architecture {family!r}')
    if any(not isinstance(kind, str) or kind not in allowed for kind in kinds):
        raise ContextMemoryError('Cannot budget checkpoint: unsupported attention layer type')
    linear = sum(kind == 'linear_attention' for kind in kinds)
    full = layers - linear
    chunk = min(32, prompt_tokens)
    recurrent = 0
    if family in ('glm5_next', 'glm5_next_text'):
        rank, rope = integer('kv_lora_rank'), _get(c, 'qk_rope_head_dim', 0)
        if isinstance(rope, bool) or not isinstance(rope, int) or rope < 0:
            raise ContextMemoryError('Cannot budget checkpoint: invalid qk_rope_head_dim')
        key_dim, value_dim = integer('qk_nope_head_dim') + rope, integer('v_head_dim')
        index_dim = integer('index_head_dim')
        index_heads = integer('index_n_heads')
        index_pool = integer('index_kpool')
        index_topk = integer('index_topk')
        index_tokens = ((total_tokens + index_pool - 1) // index_pool) * index_pool
        # Latent MLA + indexer key/score state; do not assume pool compression.
        cache = full * total_tokens * ((rank + rope) * 2 + index_dim * 4 + 16)
        linear_heads, key_head, value_head = integer('linear_num_heads'), integer('linear_head_dim'), integer('linear_head_dim')
        conv_channels = 3 * linear_heads * key_head
    elif family in ('qwen3_5_moe', 'qwen3_5_moe_text', 'qwen3_5', 'qwen3_5_text'):
        kv_heads = integer('num_key_value_heads')
        key_dim = value_dim = integer('head_dim', hidden // heads)
        cache = full * total_tokens * kv_heads * (key_dim + value_dim) * 2
        linear_heads = integer('linear_num_value_heads')
        key_head, value_head = integer('linear_key_head_dim'), integer('linear_value_head_dim')
        conv_channels = 2 * integer('linear_num_key_heads') * key_head + linear_heads * value_head
    elif family == 'gpt_oss':
        kv_heads = integer('num_key_value_heads')
        key_dim = value_dim = integer('head_dim', hidden // heads)
        # Budget full history even on sliding layers: conservative for DynamicCache.
        cache = layers * total_tokens * kv_heads * (key_dim + value_dim) * 2
    else:
        raise ContextMemoryError(f'No request-memory estimator for architecture {family!r}')
    if linear:
        conv = integer('linear_conv_kernel_dim', 4)
        recurrent = linear * (linear_heads * key_head * value_head * 4 + conv_channels * conv * 4)
    # Cache growth via concatenation can retain old and new storage together.
    cache = 2 * cache + recurrent
    # Eager attention expands grouped/latent KV for one layer, creates FP32
    # scores/probabilities and masks across the entire key history for each chunk.
    expanded = heads * total_tokens * (key_dim + value_dim) * 2
    if family.startswith('glm5'):
        # MLA expansion keeps kv_b output and separately constructed key live.
        expanded += heads * total_tokens * integer('qk_nope_head_dim') * 2
        expanded += index_dim * index_tokens * 32
        expanded += chunk * (total_tokens + index_topk) * 8  # top-k/index selection int64 workspace
    scores = heads * chunk * total_tokens * 4 * 4
    if family.startswith('glm5'):
        scores += index_heads * chunk * total_tokens * 4 * 3
    masks = chunk * total_tokens * 16
    # KDA reference chunk decay materializes channel-wise pairwise gates; even
    # short chunks are padded to64. Include overlapping intermediates explicitly.
    kda = 0
    if linear and family.startswith('glm5'):
        kda = linear_heads * 64 * 64 * key_head * 4 * 4
    activations = chunk * hidden * 4 * 32
    workspace = expanded + scores + masks + kda + activations + 256 * 1024**2
    return RequestMemory(cache, workspace, total_tokens * 32 + 1024**2)
