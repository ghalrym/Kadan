"""Experimental QwenImage21 cached-block sequence parallelism.

Not selected by the API. The caller owns two identical block replicas, compact
per-request prefix caches, rank coordination, bounded process-group lifetime and
both device reservations. Prefill remains eager in this foundation.
"""
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as functional
from diffusers.models.transformers.transformer_qwenimage21 import apply_rotary_emb_qwen


@dataclass(frozen=True)
class TokenShard:
    total: int
    rank: int
    world: int

    def __post_init__(self):
        if self.total < 1 or self.world < 1 or not 0 <= self.rank < self.world:
            raise ValueError('Invalid token partition')

    @property
    def width(self):
        return (self.total + self.world - 1) // self.world

    @property
    def start(self):
        return self.rank * self.width

    @property
    def count(self):
        return max(0, min(self.width, self.total - self.start))

    def take(self, value, dim=1):
        if value.shape[dim] != self.total:
            raise ValueError("Partition must slice the complete global target sequence")
        selected = value.narrow(dim, min(self.start, self.total), self.count)
        shape = list(value.shape)
        shape[dim] = self.width - self.count
        return torch.cat((selected, value.new_zeros(shape)), dim=dim).contiguous()


def sequence_to_heads(values, group=None):
    """One fused Q/K/V all-to-all: local rows/all heads → all rows/local heads."""
    world = dist.get_world_size(group)
    stacked = torch.stack(values)
    tensors, batch, rows, heads, depth = stacked.shape
    if heads % world:
        raise ValueError('Attention heads must divide the rank count')
    local_heads = heads // world
    send = stacked.reshape(tensors, batch, rows, world, local_heads, depth).permute(3, 0, 1, 2, 4, 5).contiguous()
    receive = torch.empty_like(send)
    dist.all_to_all_single(receive, send, group=group)
    result = receive.permute(1, 2, 0, 3, 4, 5).reshape(tensors, batch, world * rows, local_heads, depth)
    return result.unbind(0)


def heads_to_sequence(value, group=None):
    """Inverse exchange restores rank-local row order and the original head order."""
    world = dist.get_world_size(group)
    batch, rows, heads, depth = value.shape
    if rows % world:
        raise ValueError('Exchanged sequence must include equal-rank padding')
    local_rows = rows // world
    send = value.reshape(batch, world, local_rows, heads, depth).permute(1, 0, 2, 3, 4).contiguous()
    receive = torch.empty_like(send)
    dist.all_to_all_single(receive, send, group=group)
    return receive.permute(1, 2, 0, 3, 4).reshape(batch, local_rows, world * heads, depth)


def gather_target_kv(key, value, group=None):
    """Correctness reference: gather target rows only, never replicated prefix rows."""
    packed = torch.stack((key, value)).contiguous()
    received = [torch.empty_like(packed) for _ in range(dist.get_world_size(group))]
    dist.all_gather(received, packed, group=group)
    return torch.cat(received, dim=2).unbind(0)


def compact_prefix(key, value, mode, rank, world):
    """Create request-owned prefix storage from eager, post-RoPE extraction.

    Ulysses owns only this rank's head shard. The all-gather reference owns all
    heads. Clone outside compiled code; this module never enables compilation.
    """
    if mode not in ('ulysses', 'all_gather') or key.ndim != 4 or key.shape != value.shape:
        raise ValueError('Invalid prefix mode or shape')
    if world < 1 or key.shape[2] % world or not 0 <= rank < world:
        raise ValueError('Invalid prefix head partition')
    if mode == 'ulysses':
        width = key.shape[2] // world
        key, value = key[:, :, rank * width:(rank + 1) * width], value[:, :, rank * width:(rank + 1) * width]
    return key.clone().contiguous(), value.clone().contiguous()


@torch.no_grad()
def cached_block(block, hidden, modulation, rotary, prefix, shard, *,
                 mode='ulysses', key_valid=None, group=None):
    """Exact cached target-row block; both ranks execute local QKV and MLP GEMMs.

    hidden/rotary are already partitioned using original global target positions.
    key_valid covers prefix plus UNPADDED global target rows. Padded rows never
    become attention keys; the caller drops their final outputs. Prefix RoPE and
    t=0 modulation come from eager extraction, not recomputation at target time.
    """
    if mode not in ('ulysses', 'all_gather'):
        raise ValueError('Unsupported sequence parallel mode')
    if dist.get_world_size(group) != shard.world or dist.get_rank(group) != shard.rank:
        raise ValueError('Partition and process group disagree')
    if hidden.ndim != 3 or hidden.shape[1] != shard.width or rotary.shape[0] != shard.width:
        raise ValueError('Local sequence or global-position rotary slice is invalid')
    if block.attn.heads % shard.world:
        raise ValueError('Attention heads must divide the rank count')
    prefix_key, prefix_value = prefix
    expected_heads = block.attn.heads // shard.world if mode == 'ulysses' else block.attn.heads
    if prefix_key.shape != prefix_value.shape or prefix_key.shape[0] != hidden.shape[0] or prefix_key.shape[2] != expected_heads:
        raise ValueError('Prefix must contain this mode’s head ownership exactly once')
    for value in prefix:
        if value.storage_offset() or value.untyped_storage().nbytes() != value.numel() * value.element_size():
            raise ValueError('Prefix must own compact storage')
    first, second = modulation.chunk(2, dim=-1)
    target = torch.ones(shard.width, dtype=torch.bool, device=hidden.device)
    normalized, gate = block._modulate(block.img_norm1(hidden), first, target)
    attention = block.attn
    query = attention.to_q(normalized).unflatten(-1, (attention.heads, -1))
    key = attention.to_k(normalized).unflatten(-1, (attention.heads, -1))
    value = attention.to_v(normalized).unflatten(-1, (attention.heads, -1))
    query = apply_rotary_emb_qwen(attention.norm_q(query).to(value.dtype), rotary, use_real=False)
    key = apply_rotary_emb_qwen(attention.norm_k(key).to(value.dtype), rotary, use_real=False)
    if mode == 'ulysses':
        query, key, value = sequence_to_heads((query, key, value), group)
    else:
        key, value = gather_target_kv(key, value, group)
    key = torch.cat((prefix_key, key), dim=1)
    value = torch.cat((prefix_value, value), dim=1)
    prefix_rows = prefix_key.shape[1]
    padding = shard.world * shard.width - shard.total
    mask = None
    # Preserve no-mask Flash SDPA for the common all-valid/even cached case.
    # A gratuitous all-True mask can select a different CUDA backend.
    if key_valid is not None or padding:
        if key_valid is None:
            key_valid = torch.ones((hidden.shape[0], prefix_rows + shard.total), dtype=torch.bool, device=hidden.device)
        if key_valid.dtype != torch.bool or key_valid.shape != (hidden.shape[0], prefix_rows + shard.total):
            raise ValueError('Key validity must cover the prefix and unpadded global targets')
        mask = functional.pad(key_valid, (0, padding), value=False)[:, None, None, :]
    result = functional.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2),
        value.transpose(1, 2), attn_mask=mask, dropout_p=0.0, is_causal=False).transpose(1, 2)
    if mode == 'ulysses':
        result = heads_to_sequence(result.contiguous(), group)
    result = attention.to_out[1](attention.to_out[0](result.flatten(2, 3).contiguous()))
    hidden = hidden + gate.tanh() * result
    normalized, gate = block._modulate(block.img_norm2(hidden), second, target)
    hidden = hidden + gate.tanh() * block.img_mlp(normalized)
    return hidden.clip(-65504, 65504) if hidden.dtype == torch.float16 else hidden
