"""Kadan reference FP4 weight decoders (W4A32/W4A16, no native FP4 kernel).

Layout references, inspected at FreeToken d3512b43 (Apache-2.0):
https://github.com/FlashML-org/FreeToken/blob/d3512b43affe981465e03ee28cbd88f49c39b9aa/python/freetoken/kernel/triton/nvfp4_dequant.py
https://github.com/FlashML-org/FreeToken/blob/d3512b43affe981465e03ee28cbd88f49c39b9aa/python/freetoken/models/gpt_oss/weight.py
These are independently implemented format arithmetic, not an upstream dependency.
Only canonical, unswizzled checkpoint layouts are accepted. ModelOpt global scales
multiply; compressed-tensors global scales divide. Activation quantization is not
emulated. Decoding materializes dense scratch; callers must budget it separately.
"""
import torch
from torch import Tensor
from torch.nn import functional as F


def _unpack(packed: Tensor) -> Tensor:
    """Expand uint8 FP4 bytes into FP32 signed E2M1 values on the same device.

    The low nibble precedes the high nibble, doubling the final dimension.
    Reject non-byte input; this decodes canonical packing, not kernel swizzles."""
    if packed.dtype != torch.uint8:
        raise ValueError('Canonical FP4 bytes must have dtype uint8.')
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    magnitudes = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=packed.device, dtype=torch.float32)
    return magnitudes[codes & 7] * torch.where(codes < 8, 1., -1.)


def dequantize_mxfp4(blocks: Tensor, scales: Tensor, dtype=torch.float32) -> Tensor:
    """Decode GPT-OSS MXFP4 blocks into a dense tensor of the requested dtype.

    Packed bytes have shape [..., blocks, 16]; uint8 E8M0 scales have shape
    [..., blocks]. Each scale multiplies 32 decoded values, producing [..., K].
    Scale byte 255 becomes NaN. Invalid shapes, scale dtype or devices raise
    ValueError; the entire decoded result is allocated on the input device."""
    if blocks.ndim < 2 or blocks.shape[-1] != 16 or scales.shape != blocks.shape[:-1]:
        raise ValueError('MXFP4 requires 16 packed bytes and one scale per 32-value block.')
    if scales.dtype != torch.uint8 or scales.device != blocks.device:
        raise ValueError('MXFP4 scales must be E8M0 uint8 on the weight device.')
    # 255 is the E8M0 NaN encoding, not the ordinary exponent +128.
    exponent = scales.to(torch.int32) - 127
    factor = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)
    factor = torch.where(scales == 255, float('nan'), factor)
    return (_unpack(blocks) * factor.unsqueeze(-1)).flatten(-2).to(dtype)


def dequantize_nvfp4(weight: Tensor, scale: Tensor, global_scale: Tensor,
                     dtype=torch.float32, *, layout='modelopt') -> Tensor:
    """Decode canonical NVFP4 weights into dense rows on the weight device.

    ModelOpt stores uint8 [..., K/2] bytes with a global dequant multiplier.
    Compressed-tensors also accepts int32 [..., K/8] packing and divides by its
    global quant multiplier. Each E4M3FN block scale covers 16 values; global
    scales may be scalar or per row. Return the requested dtype and reject
    unsupported layouts/shapes/devices. Activation quantization is not applied."""
    if layout not in ('modelopt', 'compressed-tensors'):
        raise ValueError('Unknown NVFP4 layout; kernel-swizzled weights are unsupported.')
    if layout == 'compressed-tensors' and weight.dtype == torch.int32:
        # Explicit shifts avoid relying on CPU/GPU endianness or byte views.
        weight = torch.stack(tuple((weight >> shift) & 255 for shift in (0, 8, 16, 24)), -1).flatten(-2).to(torch.uint8)
    if weight.ndim < 2 or weight.shape[-1] % 8 or scale.shape != (*weight.shape[:-1], weight.shape[-1] // 8):
        raise ValueError('NVFP4 requires one scale per 16 values in a canonical row-major layout.')
    if scale.device != weight.device or global_scale.device != weight.device:
        raise ValueError('NVFP4 tensors must be on the same device.')
    if scale.dtype == torch.uint8:
        scale = scale.view(torch.float8_e4m3fn)
    if scale.dtype != torch.float8_e4m3fn:
        raise ValueError('NVFP4 block scales must be E4M3FN or its raw uint8 bytes.')
    global_scale = global_scale.to(torch.float32)
    if global_scale.numel() == 1:
        global_scale = global_scale.reshape(())
    else:
        if global_scale.shape == weight.shape[:-1]:
            global_scale = global_scale.unsqueeze(-1)
        elif global_scale.shape != (*weight.shape[:-1], 1):
            raise ValueError('Global scale must be scalar or one value per output row.')
    if layout == 'compressed-tensors':
        global_scale = global_scale.reciprocal()
    return (_unpack(weight) * scale.float().repeat_interleave(16, -1) * global_scale).to(dtype)


def _tiled_linear(inputs, rows, columns, decode_rows, bias, scratch_bytes):
    """Apply a matrix supplied by decode_rows(start, end) one output tile at a time.

    Return [..., rows] in the input dtype/device. Tile sizing uses a conservative
    64-byte-per-weight scratch estimate; output and activation storage are not
    included. Raise ValueError for incompatible input columns/dtype, or
    MemoryError when the scratch budget cannot hold even one decoded row."""
    if inputs.shape[-1] != columns or inputs.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError('Linear inputs must match weight columns and use a supported float dtype.')
    # Conservative accounting for eager nibble index/scaling intermediates (FP32
    # and int64), not just the final BF16 weight. Output/activation budgets belong
    # to the caller. At least one complete row must fit this reference decoder.
    bytes_per_row = columns * 64
    tile_rows = scratch_bytes // bytes_per_row
    if tile_rows < 1:
        raise MemoryError(f'Reference decode needs at least {bytes_per_row} scratch bytes.')
    output = torch.empty((*inputs.shape[:-1], rows), device=inputs.device, dtype=inputs.dtype)
    for start in range(0, rows, tile_rows):
        end = min(rows, start + tile_rows)
        weight = decode_rows(start, end)
        output[..., start:end] = F.linear(inputs, weight, None if bias is None else bias[start:end])
        del weight
    return output


def mxfp4_linear(inputs: Tensor, blocks: Tensor, scales: Tensor, bias: Tensor | None = None,
                  *, scratch_bytes=16 * 1024**2) -> Tensor:
    """Apply one MXFP4 matrix using output-row tiles and optional row bias.

    Expect blocks [out_features, K/32, 16] and matching E8M0 scales. Return
    [..., out_features] for inputs [..., K], without caching dense weights.
    Reject expert-bank tensors; decoder validation and scratch failures propagate."""
    if blocks.ndim != 3:
        raise ValueError('Linear expects one matrix, not an entire bank of experts.')
    return _tiled_linear(inputs, blocks.shape[0], blocks.shape[1] * 32,
                         lambda start, end: dequantize_mxfp4(blocks[start:end], scales[start:end], inputs.dtype),
                         bias, scratch_bytes)


def _scale_rows(scale, start, end):
    """Keep a shared scalar scale intact, or slice output-row scales for a tile."""
    return scale if scale.numel() == 1 else scale[start:end]


def nvfp4_linear(inputs: Tensor, weight: Tensor, scale: Tensor, global_scale: Tensor,
                  bias: Tensor | None = None, *, layout='modelopt', scratch_bytes=16 * 1024**2) -> Tensor:
    """Apply one NVFP4 matrix with bounded row-decode scratch and optional bias.

    Inputs end in K; weight rows use uint8 K/2 or compressed-tensors int32 K/8
    packing. Scale conventions follow dequantize_nvfp4. Return inputs.shape[:-1]
    plus the output-row dimension. Invalid matrix layouts or insufficient scratch
    raise instead of selecting a different format or quantizing activations."""
    if weight.ndim != 2:
        raise ValueError('Linear expects one matrix, not an entire bank of experts.')
    columns = weight.shape[1] * (8 if weight.dtype == torch.int32 else 2)
    return _tiled_linear(inputs, weight.shape[0], columns,
                         lambda start, end: dequantize_nvfp4(weight[start:end], scale[start:end],
                             _scale_rows(global_scale, start, end), inputs.dtype, layout=layout),
                         bias, scratch_bytes)


def fp8_linear(inputs: Tensor, weight: Tensor, scale: Tensor, bias: Tensor | None = None,
                 *, scratch_bytes=16 * 1024**2) -> Tensor:
    """Apply canonical E4M3FN weights with scalar or per-output-row dequant scales.

    Decode FP32-scaled row tiles to the input dtype before each linear operation.
    Return [..., out_features]; reject non-matrix weights and block-scale layouts.
    CPU/CUDA execution uses PyTorch reference operations, not a native FP8 kernel."""
    if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
        raise ValueError('FP8 linear requires a two-dimensional E4M3FN weight.')
    if scale.numel() != 1 and scale.shape not in ((weight.shape[0],), (weight.shape[0], 1)):
        raise ValueError('FP8 block-scaled layouts require a separate decoder.')
    def decode(start, end):
        """Decode only the requested FP8 output rows, broadcasting their scale by column."""
        scaling = _scale_rows(scale, start, end).float()
        if scaling.numel() != 1:
            scaling = scaling.reshape(-1, 1)
        return (weight[start:end].float() * scaling).to(inputs.dtype)
    return _tiled_linear(inputs, weight.shape[0], weight.shape[1], decode, bias, scratch_bytes)
