"""Dense fallback for explicitly described Klein FP8/NVFP4 transformer files.

Independent format arithmetic; no ComfyUI or quantized-kernel dependency. NVFP4
scale addresses follow NVIDIA cuBLAS SWIZZLE_32_4_4 tiles. The serialized format
uses high-nibble-first FP4, unlike Kadan's canonical ModelOpt low-first decoder.
Decoded inference uses ordinary BF16/FP32 weights, not optimized FP4/FP8 kernels.
"""
import json
import math
from pathlib import Path
import struct

import torch
from safetensors.torch import load_file

from api.inference.quantization import dequantize_nvfp4
from api.inference.resources import ResourceCancelled

SCRATCH_BYTES = 64 * 1024 ** 2
AUXILIARY = ('comfy_quant', 'weight_scale', 'weight_scale_2', 'pre_quant_scale', 'input_scale')
FLOAT8 = {'float8_e4m3fn': torch.float8_e4m3fn, 'float8_e5m2': torch.float8_e5m2}


def transformer_path(root, filename):
    """Resolve an explicit local bundle asset, never a remote file or parent path."""
    candidate = Path(filename)
    if candidate.is_absolute() or '..' in candidate.parts or '\\' in filename:
        raise ValueError('Invalid quantized transformer path')
    path = root / candidate
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Missing local quantized transformer')
    return path


def dense_size(path, dtype_bytes):
    """Estimate dense transformer storage from header shapes before loading any tensors."""
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Invalid safetensors header')
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= min(path.stat().st_size - 8, 16 * 1024 ** 2):
            raise ValueError('Invalid safetensors header length')
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError('Invalid safetensors tensor map')
    total = 0
    for key, info in header.items():
        if key == '__metadata__' or key.rsplit('.', 1)[-1] in AUXILIARY:
            continue
        if not isinstance(info, dict):
            raise ValueError('Invalid safetensors tensor description')
        shape = info.get('shape')
        if not isinstance(shape, list) or any(type(size) is not int or size < 1 for size in shape):
            raise ValueError('Unsupported checkpoint shape')
        dtype = info.get('dtype')
        if dtype not in ('BF16', 'F16', 'F32', 'F8_E4M3', 'F8_E5M2', 'U8'):
            raise ValueError('Unsupported checkpoint tensor dtype')
        packed = dtype == 'U8'
        if packed and (not key.endswith('.weight') or len(shape) != 2):
            raise ValueError('Unexpected byte tensor in transformer')
        total += math.prod(shape) * dtype_bytes * (2 if packed else 1)
    if total == 0:
        raise ValueError('Transformer contains no weights')
    return total


def _metadata(tensor):
    """Read the small per-layer format descriptor and refuse unimplemented transforms."""
    if tensor.dtype != torch.uint8 or tensor.ndim != 1 or tensor.numel() > 4096:
        raise ValueError('Invalid quantization descriptor')
    description = json.loads(bytes(tensor.tolist()).decode('utf-8'))
    if not isinstance(description, dict) or set(description) - {'format', 'full_precision_matrix_mult'}:
        raise ValueError('Unknown quantization descriptor fields')
    return description.get('format')


def _finite(value, name):
    if not value.is_floating_point() or not torch.isfinite(value.float()).all():
        raise ValueError(f'Invalid {name}')
    return value.float()


def _row_scale(scale, rows):
    if scale.numel() != 1 and tuple(scale.shape) not in ((rows,), (rows, 1)):
        raise ValueError('Weight scale must be scalar or per output row')
    return _finite(scale, 'weight scale').reshape(-1, 1)


def swizzled_scales(scales, rows, columns, first, last):
    """Gather one row tile from padded 128-row/4-column cuBLAS scale groups."""
    block_columns = columns // 16
    padded_rows = ((rows + 127) // 128) * 128
    padded_columns = ((block_columns + 3) // 4) * 4
    if scales.numel() != padded_rows * padded_columns or scales.ndim not in (1, 2):
        raise ValueError('Invalid SWIZZLE_32_4_4 scale storage')
    if scales.dtype == torch.uint8:
        scales = scales.view(torch.float8_e4m3fn)
    if scales.dtype != torch.float8_e4m3fn:
        raise ValueError('NVFP4 block scales must be E4M3 bytes')
    row = torch.arange(first, last)[:, None]
    column = torch.arange(block_columns)[None, :]
    tile = (row // 128) * (padded_columns // 4) + column // 4
    offset = tile * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + column % 4
    # CPU advanced indexing for float8 is not implemented on every torch build.
    return scales.view(torch.uint8).flatten()[offset].view(torch.float8_e4m3fn)


def decode_state_dict(state, quantization, dtype=torch.bfloat16, cancel=None):
    """Decode supported layer weights with bounded scratch and fold input column scales."""
    if quantization not in ('fp8', 'nvfp4') or dtype not in (torch.bfloat16, torch.float32):
        raise ValueError('Unsupported dense transformer fallback')
    output, consumed = {}, set()
    quantized_count = 0
    for key, weight in state.items():
        if key.rsplit('.', 1)[-1] in AUXILIARY:
            continue
        if cancel is not None and cancel.is_set():
            raise ResourceCancelled('Transformer decoding cancelled')
        if weight.device.type != 'cpu':
            raise ValueError('Decode checkpoint tensors on CPU before device placement')
        prefix = key.removesuffix('weight') if key.endswith('.weight') else None
        descriptor = state.get(prefix + 'comfy_quant') if prefix else None
        fmt = _metadata(descriptor) if descriptor is not None else None
        if descriptor is not None and fmt not in (*FLOAT8, 'nvfp4'):
            raise ValueError('Unsupported quantization format')
        if fmt is None and weight.dtype in FLOAT8.values():
            fmt = next(name for name, value in FLOAT8.items() if value == weight.dtype)
        if fmt is None:
            if weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise ValueError(f'Undescribed quantized tensor: {key}')
            output[key] = weight.to(dtype)
            continue
        if prefix is None or weight.ndim != 2:
            raise ValueError('Only linear matrix weights support quantization')
        if (quantization == 'fp8' and fmt not in FLOAT8) or (quantization == 'nvfp4' and fmt != 'nvfp4'):
            raise ValueError('Layer format does not match the registered checkpoint')
        quantized_count += 1
        rows = weight.shape[0]
        columns = weight.shape[1] * (2 if fmt == 'nvfp4' else 1)
        pre = state.get(prefix + 'pre_quant_scale')
        if pre is not None:
            if tuple(pre.shape) not in ((columns,), (1, columns)):
                raise ValueError('Input scale must have one value per matrix column')
            pre = _finite(pre, 'input column scale').reshape(1, columns)
        scale = state.get(prefix + 'weight_scale')
        if fmt in FLOAT8:
            if prefix + 'weight_scale_2' in state:
                raise ValueError('Unexpected FP8 secondary weight scale')
            if weight.dtype != FLOAT8[fmt]:
                raise ValueError('FP8 format and stored dtype disagree')
            scale = _row_scale(scale if scale is not None else torch.ones(()), rows)
        else:
            if weight.dtype != torch.uint8 or columns % 16 or scale is None:
                raise ValueError('NVFP4 requires packed bytes and block scales')
            global_scale = state.get(prefix + 'weight_scale_2')
            if global_scale is None or global_scale.numel() != 1:
                raise ValueError('NVFP4 requires one global dequantization multiplier')
            global_scale = _finite(global_scale, 'global weight scale')
        if rows == 0 or columns == 0:
            raise ValueError('Quantized matrices must not be empty')
        tile_rows = max(1, SCRATCH_BYTES // (columns * 64))
        if columns * 64 > SCRATCH_BYTES:
            raise ValueError('Matrix row exceeds the decoder scratch budget')
        result = torch.empty((rows, columns), dtype=dtype)
        for first in range(0, rows, tile_rows):
            if cancel is not None and cancel.is_set():
                raise ResourceCancelled('Transformer decoding cancelled')
            last = min(rows, first + tile_rows)
            if fmt in FLOAT8:
                decoded = weight[first:last].float() * (scale if scale.numel() == 1 else scale[first:last])
            else:
                block_scale = swizzled_scales(scale, rows, columns, first, last)
                if not torch.isfinite(block_scale.float()).all():
                    raise ValueError('NVFP4 block scale is not finite')
                packed = weight[first:last]
                canonical = ((packed & 15) << 4) | (packed >> 4)
                decoded = dequantize_nvfp4(canonical, block_scale, global_scale)
            if pre is not None:
                decoded *= pre
            if not torch.isfinite(decoded).all():
                raise ValueError('Decoded weights are not finite')
            converted = decoded.to(dtype)
            if not torch.isfinite(converted).all():
                raise ValueError('Decoded weights exceed the inference dtype range')
            result[first:last] = converted
        output[key] = result
        consumed.update(prefix + name for name in AUXILIARY if prefix + name in state)
    if quantized_count == 0:
        raise ValueError('Checkpoint has no supported quantized matrices')
    if any(key.rsplit('.', 1)[-1] in AUXILIARY and key not in consumed for key in state):
        raise ValueError('Orphaned or unsupported quantization metadata')
    return output


def load_transformer(root, filename, quantization, dtype, diffusers, cancel):
    """Pass decoded original keys and matching local config to the native Flux2 loader."""
    if not (root / 'transformer/config.json').is_file():
        raise ValueError('Missing local transformer configuration')
    state = load_file(str(transformer_path(root, filename)), device='cpu')
    decoded = decode_state_dict(state, quantization, dtype, cancel)
    state.clear()
    for prefix in ('model.diffusion_model.', 'diffusion_model.'):
        if decoded and all(key.startswith(prefix) for key in decoded):
            decoded = {key.removeprefix(prefix): value for key, value in decoded.items()}
            break
    return diffusers.Flux2Transformer2DModel.from_single_file(decoded, config=str(root),
        subfolder='transformer', local_files_only=True, torch_dtype=dtype)
