"""Offline verified checkpoint conversion only; no model execution or download.

Run against a trusted installed OpenAI Whisper .pt with its expected SHA-256.
The C++ runtime reads only the resulting safetensors and dimensions files.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile

import torch

DIMENSIONS=('n_mels','n_audio_ctx','n_audio_state','n_audio_head','n_audio_layer',
            'n_vocab','n_text_ctx','n_text_state','n_text_head','n_text_layer')


def export(source, expected, destination, cancel=None):
    def check():
        if cancel is not None and cancel.is_set():
            raise InterruptedError('Native asset preparation cancelled')
    check()
    if not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('expected SHA-256 required')
    if source.is_symlink() or not source.is_file():
        raise ValueError('regular source required')
    if destination.exists():
        raise FileExistsError(destination)
    before = source.stat()
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        while (chunk := stream.read(1024 * 1024)):
            check()
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError('source SHA-256 mismatch')
    torch.set_num_threads(1)
    checkpoint = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    check()
    dims = checkpoint['dims']
    state = checkpoint['model_state_dict']
    if set(dims) != set(DIMENSIONS) or not all(
        type(dims[k]) is int and 0 < dims[k] <= 52000 for k in DIMENSIONS
    ):
        raise ValueError('invalid dimensions')
    if not isinstance(state, dict) or not 1 <= len(state) <= 4096:
        raise ValueError('invalid state')
    metadata = {}
    offset = 0
    for name, tensor in state.items():
        if (
            not isinstance(name, str)
            or len(name) > 256
            or not isinstance(tensor, torch.Tensor)
            or tensor.layout != torch.strided
            or tensor.ndim not in (1, 2, 3)
        ):
            raise ValueError('invalid tensor')
        if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError('unsupported dtype')
        size = tensor.numel() * tensor.element_size()
        if size <= 0 or size > 512 * 1024 * 1024:
            raise ValueError('tensor size bound')
        metadata[name] = dict(
            dtype={torch.float32: 'F32', torch.float16: 'F16', torch.bfloat16: 'BF16'}[tensor.dtype],
            shape=list(tensor.shape),
            data_offsets=[offset, offset + size],
        )
        offset += size
    if offset > 8 * 1024 ** 3:
        raise ValueError('checkpoint size bound')
    header = json.dumps(metadata, separators=(',', ':')).encode()
    if len(header) > 4 * 1024 * 1024:
        raise ValueError('header size bound')
    temporary = Path(tempfile.mkdtemp(prefix='.whisper-export-', dir=destination.parent))
    try:
        output = temporary / 'model.safetensors'
        digest = hashlib.sha256()
        with output.open('xb') as stream:
            for data in (struct.pack('<Q', len(header)), header):
                stream.write(data)
                digest.update(data)
            for tensor in state.values():
                check()
                raw = tensor.detach().contiguous().view(torch.uint8).numpy().reshape(-1)
                for start in range(0, len(raw), 1024 * 1024):
                    check()
                    data = memoryview(raw[start:start + 1024 * 1024])
                    stream.write(data)
                    digest.update(data)
            stream.flush()
            os.fsync(stream.fileno())
        after = source.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('source changed')
        (temporary / 'dimensions.txt').write_text(' '.join((str(dims[k]) for k in DIMENSIONS)) + '\n')
        report = dict(
            source_sha256=actual,
            source_bytes=before.st_size,
            output_sha256=digest.hexdigest(),
            output_bytes=output.stat().st_size,
            tensors=len(state),
            dimensions=dims,
            torch_version=torch.__version__,
            model_execution=False,
        )
        (temporary / 'export.json').write_text(json.dumps(report, indent=2) + '\n')
        # A new destination only; never replace a checkpoint or source data.
        libc = ctypes.CDLL(None, use_errno=True)
        rename = libc.renameat2
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(temporary), -100, os.fsencode(destination), 1) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(destination))
        return report
    except BaseException:
        shutil.rmtree(temporary)
        raise
