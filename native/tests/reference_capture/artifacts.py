"""Bounded one-token capture format and exact diagnostics; stdlib only."""
import hashlib
import math
import os
from pathlib import Path
import struct

MAGIC = 0x4B4D4331
DONE = 0x444F4E45
MAX_VOCAB = 262144


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(token, selected, eos, logits):
    require(0 < len(logits) <= MAX_VOCAB, 'capture_vocabulary')
    values = [struct.unpack('<f', struct.pack('<f', v))[0] for v in logits]
    require(all(math.isfinite(v) for v in values), 'capture_nonfinite')
    require(type(token) is int and 0 <= token < len(values), 'capture_input')
    require(type(selected) is int and selected == max(range(len(values)), key=values.__getitem__), 'capture_greedy')
    require(type(eos) is bool, 'capture_eos')
    return (struct.pack('<8I', MAGIC, 1, len(values), 1, token, selected, int(eos), 1)
            + struct.pack('<' + 'f' * len(values), *values) + struct.pack('<I', DONE))


def decode(data):
    require(len(data) >= 36, 'capture_truncated')
    magic, version, vocab, count, token, selected, eos, progress = struct.unpack_from('<8I', data)
    require(magic == MAGIC and version == 1 and 0 < vocab <= MAX_VOCAB and count == 1, 'capture_header')
    require(len(data) == 36 + 4 * vocab, 'capture_size_or_trailing')
    require(struct.unpack_from('<I', data, len(data) - 4)[0] == DONE, 'capture_done')
    require(token < vocab and selected < vocab and eos in (0, 1) and progress == 1, 'capture_record')
    logits = list(struct.unpack_from('<' + 'f' * vocab, data, 32))
    require(all(math.isfinite(v) for v in logits), 'capture_nonfinite')
    require(selected == max(range(vocab), key=logits.__getitem__), 'capture_greedy')
    return {'vocab': vocab, 'input': token, 'selected': selected, 'eos': bool(eos), 'progress': progress, 'logits': logits}


def read(path):
    # Bound before allocation even for malformed/special input files.
    with Path(path).open('rb') as stream:
        data = stream.read(36 + MAX_VOCAB * 4 + 1)
    return decode(data)


def ordered_bits(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    # Signed zero has zero numeric distance, as in abs_tol=rel_tol=0.
    return 0x80000000 - (bits & 0x7FFFFFFF) if bits >> 31 else 0x80000000 + bits


def margin(logits):
    order = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
    return None if len(order) == 1 else logits[order[0]] - logits[order[1]]


def compare(expected, actual):
    require(expected['vocab'] == actual['vocab'], 'capture_shape')
    pairs = list(zip(expected['logits'], actual['logits']))
    mismatches = sum(e != a for e, a in pairs)
    record_equal = all(expected[k] == actual[k] for k in ('input', 'selected', 'eos', 'progress'))
    return {
        'accepted': record_equal and mismatches == 0,
        'absolute_tolerance': 0, 'relative_tolerance': 0,
        'record_equal': record_equal, 'mismatch_count': mismatches,
        'max_absolute_error': max(abs(e - a) for e, a in pairs),
        'max_float32_ulp_distance': max(abs(ordered_bits(e) - ordered_bits(a)) for e, a in pairs),
        'max_bfloat16_step_distance': (max(abs(ordered_bits(e) - ordered_bits(a)) // 65536 for e, a in pairs)
            if all((struct.unpack('<I', struct.pack('<f', v))[0] & 65535) == 0 for pair in pairs for v in pair) else None),
        'expected_token_margin': margin(expected['logits']),
        'actual_token_margin': margin(actual['logits']),
        'diagnostics_never_override_acceptance': True,
    }


class ExclusiveOutput:
    """Create once; never overwrite/follow a symlink. Exit status is required even when bytes contain DONE."""
    def __init__(self, path):
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)

    def write(self, data):
        require(self.fd >= 0, 'output_closed')
        view = memoryview(data)
        while view:
            written = os.write(self.fd, view)
            if written <= 0:
                raise OSError('output_short_write')
            view = view[written:]

    def finish(self):
        require(self.fd >= 0, 'output_closed')
        os.fsync(self.fd)
        fd, self.fd = self.fd, -1
        os.close(fd)

    def close(self):
        if self.fd >= 0:
            fd, self.fd = self.fd, -1
            os.close(fd)


def sha256(data):
    return hashlib.sha256(data).hexdigest()
