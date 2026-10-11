"""Export pinned Whisper English decoding assets; no model load or download."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct

import numpy as np
import whisper
from whisper.tokenizer import get_tokenizer


def export(vocabulary, destination):
    if importlib.metadata.version('openai-whisper') != '20250625':
        raise ValueError('pinned Whisper package required')
    if vocabulary not in (51864, 51865, 51866):
        raise ValueError('unsupported Whisper vocabulary')
    multi = vocabulary >= 51865
    languages = vocabulary - 51765 - int(multi)
    tokenizer = get_tokenizer(multi, num_languages=languages, language='en', task='transcribe')
    if tokenizer.encoding.n_vocab != vocabulary:
        raise ValueError('tokenizer vocabulary mismatch')
    destination.mkdir()
    output = bytearray(b'KDWVOC01')
    def number(n):
        output.extend(struct.pack('<I', n))
    def array(values):
        number(len(values))
        for value in values:
            number(value)
    number(vocabulary)
    number(tokenizer.eot)
    array(tokenizer.sot_sequence_including_notimestamps)
    array(sorted(set(tokenizer.non_speech_tokens) | set(range(tokenizer.eot + 1, vocabulary))))
    array(tokenizer.encode(' ') + [tokenizer.eot])
    for token in range(vocabulary):
        raw = tokenizer.encoding.decode_single_token_bytes(token)
        if len(raw) > 128:
            raise ValueError('token byte bound')
        number(len(raw))
        output.extend(raw)
    (destination / 'english.tokens').write_bytes(output)
    asset = Path(whisper.__file__).parent / 'assets/mel_filters.npz'
    if hashlib.sha256(asset.read_bytes()).hexdigest() != '7450ae70723a5ef9d341e3cee628c7cb0177f36ce42c44b7ed2bf3325f0f6d4c':
        raise ValueError('filter asset hash mismatch')
    hashes = {}
    with np.load(asset, allow_pickle=False) as filters:
        for bins in (80, 128):
            values = filters[f'mel_{bins}'].astype('<f4')
            data = values.tobytes()
            (destination / f'mel-{bins}.f32').write_bytes(data)
            hashes[str(bins)] = hashlib.sha256(data).hexdigest()
    report = dict(
        language='en',
        whisper_version='20250625',
        vocabulary=vocabulary,
        token_sha256=hashlib.sha256(output).hexdigest(),
        mel_sha256=hashes,
        prompt=list(tokenizer.sot_sequence_including_notimestamps),
        eos=tokenizer.eot,
    )
    (destination / 'assets.json').write_text(json.dumps(report, indent=2))
    return report
