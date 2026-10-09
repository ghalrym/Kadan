"""Independent scalar equations and lifecycle fixtures; no inference runner dependency."""
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def fixture(path, kind='valid'):
    tensors = {}
    def add(name, shape, values):
        tensors[name] = (shape, values)
    add('latents_mean', [24], [(i-12)/32 for i in range(24)])
    add('latents_std', [24], [0 if kind == 'std' else (i+1)/16 for i in range(24)])
    add('post_quant_conv.weight', [24,24,1,1,1], [((r*3+c)%11-5)/32 for r in range(24) for c in range(24)])
    add('post_quant_conv.bias', [24], [i/64 for i in range(24)])
    add('decoder.x_embedder.weight', [2048,24], [((r+c*7)%13-6)/64 for r in range(2048) for c in range(24)])
    add('decoder.x_embedder.bias', [2048], [(r%19-9)/128 for r in range(2048)])
    if kind == 'nan': tensors['latents_mean'][1][0] = math.nan
    # Exercise signed zero, subnormal, and largest finite half during conversion.
    if kind == 'valid':
        tensors['decoder.x_embedder.weight'][1][:3] = [-0.0, 2**-24, 65504.0]
    payload = bytearray()
    header = {}
    for name, (shape, values) in tensors.items():
        start = len(payload)
        payload.extend(struct.pack('<'+'e'*len(values), *values))
        header[name] = dict(dtype='BF16' if kind == 'wrong' else 'F16', shape=shape, data_offsets=[start, len(payload)])
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(encoded))+encoded+payload)
    return {name: values for name, (_, values) in tensors.items()}


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def verify(output, weights, values):
    with output.open('rb') as f:
        assert f.readline() == b'KADAN_H3_DECODER_INPUT_V1\n'
        tokens, hidden = map(int, f.readline().split())
        assert hidden == 2048 and tokens == len(values)//24
        assert f.readline() == b'F32LE\n'
        data=f.read()
    assert len(data)==tokens*2048*4
    actual=struct.unpack('<'+'f'*(len(data)//4),data)
    expected=[]
    for token in range(tokens):
        z=[f32(f32(values[token*24+c]*weights['latents_std'][c])+weights['latents_mean'][c]) for c in range(24)]
        p=[]
        for r in range(24):
            v=weights['post_quant_conv.bias'][r]
            for c in range(24): v=f32(v+f32(weights['post_quant_conv.weight'][r*24+c]*z[c]))
            p.append(v)
        for r in range(2048):
            v=weights['decoder.x_embedder.bias'][r]
            for c in range(24): v=f32(v+f32(weights['decoder.x_embedder.weight'][r*24+c]*p[c]))
            expected.append(v)
    assert all(math.isfinite(x) for x in actual)
    assert data == struct.pack('<'+'f'*len(expected), *expected), max(abs(a-b) for a,b in zip(actual,expected))
    print(f'PASS: {len(actual)} component values bitwise match independent scalar FP32 equations')


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        weights=fixture(root/'weights.safetensors')
        for kind in ('wrong','nan','std'): fixture(root/f'{kind}.safetensors',kind)
        subprocess.run([sys.argv[1],str(root)],check=True)
        values=[(i%17-8)/8 for i in range(72)]
        verify(root/'result',weights,values)
        (root/'input').write_bytes(struct.pack('<72f',*values))
        subprocess.run([sys.argv[2],str(root),'weights.safetensors',str(root/'input'),str(root/'cli')],check=True)
        verify(root/'cli',weights,values)
