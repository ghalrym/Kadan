"""Explicit CPU-only selected-tensor check against an installed H3 VAE checkpoint.

Not a full decoder reference or an upstream H3 parity certification.
Usage: python video_checkpoint.py COMPONENT CHECKPOINT NEW_OUTPUT_DIRECTORY
"""
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess
import sys

from video_fixture import verify


def run(binary, checkpoint, output):
    output.mkdir(parents=True, exist_ok=False)
    before = checkpoint.stat()
    names = ('latents_mean', 'latents_std', 'post_quant_conv.weight',
             'post_quant_conv.bias', 'decoder.x_embedder.weight', 'decoder.x_embedder.bias')
    weights, hashes = {}, {}
    with checkpoint.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if length > 1024 * 1024:
            raise ValueError('checkpoint header exceeds component bound')
        encoded = stream.read(length)
        header = json.loads(encoded)
        for name in names:
            tensor = header[name]
            assert tensor['dtype'] == 'F16'
            start, end = tensor['data_offsets']
            assert 0 <= start < end <= before.st_size - length - 8
            assert end - start <= 100000
            stream.seek(8 + length + start)
            data = stream.read(end - start)
            assert len(data) == end - start
            weights[name] = list(struct.unpack('<' + 'e' * (len(data)//2), data))
            hashes[name] = hashlib.sha256(data).hexdigest()
    rng = random.Random(42)
    values = [0.0]*24 + [(i-12)/16 for i in range(24)] + [rng.uniform(-1, 1) for _ in range(24)]
    packed = struct.pack('<72f', *values)
    values = list(struct.unpack('<72f', packed))
    (output/'input.f32').write_bytes(packed)
    result = subprocess.run([str(binary), str(checkpoint.parent), checkpoint.name,
                             str(output/'input.f32'), str(output/'component.tensor')],
                            capture_output=True, text=True, check=True)
    verify(output/'component.tensor', weights, values)
    after = checkpoint.stat()
    assert (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
    report = dict(checkpoint=str(checkpoint), checkpoint_bytes=before.st_size,
                  checkpoint_mtime_ns=before.st_mtime_ns, header_sha256=hashlib.sha256(encoded).hexdigest(),
                  selected_tensor_sha256=hashes, tokens=3, compared_values=6144,
                  max_absolute_error=0, reference='independent scalar FP32 equations with explicit rounding',
                  component_stdout=result.stdout, full_video_generation=False, gpu_execution=False,
                  output_sha256=hashlib.sha256((output/'component.tensor').read_bytes()).hexdigest())
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())
