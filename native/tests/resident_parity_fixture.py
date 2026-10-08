"""Prepare tiny synthetic GPU parity inputs in a new directory; never runs CUDA."""
import argparse
import json
from pathlib import Path
import shutil
import struct

from model_fixture import create


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    good = args.output / 'good'
    good.mkdir()
    create(good)
    bad = args.output / 'bad'
    shutil.copytree(good, bad)
    path = next(bad.glob('*.safetensors'))
    data = bytearray(path.read_bytes())
    length = struct.unpack_from('<Q', data)[0]
    header = json.loads(data[8:8 + length])
    name = next(name for name in header if name.endswith('.input_scale'))
    offset = 8 + length + header[name]['data_offsets'][0]
    struct.pack_into('<f', data, offset, float('nan'))
    path.write_bytes(data)
    print(f'good={good}\nbad={bad}\ninvalid_tensor={name}')


if __name__ == '__main__':
    main()
