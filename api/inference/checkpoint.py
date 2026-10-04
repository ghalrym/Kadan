"""Local safetensors access without loading/dequantizing an entire checkpoint."""
import json
from pathlib import Path
from safetensors import safe_open


class SafeTensorReader:
    def __init__(self, path):
        self.path = Path(path).resolve()
        index = json.loads((self.path / 'model.safetensors.index.json').read_text())
        self.weight_map = index['weight_map']
        self.keys = set(self.weight_map)
        self._headers = {}
        for filename in set(self.weight_map.values()):
            target = (self.path / filename).resolve()
            if target.parent != self.path or not target.is_file():
                raise ValueError('Checkpoint shard must be a local root file')
            with target.open('rb') as stream:
                size = int.from_bytes(stream.read(8), 'little')
                if not 0 < size <= 100_000_000:
                    raise ValueError('Invalid safetensors header size')
                header = json.loads(stream.read(size))
            self._headers.update({key: value for key, value in header.items() if key != '__metadata__'})
        if not self.keys <= self._headers.keys():
            raise ValueError('Checkpoint index references missing tensors')

    def shape(self, name):
        return tuple(self._headers[name]['shape'])

    def nbytes(self, name):
        start, end = self._headers[name]['data_offsets']
        return end - start

    def tensor(self, name):
        with safe_open(self.path / self.weight_map[name], framework='pt', device='cpu') as shard:
            return shard.get_tensor(name)

    def expert(self, name, index):
        with safe_open(self.path / self.weight_map[name], framework='pt', device='cpu') as shard:
            return shard.get_slice(name)[index]
