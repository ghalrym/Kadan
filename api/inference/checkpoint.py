"""Local safetensors access without loading/dequantizing an entire checkpoint."""
import json
from contextlib import ExitStack
from pathlib import Path
from safetensors import safe_open


class SafeTensorReader:
    def __init__(self, path):
        """Index local shard headers without decoding weights. Reject nonlocal root files,
        invalid header lengths and index names absent from the parsed headers.
        File/JSON errors propagate; safetensors validates tensor layout on access.
        """
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
        self._shards = {}
        self._stack = ExitStack()

    def _shard(self, name):
        """Share one mapping per shard across all tensors and expert slices.

        Returned tensors retain their backing storage after a reader is closed.
        Reopening for each tensor would retain a whole-file mapping per tensor,
        exhausting virtual address space on checkpoints with many experts.
        """
        filename = self.weight_map[name]
        if filename not in self._shards:
            self._shards[filename] = self._stack.enter_context(
                safe_open(self.path / filename, framework='pt', device='cpu'))
        return self._shards[filename]

    def close(self):
        """Release reader handles; tensor owners retain only the mappings they still need."""
        self._stack.close()
        self._shards.clear()

    def shape(self, name):
        """Return the indexed tensor dimensions; raise KeyError for an unknown name."""
        return tuple(self._headers[name]['shape'])

    def nbytes(self, name):
        """Return stored byte length from the header, not decoded tensor memory."""
        start, end = self._headers[name]['data_offsets']
        return end - start

    def tensor(self, name):
        """Return a CPU safetensors tensor by name without dequantizing it; file and lookup errors
        propagate.
        """
        return self._shard(name).get_tensor(name)

    def expert(self, name, index):
        """Return one leading-dimension expert slice on CPU; invalid names or indices propagate
        from safetensors.
        """
        return self._shard(name).get_slice(name)[index]
