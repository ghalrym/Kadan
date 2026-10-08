"""CPU-only metadata planner checks; generated synthetic fixture, no CUDA calls."""
import json
import pathlib
import subprocess
import sys
import tempfile

from model_fixture import create


def main():
    worker = sys.argv[1]
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        create(root)
        config_path = root / "config.json"
        config = json.loads(config_path.read_text())
        config["text_config"]["max_position_embeddings"] = 256
        config_path.write_text(json.dumps(config))
        for capacity in (1, 256):
            result = subprocess.run(
                [worker, '--plan', str(root), str(capacity), str(256 * 1024**2)],
                capture_output=True, text=True, timeout=15, check=True,
            )
            fields = result.stdout.split()
            assert fields[:2] == ['plan', '1'] and len(fields) == 7
            host, arena, vocab, actual_capacity, staging = map(int, fields[2:])
            assert host == 258 * 1024**2 and arena > 0 and vocab == 16
            assert actual_capacity == capacity and 0 < staging <= 1024**2
        for capacity, metadata in [('0', '268435456'), ('262145', '268435456'),
                                   ('1', '0'), ('1', '268435457'), ('-1', '268435456')]:
            result = subprocess.run([worker, '--plan', str(root), capacity, metadata],
                                    capture_output=True, text=True, timeout=15)
            assert result.returncode == 1 and result.stdout == 'error terminal\n'


if __name__ == '__main__':
    main()
