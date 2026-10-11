"""Export tokenizer metadata only; no model execution or downloads."""
import argparse
import hashlib
import json
from pathlib import Path


def export(root, destination):
    # Optional packaging dependency; absent from worker/API execution.
    from transformers import AutoTokenizer
    if destination.exists():
        raise ValueError('Destination must not already exist')
    destination.mkdir(mode=0o700, parents=True)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    output = destination / 'tokenizer.json'
    tokenizer.backend_tokenizer.save(str(output))
    inputs = {}
    for name in ('vocab.json', 'merges.txt', 'tokenizer_config.json', 'config.json'):
        with (root / name).open('rb') as source:
            inputs[name] = hashlib.file_digest(source, 'sha256').hexdigest()
    with output.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    report = dict(version=1, tokenizer_sha256=digest, source_sha256=inputs,
                  tokenizer_class=type(tokenizer).__name__, python_inference=False)
    (destination / 'manifest.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    export(args.checkpoint, args.destination)
