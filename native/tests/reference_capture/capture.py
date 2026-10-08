"""Synthetic-only, one-token CPU reference capture. Actual models are rejected."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
from .artifacts import ExclusiveOutput, encode, require, sha256

PINS = Path(__file__).with_name('backend-pins.json')
REPO = Path(__file__).resolve().parents[3]


def bounded_file(path, maximum):
    require(not path.is_symlink() and path.is_file(), 'fixture_file_type:' + path.name)
    require(path.stat().st_size <= maximum, 'fixture_file_bound')
    with path.open('rb') as stream:
        value = stream.read(maximum + 1)
    require(len(value) <= maximum, 'fixture_file_bound')
    return value


def preflight(root, token):
    require(root.is_dir() and not root.is_symlink(), 'fixture_root')
    raw = json.loads(bounded_file(root / 'config.json', 65536))
    require(raw.get('model_type') == 'qwen3_5_moe', 'fixture_family')
    c = raw['text_config']
    # Before any tensor/shard read or Torch import. No real-model override exists.
    limits = {'hidden_size': 256, 'vocab_size': 16, 'num_hidden_layers': 8,
              'num_experts': 12, 'num_experts_per_tok': 8,
              'moe_intermediate_size': 32, 'shared_expert_intermediate_size': 32,
              'num_attention_heads': 4, 'num_key_value_heads': 4, 'head_dim': 8,
              'linear_num_key_heads': 4, 'linear_num_value_heads': 4,
              'linear_key_head_dim': 4, 'linear_value_head_dim': 4,
              'linear_conv_kernel_dim': 4, 'max_position_embeddings': 8}
    require(all(type(c.get(k)) is int and 0 < c[k] <= v for k, v in limits.items()), 'synthetic_dimensions_only')
    require(c['dtype'] == 'bfloat16' and type(token) is int and 0 <= token < c['vocab_size'], 'fixture_token_or_dtype')
    require(len(c['layer_types']) == c['num_hidden_layers'] and 'full_attention' in c['layer_types'], 'fixture_layers')
    generation = json.loads(bounded_file(root / 'generation_config.json', 65536))
    eos = generation['eos_token_id']
    require(isinstance(eos, list) and 0 < len(eos) <= 16 and len(set(eos)) == len(eos)
            and all(type(n) is int and 0 <= n < c['vocab_size'] for n in eos), 'fixture_eos')
    index = json.loads(bounded_file(root / 'model.safetensors.index.json', 262144))
    require(set(index['weight_map'].values()) == {'fixture.safetensors'}, 'synthetic_shard_only')
    names = ['config.json', 'generation_config.json', 'model.safetensors.index.json',
             'fixture.safetensors', 'tokenizer.json', 'tokenizer_config.json']
    if (root / 'special_tokens_map.json').exists():
        names.append('special_tokens_map.json')
    pins = {name: sha256(bounded_file(root / name, 2 * 1024**2)) for name in names}
    return c, eos, pins


def verify_backend_sources():
    pins = json.loads(PINS.read_text())
    for name, version in pins['packages'].items():
        require(importlib.metadata.version(name) == version, 'backend_package:' + name)
    for name, want in pins['installed_sources'].items():
        distribution = importlib.metadata.distribution(name.split('/')[0])
        require(sha256(Path(distribution.locate_file(name)).read_bytes()) == want, 'backend_source:' + name)
    for name, want in pins['reference_sources'].items():
        require(sha256((REPO / name).read_bytes()) == want, 'reference_source:' + name)
    return pins


def run(root, token, output, diagnostic, timeout):
    require(1 <= timeout <= 120, 'synthetic_timeout')
    root = Path(root)
    c, eos, fixture_pins = preflight(root, token)
    pins = verify_backend_sources()
    for path in (Path(output), Path(diagnostic)):
        require(root.resolve() not in path.resolve().parents, 'output_inside_fixture')
    capture = ExclusiveOutput(output)
    try:
        details = ExclusiveOutput(diagnostic)
    except BaseException:
        capture.close()
        raise
    # Fail-stop child process; the outer supervisor must observe actual exit.
    def expired(signum, frame):
        os._exit(124)
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(timeout)
    try:
        for name, value in {'CUDA_VISIBLE_DEVICES': '', 'USE_HUB_KERNELS': 'NO',
                            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                            'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                            'OPENBLAS_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false'}.items():
            os.environ[name] = value
        require('torch' not in sys.modules and 'transformers' not in sys.modules, 'fresh_process_required')
        # Delayed import is necessary: reject real dimensions and source drift
        # before importing any numerical/backend package or loading payloads.
        backend = importlib.import_module('native.tests.reference_capture.cpu_backend')
        logits, report = backend.forward(root, token)
        require({k: report[k] for k in pins["runtime_build"]} == pins["runtime_build"], "backend_runtime_build")
        _, _, after = preflight(root, token)
        require(after == fixture_pins, 'fixture_changed')
        selected = max(range(len(logits)), key=logits.__getitem__)
        data = encode(token, selected, selected in eos, logits)
        wrapper_hashes = {p.name: sha256(p.read_bytes()) for p in sorted(PINS.parent.glob('*.py'))}
        report.update(wrapper_hashes=wrapper_hashes, status='complete', fixture_hashes=fixture_pins, pins= pins,
                      input=token, selected=selected, eos=selected in eos, capacity=1,
                      capture_sha256=sha256(data), capture_bytes=len(data),
                      reference_layer_hashes_cannot_localize_native_divergence=True)
        # A valid capture is published only after backend cleanup and diagnostics.
        details.write(json.dumps(report, allow_nan=False, indent=2).encode() + b'\n')
        details.finish()
        capture.write(data[:-4])
        capture.write(data[-4:])
        capture.finish()
        signal.alarm(0)
        return report
    finally:
        capture.close()
        details.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-root', required=True)
    parser.add_argument('--token', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--diagnostic', required=True)
    parser.add_argument('--timeout', type=int, default=60)
    args = parser.parse_args()
    try:
        report = run(args.synthetic_root, args.token, args.output, args.diagnostic, args.timeout)
        print(json.dumps({k: report[k] for k in ('status', 'input', 'selected', 'capture_sha256', 'zero_reservations')}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error': str(error), 'discard_incomplete_capture': True}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
