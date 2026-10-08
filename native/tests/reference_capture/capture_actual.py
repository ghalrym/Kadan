"""Gated, CPU-only one-BOS reference. No approval or shard manifest is supplied."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys

from .artifacts import ExclusiveOutput, encode, require, sha256
from .capture import PINS, REPO, verify_backend_sources
from .limits import ACTUAL_LIMITS

SNAPSHOT = 'small-1355db6a052410cfd62085d94b58866fd0f2c3c5'
REVISION = SNAPSHOT.removeprefix('small-')
METADATA = {
    'config.json': '58aefa1c9eff7989f431d748f2ddec39446cb1fd2a69acc46e285c6a37b0cecc',
    'generation_config.json': 'e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e',
    'tokenizer_config.json': '5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b',
}
DIGEST = re.compile(r'[0-9a-f]{64}')
BUFFER = 1024**2
MAX_TOTAL = 32 * 1024**3


def plain_path(path):
    path = Path(path).absolute()
    require(all(not p.is_symlink() for p in (path, *path.parents)), 'symlink_path')
    return path


def identity(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def checked_file(path, size, digest, collect=False):
    """No-follow regular file, fixed buffer, bounded total, before/after identity."""
    path = plain_path(path)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_size == size, 'file_size_or_type')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        require(identity(os.fstat(stream.fileno())) == identity(before), 'file_replaced')
        h, chunks, total = hashlib.sha256(), [], 0
        while True:
            chunk = stream.read(min(BUFFER, size - total + 1))
            if not chunk:
                break
            total += len(chunk)
            require(total <= size, 'file_grew')
            h.update(chunk)
            if collect:
                chunks.append(chunk)
        require(total == size and identity(os.fstat(stream.fileno())) == identity(before)
                and identity(path.stat()) == identity(before), 'file_changed')
    require(h.hexdigest() == digest, 'file_digest:' + path.name)
    return b''.join(chunks) if collect else identity(before)


def bounded_json(path, maximum, digest=None):
    size = plain_path(path).stat().st_size
    require(0 < size <= maximum, 'json_bound')
    # Only manifest bootstrap lacks an in-document digest; caller must pin it.
    require(digest is not None and DIGEST.fullmatch(digest), 'json_digest_required')
    return json.loads(checked_file(path, size, digest, collect=True))


def validate_manifest(m):
    require(m['schema'] == 1 and m['execution_authorized'] is True
            and m['snapshot_writer_exclusion'] is True, 'explicit_approval_required')
    require(m['snapshot'] == SNAPSHOT and m['revision'] == REVISION, 'snapshot_pin')
    require(m['input_ids'] == [248044] and m['capacity'] == 1
            and m['timeout_seconds'] == 1800, 'actual_input_or_timeout')
    require(m['host_bytes'] == ACTUAL_LIMITS.host_bytes
            and m['request_bytes'] == ACTUAL_LIMITS.request_bytes, 'actual_budget')
    pins = json.loads(PINS.read_text())
    require(m['image_id'] == pins['image_id'] and m['backend_pins_sha256'] == sha256(PINS.read_bytes()), 'backend_pin')
    require(re.fullmatch('[0-9a-f]{40}', m['source_revision']) is not None, 'source_revision')
    sources = {p.name: sha256(p.read_bytes()) for p in PINS.parent.glob('*.py')}
    require(m['wrapper_sources'] == sources, 'wrapper_sources')
    files = m['files']
    require(type(files) is dict and 5 <= len(files) <= 128, 'file_count')
    for name, pin in files.items():
        require(re.fullmatch(r'[A-Za-z0-9_.-]+', name) is not None
                and name not in ('.', '..'), 'file_name')
        require(type(pin['size']) is int and 0 < pin['size'] <= MAX_TOTAL
                and DIGEST.fullmatch(pin['sha256']) is not None, 'file_pin')
    require(sum(p['size'] for p in files.values()) <= MAX_TOTAL, 'snapshot_size')
    for name, digest in METADATA.items():
        require(files[name]['sha256'] == digest and files[name]['size'] <= BUFFER, 'metadata_pin')
    require(files['model.safetensors.index.json']['size'] <= 16 * BUFFER, 'index_bound')
    return files


def preflight(root, m):
    files = validate_manifest(m)
    root = plain_path(root)
    require(root.is_dir() and root.name == SNAPSHOT, 'snapshot_root')
    head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], check=True,
                          capture_output=True, text=True, timeout=5).stdout.strip()
    require(head == m['source_revision'], 'source_head')
    verify_backend_sources()  # No numerical imports or shard reads.
    index_pin = files['model.safetensors.index.json']
    index = bounded_json(root / 'model.safetensors.index.json', 16 * BUFFER, index_pin['sha256'])
    shards = set(index['weight_map'].values())
    require(shards and all(re.fullmatch(r'model-[0-9]+-of-[0-9]+\.safetensors', s) for s in shards), 'index_shards')
    required = set(METADATA) | {'model.safetensors.index.json', 'tokenizer.json'} | shards
    require(set(files) in (required, required | {'special_tokens_map.json'}), 'manifest_inventory')
    # Reject extra loader-visible files, including unpinned alternate weights/code.
    require({p.name for p in root.iterdir()} == set(files), 'snapshot_inventory')
    for name in METADATA:
        pin = files[name]
        checked_file(root / name, pin['size'], pin['sha256'])
    config = bounded_json(root / 'config.json', BUFFER, METADATA['config.json'])
    c = config['text_config']
    require(config['model_type'] == 'qwen3_5_moe' and c['hidden_size'] == 2048
            and c['vocab_size'] == 248320 and c['num_hidden_layers'] == 40
            and c['num_experts'] == 256 and c['num_experts_per_tok'] == 8
            and c['dtype'] == 'bfloat16', 'actual_architecture')
    generation = bounded_json(root / 'generation_config.json', BUFFER, METADATA['generation_config.json'])
    require(generation['bos_token_id'] == 248044 and set(generation['eos_token_id']) == {248044,248046}, 'actual_tokens')
    return files


def enforce_isolation(cgroup=Path('/sys/fs/cgroup')):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'void', 'gpu_visibility')
    require(not list(Path('/dev').glob('nvidia*')) and not Path('/dev/dri').exists(), 'gpu_devices')
    require({p.name for p in Path('/sys/class/net').iterdir()} <= {'lo'}, 'network_isolation')
    for name, ceiling in [('memory.max', 40 * 1024**3), ('pids.max',256)]:
        value = (cgroup / name).read_text().strip()
        require(value.isdecimal() and 0 < int(value) <= ceiling, 'cgroup:' + name)
    require((cgroup / 'memory.swap.max').read_text().strip() == '0', 'cgroup_swap')
    quota, period = (cgroup / 'cpu.max').read_text().split()
    require(quota.isdecimal() and period.isdecimal() and 0 < int(quota) <= 2 * int(period), 'cgroup_cpu')


def run(root, manifest, manifest_sha256, output_dir, execute=False):
    require(execute is True, 'execution_intent_required')
    require('torch' not in sys.modules and 'transformers' not in sys.modules, 'fresh_process_required')
    m = bounded_json(Path(manifest), BUFFER, manifest_sha256)
    validate_manifest(m)
    enforce_isolation()
    # Internal alarm covers source verification, hashing, load, one forward and publication.
    def expired(signum, frame):
        os._exit(124)
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(1800)
    root = plain_path(root)
    destination = plain_path(output_dir)
    require(root != destination and root not in destination.parents, 'output_inside_snapshot')
    destination.mkdir(mode=0o700)  # Entire run directory must be new.
    capture = ExclusiveOutput(destination / 'reference.capture')
    details = None
    try:
        details = ExclusiveOutput(destination / 'diagnostic.json')
        files = preflight(root, m)
        identities = {name: checked_file(root / name, p['size'], p['sha256']) for name,p in files.items()}
        for name, value in {'USE_HUB_KERNELS':'NO', 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1',
                            'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2',
                            'TOKENIZERS_PARALLELISM':'false'}.items():
            os.environ[name] = value
        # Deliberately delayed until authorization, provenance, isolation and all hashes pass.
        backend = importlib.import_module('native.tests.reference_capture.cpu_backend')
        logits, report = backend.forward(root, 248044, ACTUAL_LIMITS)
        pins = json.loads(PINS.read_text())
        require({k: report[k] for k in pins['runtime_build']} == pins['runtime_build'], 'backend_runtime_build')
        require(report['zero_reservations'] is True and report['calls']['model'] == 1
                and len(logits) == 248320, 'actual_completion')
        require(preflight(root, m) == files, 'manifest_changed')
        for name,p in files.items():
            require(checked_file(root/name, p['size'], p['sha256']) == identities[name], 'snapshot_changed')
        require(bounded_json(Path(manifest), BUFFER, manifest_sha256) == m, 'manifest_changed')
        selected = max(range(len(logits)), key=logits.__getitem__)
        data = encode(248044, selected, selected in {248044,248046}, logits)
        report.update(status='complete', manifest_sha256=manifest_sha256,
                      capture_sha256=sha256(data), capture_bytes=len(data),
                      input=248044, selected=selected, eos=selected in {248044,248046},
                      canonical_native_parity='unknown; requires separate exact 0/0 comparison',
                      raw_hf_ties_preserved=True)
        diagnostic = json.dumps(report, allow_nan=False, indent=2).encode() + b'\n'
        require(len(diagnostic) <= 8 * BUFFER and len(data) == 993316, 'output_bound')
        details.write(diagnostic)
        details.finish()
        capture.write(data)
        capture.finish()
        return report
    finally:
        capture.close()
        if details is not None:
            details.close()
        signal.alarm(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-approved-actual-reference', action='store_true')
    parser.add_argument('--snapshot-root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    try:
        report = run(args.snapshot_root, args.manifest, args.manifest_sha256, args.output_dir,
                     args.execute_approved_actual_reference)
        print(json.dumps({k:report[k] for k in ('status','selected','capture_sha256','zero_reservations')}))
        return 0
    except Exception as error:
        print(json.dumps({'status':'failed','error':str(error)[:4096], 'discard_capture':True}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
