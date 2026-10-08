"""Print an isolated API container command; this module cannot execute it."""
import argparse
import json
from pathlib import PurePosixPath
import re


def absolute(value):
    path = PurePosixPath(value)
    if not path.is_absolute() or '..' in path.parts or ',' in value or '\n' in value or value == '/':
        raise ValueError('expected absolute non-root path without traversal/comma/newline')
    return str(path)


def command(image, state_dir, checkpoint_dir, checkpoint_relative, env_file, network,
            gpu, host_bytes, device_bytes, port):
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
        raise ValueError('use the exact built local image ID')
    if not re.fullmatch(r'kadan-native-review-[a-z0-9-]+', network):
        raise ValueError('use a separately reviewed private test network')
    relative = PurePosixPath(checkpoint_relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts or ',' in checkpoint_relative or '\n' in checkpoint_relative:
        raise ValueError('checkpoint target must be a relative path below isolated model state')
    state, checkpoint, env = map(absolute, (state_dir, checkpoint_dir, env_file))
    if state == checkpoint or PurePosixPath(state) in PurePosixPath(checkpoint).parents or PurePosixPath(checkpoint) in PurePosixPath(state).parents:
        raise ValueError('state and checkpoint source directories must be disjoint')
    if not (0 <= gpu < 64 and host_bytes >= 770 * 1024**2 and device_bytes > 512 * 1024**2 and 1024 <= port <= 65535):
        raise ValueError('invalid device, host/device envelope or port')
    return ['docker', 'run', '--rm', '--init', '--name', 'kadan-native-api-review',
            '--network', network, '--gpus', f'device={gpu}', '--cpus', '2',
            '--memory', str(host_bytes), '--memory-swap', str(host_bytes), '--pids-limit', '256',
            '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--read-only',
            '--tmpfs', '/tmp:rw,nosuid,nodev,size=268435456', '--stop-timeout', '300',
            '--publish', f'127.0.0.1:{port}:8000', '--env-file', env,
            '--env', 'KADAN_LLM_BACKEND=native', '--env', 'KADAN_GPU=0',
            '--env', 'KADAN_NATIVE_WORKER=/opt/kadan/bin/run-model-worker',
            '--env', 'KADAN_GPU_BUDGET_BYTES=' + json.dumps({'0': device_bytes}, separators=(',', ':')),
            '--env', 'KADAN_MODEL_DIR=/var/lib/kadan/models', '--env', 'HF_HUB_OFFLINE=1',
            '--env', 'TRANSFORMERS_OFFLINE=1', '--env', 'HF_HOME=/tmp/hf',
            '--mount', f'type=bind,src={state},dst=/var/lib/kadan/models',
            '--mount', f'type=bind,src={checkpoint},dst=/var/lib/kadan/models/{relative},readonly', image]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('image', 'state-dir', 'checkpoint-dir', 'checkpoint-relative', 'env-file', 'network'):
        parser.add_argument('--' + option, required=True)
    for option in ('gpu', 'host-bytes', 'device-bytes', 'port'):
        parser.add_argument('--' + option, required=True, type=int)
    args = vars(parser.parse_args())
    try:
        result = command(**args)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({'preview_only': True, 'command': result}, indent=2))


if __name__ == '__main__':
    main()
