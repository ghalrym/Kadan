"""Print or execute an immutable-source test image build; never starts a container."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile

PIN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._:/-]*@sha256:[0-9a-f]{64}\Z")
LOCAL_IMAGE = re.compile(r"sha256:[0-9a-f]{64}\Z")
DEFAULT_TOOLCHAIN = 'nvidia/cuda@sha256:6b6617592b94e7dcc6ffbe6d00720eed27bc6e3b4f06b26b93b4070c31f57391'
DEFAULT_API = 'sha256:985fc03e9ad5fee38fc50b5f088305c9ab94c9a7b3ac53addf2b20622f7c4e4e'
REVISION = re.compile(r"[0-9a-f]{40}\Z")
TAG = re.compile(r"[a-z0-9][a-z0-9._/-]*(?::[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127})?\Z")


def commands(toolchain_image, api_image, revision, tag):
    if not PIN.fullmatch(toolchain_image):
        raise ValueError('toolchain must be a name@sha256:64-lowercase-hex digest')
    if not (PIN.fullmatch(api_image) or LOCAL_IMAGE.fullmatch(api_image)):
        raise ValueError('API image must be a registry digest or exact local image ID')
    api_base = 'kadan-native-api-base:' + api_image[7:] if LOCAL_IMAGE.fullmatch(api_image) else api_image
    if not REVISION.fullmatch(revision):
        raise ValueError('revision must be the complete 40-character commit SHA')
    if not TAG.fullmatch(tag):
        raise ValueError('invalid local output image tag')
    archive = ['git', 'archive', '--format=tar', revision, 'api', 'native']
    build = ['docker', 'build', '--pull=false', '--platform=linux/amd64', '--file', 'native/packaging/Dockerfile',
             '--build-arg', f'TOOLCHAIN_IMAGE={toolchain_image}',
             '--build-arg', f'API_IMAGE={api_base}',
             '--build-arg', f'SOURCE_REVISION={revision}',
             '--tag', tag, '-']
    return archive, build


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--toolchain-image', default=DEFAULT_TOOLCHAIN)
    parser.add_argument('--api-image', default=DEFAULT_API)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--build', action='store_true', help='execute build (default: print only)')
    args = parser.parse_args(argv)
    try:
        archive, build = commands(args.toolchain_image, args.api_image, args.revision, args.tag)
    except ValueError as error:
        parser.error(str(error))
    local_tag = 'kadan-native-api-base:' + args.api_image[7:] if LOCAL_IMAGE.fullmatch(args.api_image) else None
    prepare = [['docker', 'image', 'inspect', args.api_image, '--format', '{{.Id}}'],
               ['docker', 'image', 'tag', args.api_image, local_tag]] if local_tag else []
    print(json.dumps({'prepare_local_base': prepare, 'archive': archive, 'build': build,
                      'executes_build': args.build}, indent=2))
    if not args.build:
        return 0
    root = Path(__file__).resolve().parents[2]
    # git archive reads only committed source. Untracked deployment files, model
    # payloads and local credentials never enter this build context.
    resolved = subprocess.check_output(['git', 'rev-parse', '--verify', args.revision + '^{commit}'],
                                       cwd=root, text=True).strip()
    if resolved != args.revision:
        raise ValueError('commit identity changed')
    subprocess.run(['git', 'cat-file', '-e', args.revision + ':native/packaging/Dockerfile'],
                   cwd=root, check=True)
    if local_tag:
        identity = subprocess.check_output(prepare[0], cwd=root, text=True).strip()
        if identity != args.api_image:
            raise ValueError('local API image identity mismatch')
        subprocess.run(prepare[1], cwd=root, check=True)
    with tempfile.TemporaryFile() as context:
        subprocess.run(archive, cwd=root, stdout=context, check=True)
        context.seek(0)
        subprocess.run(build, cwd=root, stdin=context, check=True)
    if local_tag:
        identity = subprocess.check_output(['docker', 'image', 'inspect', local_tag, '--format', '{{.Id}}'],
                                           cwd=root, text=True).strip()
        if identity != args.api_image:
            raise ValueError('local base tag changed during build; discard output image')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
