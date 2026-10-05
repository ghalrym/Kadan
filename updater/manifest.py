"""Build release metadata from tested images; invoked only by publishing CI."""
import json
import os
import subprocess

from api.services.release import identity
from updater.releases import validate_manifest


def main():
    release = {'format': 1, **identity(), 'ci_run': int(os.environ['KADAN_CI_RUN']),
               'publish_run': int(os.environ['GITHUB_RUN_ID'])}
    for service in ('api', 'frontend'):
        repository = f'ghcr.io/ghalrym/kadan-{service}'
        value = json.loads(subprocess.check_output(['docker', 'image', 'inspect',
                                                    repository + ':' + release['commit']]))[0]
        if value['Config']['Labels']['org.opencontainers.image.revision'] != release['commit']:
            raise RuntimeError('Image revision mismatch')
        release[service] = next(digest for digest in value['RepoDigests'] if digest.startswith(repository + '@'))
    print(json.dumps(validate_manifest(release), indent=2))


if __name__ == '__main__':
    main()
