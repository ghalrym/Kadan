"""Disposable real-container release smoke; no host installation or credentials."""
import json
import os
import subprocess
import time
import uuid
from urllib.request import ProxyHandler, build_opener
from urllib.error import HTTPError

from api.services.release import migration_fingerprint

urlopen = build_opener(ProxyHandler({})).open


def main():
    commit = os.environ['KADAN_COMMIT']
    prefix = 'kadan-smoke-' + uuid.uuid4().hex[:12]
    api = os.environ.get('KADAN_TEST_API_IMAGE', f'ghcr.io/ghalrym/kadan-api:{commit}')
    frontend = os.environ.get('KADAN_TEST_FRONTEND_IMAGE', f'ghcr.io/ghalrym/kadan-frontend:{commit}')
    def run(*args):
        return subprocess.check_output(['docker', *args], text=True).strip()
    containers = []
    try:
        run('network', 'create', prefix)
        containers.append(prefix + '-db')
        run('run', '-d', '--name', containers[-1], '--network', prefix, '--network-alias', 'postgres',
            '-e', 'POSTGRES_HOST_AUTH_METHOD=trust', '-e', 'POSTGRES_DB=kadan', '-e', 'POSTGRES_USER=kadan', 'postgres:18')
        for _ in range(60):
            result = subprocess.run(['docker', 'exec', containers[-1], 'pg_isready', '-U', 'kadan'], stdout=subprocess.DEVNULL)
            if result.returncode == 0:
                break
            time.sleep(1)
        run('run', '--rm', '--network', prefix, api, 'alembic', '-c', 'api/alembic.ini', 'upgrade', 'head')
        containers.append(prefix + '-api')
        run('run', '-d', '--name', containers[-1], '--network', prefix, '--network-alias', 'api',
            '-e', 'KADAN_HOST_KEY_FILE=/tmp/test-host-key', api)
        # Fixed test-only value inside this disposable container, never host setup.
        run('exec', containers[-1], 'python', '-c',
            'from pathlib import Path; Path("/tmp/test-host-key").write_text("disposable-test-only")')
        containers.append(prefix + '-frontend')
        run('run', '-d', '--name', containers[-1], '--network', prefix, '-p', '127.0.0.1::8080', frontend)
        port = run('port', containers[-1], '8080/tcp').split(':')[-1]
        for _ in range(60):
            try:
                with urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                    assert json.load(response)['status'] == 'ok'
                break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError('Release frontend/API did not become healthy')
        with urlopen(f'http://127.0.0.1:{port}/version.json') as response:
            assert json.load(response)['commit'] == commit
        with urlopen(f'http://127.0.0.1:{port}/v1/updates') as response:
            status = json.load(response)
            assert status['current'] == commit and status['configured'] is False
        with urlopen(f'http://127.0.0.1:{port}/settings') as response:
            assert '<html' in response.read().decode().lower()
        readiness = json.loads(run('exec', prefix + '-api', 'python', '-c',
            'import urllib.request; r=urllib.request.Request("http://127.0.0.1:8000/internal/updates/ready",'
            'headers={"X-Kadan-Host":"disposable-test-only"}); print(urllib.request.urlopen(r).read().decode())'))
        assert readiness['ready'] and readiness['commit'] == commit
        assert readiness['migrations'] == migration_fingerprint() and readiness['model'] == 'unloaded'
        try:
            urlopen(f'http://127.0.0.1:{port}/internal/updates/ready')
        except HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError('Internal host control was exposed by the frontend')
        print('Paired production frontend/API, migrations and disabled updater smoke passed.')
    finally:
        for name in reversed(containers):
            subprocess.run(['docker', 'rm', '-f', '-v', name], stdout=subprocess.DEVNULL)
        subprocess.run(['docker', 'network', 'rm', prefix], stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
