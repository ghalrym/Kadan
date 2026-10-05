"""Fixed Compose project/template. Browser data never becomes command arguments.

Postgres and external volume identities are not replaced during routine updates.
Only API/frontend receive SIGTERM; no timeout ever escalates to SIGKILL.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.request import Request, ProxyHandler, build_opener

from updater.releases import validate_manifest

# Host-control credentials and loopback readiness never traverse an environment proxy.
local_open = build_opener(ProxyHandler({})).open


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.chmod(0o600)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Docker:
    def __init__(self, root, config):
        self.root, self.config = Path(root), config
        for key in ('project', 'model_volume', 'postgres_volume', 'hf_volume'):
            if not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}', config[key]):
                raise ValueError('Invalid fixed project/volume identity')
        for key in ('POSTGRES_DB', 'POSTGRES_USER'):
            if not re.fullmatch('[a-zA-Z_][a-zA-Z0-9_]{0,62}', config[key]):
                raise ValueError('Database name/user must be simple PostgreSQL identifiers')
        if len({config[key] for key in ('model_volume', 'postgres_volume', 'hf_volume')}) != 3:
            raise ValueError('Storage volumes must be distinct')
        self.compose = ['docker', 'compose', '--project-name', config['project'],
                        '--file', str(self.root / 'compose.json')]

    def run(self, args, timeout=120, **kwargs):
        # Do not pass ambient Compose files, projects or env files into commands.
        environment = {key: value for key, value in os.environ.items() if not key.startswith('COMPOSE_')}
        return subprocess.run(args, check=True, timeout=timeout, env=environment,
                              stderr=subprocess.PIPE, **kwargs)

    def render(self, release):
        release = validate_manifest(release)
        database = {key: self.config[key] for key in ('POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD')}
        database.update(POSTGRES_HOST='postgres', POSTGRES_PORT='5432')
        api = {'image': release['api'], 'init': True, 'stop_grace_period': '120s',
               'environment': {**database, 'KADAN_MODEL_DIR': '/var/lib/kadan/models',
                   'HF_HOME': '/var/lib/kadan/huggingface', 'KADAN_GPU': self.config.get('gpu', '0'),
                   'KADAN_UPDATER_SOCKET': '/run/kadan-updater/control.sock',
                   'KADAN_HOST_KEY_FILE': '/run/kadan-updater/host-key',
                   'KADAN_MAINTENANCE_FILE': '/run/kadan-updater/maintenance'},
               'volumes': ['model_data:/var/lib/kadan/models', 'hf_data:/var/lib/kadan/huggingface',
                           str(self.root / 'control') + ':/run/kadan-updater:ro'],
               'ports': ['127.0.0.1:8000:8000'],
               'deploy': {'resources': {'reservations': {'devices': [
                   {'driver': 'nvidia', 'count': 'all', 'capabilities': ['gpu']}]}}}}
        value = {'services': {
            'postgres': {'image': 'postgres:18', 'environment': database,
                         'volumes': ['postgres_data:/var/lib/postgresql']},
            'api': api,
            'frontend': {'image': release['frontend'], 'init': True, 'stop_grace_period': '120s',
                         'ports': ['127.0.0.1:5173:8080']}},
            'volumes': {key: {'external': True, 'name': self.config[name]} for key, name in
                        (('model_data', 'model_volume'), ('postgres_data', 'postgres_volume'), ('hf_data', 'hf_volume'))}}
        def literal(item):
            if isinstance(item, str):
                return item.replace('$', '$$')
            if isinstance(item, list):
                return [literal(child) for child in item]
            if isinstance(item, dict):
                return {key: literal(child) for key, child in item.items()}
            return item
        # Compose interpolates even JSON strings. Existing database passwords
        # and paths must remain literal rather than read host environment values.
        atomic_json(self.root / 'compose.json', literal(value))

    def pull(self, release):
        validate_manifest(release)
        for image in (release['api'], release['frontend']):
            self.run(['docker', 'pull', image], timeout=1800, stdout=subprocess.DEVNULL)
            inspect = json.loads(self.run(['docker', 'image', 'inspect', image], stdout=subprocess.PIPE).stdout)[0]
            if inspect['Config'].get('Labels', {}).get('org.opencontainers.image.revision') != release['commit']:
                raise RuntimeError('Image revision does not match the paired release')

    def check_volumes(self):
        for key in ('model_volume', 'postgres_volume', 'hf_volume'):
            self.run(['docker', 'volume', 'inspect', self.config[key]], stdout=subprocess.DEVNULL)
        # Existing containers must use the same project and volume identities.
        for service, volume in (('api', 'model_volume'), ('postgres', 'postgres_volume')):
            ids = self.run(['docker', 'ps', '-aq', '--filter', f'label=com.docker.compose.project={self.config["project"]}',
                            '--filter', f'label=com.docker.compose.service={service}'], stdout=subprocess.PIPE).stdout.decode().split()
            for identifier in ids:
                value = json.loads(self.run(['docker', 'inspect', identifier], stdout=subprocess.PIPE).stdout)[0]
                names = {mount.get('Name') for mount in value['Mounts']}
                if self.config[volume] not in names:
                    raise RuntimeError('Existing project uses different storage; refusing adoption')

    def bootstrap_prepare(self, release):
        self.check_volumes()
        for service in ('api', 'frontend', 'migrate'):
            running = self.run(['docker', 'ps', '-q', '--filter', f'label=com.docker.compose.project={self.config["project"]}',
                                '--filter', f'label=com.docker.compose.service={service}'], stdout=subprocess.PIPE).stdout.strip()
            if running:
                raise RuntimeError('Finish active work and stop the legacy app before one-time adoption')
        self.pull(release)
        legacy = self.run(['docker', 'ps', '-aq', '--filter', f'label=com.docker.compose.project={self.config["project"]}',
                           '--filter', 'label=com.docker.compose.service=api'], stdout=subprocess.PIPE).stdout.decode().split()
        if len(legacy) > 1:
            raise RuntimeError('Multiple legacy API containers require manual cache migration')
        if legacy:
            value = json.loads(self.run(['docker', 'inspect', legacy[0]], stdout=subprocess.PIPE).stdout)[0]
            if (self.config['hf_volume'] not in {mount.get('Name') for mount in value['Mounts']}
                    and self.config.get('legacy_cache_already_preserved') is not True):
                cache_variables = ('HF_HOME=', 'HF_HUB_CACHE=', 'HUGGINGFACE_HUB_CACHE=', 'TRANSFORMERS_CACHE=')
                if any(item.startswith(cache_variables) for item in value['Config'].get('Env', [])):
                    raise RuntimeError('Custom legacy cache paths require explicit owner migration before adoption')
                # Keep the old container until its cache is copied. A custom
                # HF_HOME/cache layout requires explicit owner migration.
                transfer = self.root / 'legacy-cache'
                transfer.mkdir(mode=0o700, exist_ok=True)
                try:
                    self.run(['docker', 'cp', legacy[0] + ':/root/.cache/huggingface', str(transfer)],
                             stdout=subprocess.DEVNULL)
                    self.run(['docker', 'run', '--rm', '--network=none', '--entrypoint=python',
                              '--mount', f'type=bind,src={transfer},dst=/source,readonly',
                              '--mount', f'type=volume,src={self.config["hf_volume"]},dst=/target', release['api'],
                              '-c', 'import shutil; shutil.copytree("/source/huggingface", "/target", dirs_exist_ok=True, symlinks=True)'],
                             timeout=600, stdout=subprocess.DEVNULL)
                except subprocess.CalledProcessError:
                    raise RuntimeError('Preserve the legacy HF cache manually before adoption; see setup instructions') from None
                finally:
                    shutil.rmtree(transfer)
        self.render(release)
        self.run(self.compose + ['up', '-d', '--no-recreate', '--no-deps', 'postgres'], stdout=subprocess.DEVNULL)
        def database_ready():
            try:
                self.run(self.compose + ['exec', '-T', 'postgres', 'pg_isready', '-U', self.config['POSTGRES_USER'],
                                         '-d', self.config['POSTGRES_DB']], stdout=subprocess.DEVNULL)
                return True
            except subprocess.CalledProcessError:
                return False
        self.wait(database_ready)
        return self.backup()

    def bootstrap_migrate(self):
        self.run(self.compose + ['run', '--rm', '--no-deps', 'api', 'alembic', '-c', 'api/alembic.ini', 'upgrade', 'head'],
                 timeout=300, stdout=subprocess.DEVNULL)

    def bootstrap_migration_complete(self, release):
        """Read the DB revision after interruption; do not execute upgrade again."""
        self.check_volumes()
        for service in ('api', 'frontend', 'migrate'):
            running = self.run(['docker', 'ps', '-q', '--filter', f'label=com.docker.compose.project={self.config["project"]}',
                                '--filter', f'label=com.docker.compose.service={service}'], stdout=subprocess.PIPE).stdout.strip()
            if running:
                raise RuntimeError('A setup/app container is still running; inspect it before migration recovery')
        self.render(release)
        script = (
            'import json; from alembic.config import Config; from alembic.script import ScriptDirectory; '
            'from sqlalchemy import inspect, text; from api.database import get_engine; '
            'engine=get_engine(); expected=ScriptDirectory.from_config(Config("api/alembic.ini")).get_heads(); '
            'connection=engine.connect(); '
            'actual=list(connection.execute(text("SELECT version_num FROM alembic_version")).scalars()) '
            'if inspect(connection).has_table("alembic_version") else []; '
            'print(json.dumps(sorted(actual)==sorted(expected))); connection.close()'
        )
        result = self.run(self.compose + ['run', '--rm', '--no-deps', 'api', 'python', '-c', script],
                          timeout=120, stdout=subprocess.PIPE)
        return json.loads(result.stdout) is True

    def internal(self, operation, method='GET'):
        if operation not in ('drain', 'resume', 'ready', 'prepare'):
            raise ValueError('Unknown internal operation')
        key = (self.root / 'control/host-key').read_text().strip()
        request = Request('http://127.0.0.1:8000/internal/updates/' + operation,
                          method=method, headers={'X-Kadan-Host': key})
        with local_open(request, timeout=5) as response:
            return json.load(response)

    def wait(self, predicate, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.5)
        raise TimeoutError('Operation exceeded its deadline; inspect the host updater.')

    def backup(self):
        destination = self.root / 'backups'
        destination.mkdir(mode=0o700, exist_ok=True)
        path = destination / (str(time.time_ns()) + '.dump')
        try:
            with path.open('xb') as output:
                path.chmod(0o600)
                self.run(self.compose + ['exec', '-T', 'postgres', 'pg_dump', '-Fc',
                    '-U', self.config['POSTGRES_USER'], '-d', self.config['POSTGRES_DB']], stdout=output)
                output.flush()
                os.fsync(output.fileno())
            if not path.stat().st_size:
                raise RuntimeError('Database backup was empty')
            return str(path)
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    def stop_pair(self):
        for service in ('frontend', 'api'):
            identifiers = self.run(self.compose + ['ps', '-q', service], stdout=subprocess.PIPE).stdout.decode().split()
            for identifier in identifiers:
                if not re.fullmatch('[0-9a-f]{12,64}', identifier):
                    raise RuntimeError('Unexpected container identity')
                self.run(['docker', 'kill', '--signal=SIGTERM', identifier], stdout=subprocess.DEVNULL)
                self.wait(lambda: self.run(['docker', 'inspect', '--format', '{{.State.Running}}', identifier],
                                          stdout=subprocess.PIPE).stdout.strip() == b'false')

    def api_stopped(self):
        ids = self.run(self.compose + ['ps', '-q', 'api'], stdout=subprocess.PIPE).stdout.split()
        return not ids

    def start_pair(self, release):
        self.render(release)
        self.run(self.compose + ['up', '--detach', '--no-deps', '--pull', 'never', 'api', 'frontend'],
                 stdout=subprocess.DEVNULL)

    def ready(self, release, models=False):
        try:
            api = self.internal('ready')
            with local_open('http://127.0.0.1:5173/version.json', timeout=5) as response:
                frontend = json.load(response)
            return (api.get('ready') is True and api['commit'] == frontend['commit'] == release['commit']
                    and all(api[key] == release[key] for key in ('migrations', 'data_epoch'))
                    and (not models or api.get('model') in ('ready', 'offloaded', 'unloaded')))
        except (OSError, ValueError, KeyError):
            return False
