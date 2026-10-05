"""One-time local setup and host service. Nothing here runs during pip install."""
import argparse
import fcntl
import http.client
import json
import os
from pathlib import Path
import secrets
import socket
import threading
import time

from updater.docker import Docker, atomic_json
from updater.engine import Engine
from updater.releases import Releases
from updater.server import Access, UnixServer, handler


class LocalConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__('localhost', timeout=5)
        self.path = str(path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


def local(root, operation):
    with LocalConnection(root / 'admin.sock') as connection:
        connection.request('POST', '/' + operation)
        result = connection.getresponse()
        if result.status >= 400:
            raise RuntimeError('Local updater command failed')
        return json.load(result)


def bootstrap_install(root, commit, allow_migrations, docker, releases):
    """Resume a durable setup journal, never blindly repeat uncertain migrations.

    A stage is committed before each non-idempotent boundary. Process death needs
    no exception handler to leave a resumable preparing/starting/activating stage.
    An uncertain migration advances only after a read-only schema verification.
    """
    control = root / 'control'
    control.mkdir(mode=0o755, exist_ok=True)
    key = control / 'host-key'
    path = root / 'state.json'
    existing = json.loads(path.read_text()) if path.exists() else None
    retry = (existing and isinstance(existing.get('bootstrap'), dict)
             and existing['current']['commit'] == commit)
    if not commit or not allow_migrations or (existing and not retry):
        raise ValueError('Bootstrap needs an exact release commit and explicit initial migrations; only journaled same-release setup can resume')
    release = releases.verified(commit)
    if existing and existing['current'] != release:
        raise ValueError('The verified release changed; preserve setup evidence and inspect locally')
    if not key.exists():
        key.write_text(secrets.token_urlsafe(32))
        key.chmod(0o644)
    (control / 'maintenance').touch(mode=0o644)
    state = existing or {'current': release, 'previous': None,
                         'bootstrap': {'stage': 'preparing', 'failures': []}}
    journal = state['bootstrap']
    if journal.get('stage') not in ('preparing', 'migrating', 'migrated', 'starting', 'activating'):
        raise ValueError('Unknown bootstrap checkpoint; preserve state and inspect locally')

    def checkpoint(stage, **evidence):
        journal.update(stage=stage, **evidence)
        state.update(phase='bootstrapping', message=f'Bootstrap {stage}; only this exact release may resume.')
        atomic_json(path, state)

    checkpoint(journal['stage'])
    try:
        if journal['stage'] == 'preparing':
            backup = docker.bootstrap_prepare(release)
            checkpoint('migrating', backup=backup)
            docker.bootstrap_migrate()
            checkpoint('migrated')
        elif journal['stage'] == 'migrating':
            # The prior process may have died before, during, or just after the
            # command. Even an unchanged DB revision cannot prove that a migration
            # with external/nontransactional effects is safe to execute again.
            if not docker.bootstrap_migration_complete(release):
                raise RuntimeError('Migration outcome is uncertain. Inspect/finish the exact release migration manually, then rerun bootstrap; it will not be repeated automatically.')
            checkpoint('migrated')
        if journal['stage'] == 'migrated':
            checkpoint('starting')
        if journal['stage'] == 'starting':
            docker.bootstrap_database(release)
            docker.start_pair(release)
            checkpoint('activating')
        if journal['stage'] == 'activating':
            docker.bootstrap_database(release)
            if docker.api_stopped():
                docker.start_pair(release)
            engine = Engine(root, docker, releases, state)
            checkpoint('activating')
            engine.activate(release)
            state.pop('bootstrap')
            state['bootstrap_history'] = journal
            engine.save('idle', 'Installed. Start the updater service and pair this browser.')
    except BaseException as exc:
        journal['failures'].append({'stage': journal['stage'], 'type': type(exc).__name__})
        state['bootstrap'] = journal
        state.update(phase='bootstrapping', message='Setup interrupted or failed. Inspect the checkpoint, then resume this exact bootstrap command.')
        atomic_json(path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/var/lib/kadan-updater'))
    parser.add_argument('command', choices=('bootstrap', 'serve', 'pair', 'recover', 'status'))
    parser.add_argument('--commit')
    parser.add_argument('--allow-initial-migrations', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if os.geteuid() != 0:
        parser.error('Run this host-only command as root; never from the web container')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.stat().st_uid != 0 or root.stat().st_mode & 0o077:
        parser.error('The updater state directory must be root-owned with mode 0700')
    if args.command in ('pair', 'recover'):
        print(json.dumps(local(root, args.command)))
        return
    if args.command == 'status':
        print((root / 'state.json').read_text())
        return
    config_path = root / 'config.json'
    if config_path.stat().st_uid != 0 or config_path.stat().st_mode & 0o077:
        parser.error('config.json must be root-owned with mode 0600')
    config = json.loads(config_path.read_text())
    access = Access(root, config['origin'])
    docker, releases = Docker(root, config), Releases()
    with (root / 'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        control = root / 'control'
        control.mkdir(mode=0o755, exist_ok=True)
        key = control / 'host-key'
        if args.command == 'bootstrap':
            try:
                bootstrap_install(root, args.commit, args.allow_initial_migrations, docker, releases)
            except ValueError as exc:
                parser.error(str(exc))
            return
        if not key.is_file():
            parser.error('Run one-time bootstrap before starting the service')
        state = json.loads((root / 'state.json').read_text())
        if 'bootstrap' in state or state['phase'] == 'bootstrap_failed':
            parser.error('Finish the failed bootstrap before starting the updater service')
        engine = Engine(root, docker, releases, state)
        servers = []
        for path, admin in ((control / 'control.sock', False), (root / 'admin.sock', True)):
            path.unlink(missing_ok=True)
            server = UnixServer(str(path), handler(engine, access, admin))
            path.chmod(0o600 if admin else 0o666)
            servers.append(server)
            threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            while True:
                engine.check()
                time.sleep(6 * 3600)
        finally:
            for server in servers:
                server.shutdown()


if __name__ == '__main__':
    main()
