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
            if not args.commit or not args.allow_initial_migrations or (root / 'state.json').exists():
                parser.error('Bootstrap needs an exact release commit, explicit initial migrations, and no existing state')
            release = releases.verified(args.commit)
            # Generated only when the owner explicitly runs bootstrap on their host.
            key.write_text(secrets.token_urlsafe(32))
            key.chmod(0o644)
            (control / 'maintenance').touch(mode=0o644)
            state = {'current': release, 'previous': None, 'phase': 'recovery_required',
                     'message': 'Initial setup is incomplete; finish bootstrap or inspect locally.'}
            atomic_json(root / 'state.json', state)
            docker.bootstrap(release)
            engine = Engine(root, docker, releases, state)
            engine.activate(release)
            engine.save('idle', 'Installed. Start the updater service and pair this browser.')
            return
        if not key.is_file():
            parser.error('Run one-time bootstrap before starting the service')
        state = json.loads((root / 'state.json').read_text())
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
