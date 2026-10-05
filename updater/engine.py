"""One durable, owner-triggered update transaction; automatic checks never install."""
import threading

from updater.docker import atomic_json
from updater.releases import compatible, validate_manifest

BUSY = {'checking', 'pulling', 'draining', 'backup', 'restarting', 'verifying', 'restoring', 'rollback'}


class Cancelled(RuntimeError):
    pass


class Engine:
    def __init__(self, root, docker, releases, state):
        self.root, self.docker, self.releases = root, docker, releases
        self.lock = threading.RLock()
        self.operation = threading.Lock()
        self.cancel = threading.Event()
        self.state = state
        validate_manifest(state['current'])
        self.available = None
        self.check_error = None
        if state['phase'] in BUSY or (root / 'control/maintenance').exists():
            self.save('recovery_required', 'An update was interrupted. Inspect the host and run recover locally.')

    def save(self, phase, message, error=None):
        with self.lock:
            self.state.update(phase=phase, message=message, error=error)
            atomic_json(self.root / 'state.json', self.state)

    def status(self, authorized=False):
        with self.lock:
            return {'current': self.state['current']['commit'],
                    'available': self.available['commit'] if self.available else None,
                    'configured': True, 'authorized': authorized, 'phase': self.state['phase'],
                    'message': self.state['message'], 'error': self.state.get('error') or self.check_error,
                    'can_cancel': self.state['phase'] in ('checking', 'pulling', 'draining')}

    def check(self):
        if not self.operation.acquire(blocking=False):
            return
        try:
            candidate = self.releases.latest()
            available = candidate if self.releases.upgrade(self.state['current'], candidate) else None
            with self.lock:
                self.available, self.check_error = available, None
                if self.state['phase'] in ('idle', 'manual_upgrade'):
                    if available and not compatible(self.state['current'], available):
                        self.save('manual_upgrade', 'This release changes stored data. A manual upgrade is required.')
                    else:
                        self.save('idle', 'Update available.' if available else 'Kadan is up to date.')
        except Exception:
            self.check_error = 'Could not verify a tested release. Check network and release/package setup.'
        finally:
            self.operation.release()

    def start(self, commit):
        if not self.operation.acquire(blocking=False):
            raise ValueError('An update or release check is already running')
        try:
            with self.lock:
                if self.state['phase'] not in ('idle', 'failed', 'complete', 'rolled_back'):
                    raise ValueError('Host recovery or setup is required before updating')
                if not self.available or commit != self.available['commit']:
                    raise ValueError('The displayed release changed; refresh before updating')
                if not compatible(self.state['current'], self.available):
                    raise ValueError('This release requires a manual data migration')
                self.cancel.clear()
                self.save('checking', 'Verifying the selected release…')
                threading.Thread(target=self._run, args=(commit,), daemon=True).start()
        except BaseException:
            self.operation.release()
            raise

    def request_cancel(self):
        with self.lock:
            if not self.status()['can_cancel']:
                raise ValueError('Restart has begun; wait for verification or recovery')
            self.cancel.set()
            self.state['message'] = 'Cancel requested; waiting for the current safe boundary…'

    def cancelled(self):
        if self.cancel.is_set():
            raise Cancelled()

    def drain(self):
        self.docker.internal('drain', 'POST')
        def finished():
            self.cancelled()
            status = self.docker.internal('drain')
            if status['state'] in ('error', 'busy'):
                raise RuntimeError('Drain did not finish safely')
            return status['state'] == 'drained'
        self.docker.wait(finished, timeout=150)

    def activate(self, release):
        self.docker.wait(lambda: self.docker.ready(release), timeout=120)
        self.docker.internal('prepare', 'POST')
        self.docker.wait(lambda: self.docker.ready(release, models=True), timeout=600)
        (self.root / 'control/maintenance').unlink(missing_ok=True)
        self.docker.internal('resume', 'POST')

    def safe_stop(self):
        """A lost resume response may hide newly admitted work; always drain again."""
        self.cancel.clear()
        try:
            self.drain()
        except OSError:
            if not self.docker.api_stopped():
                raise RuntimeError('API is unreachable but still running; refusing to stop unknown work') from None
        self.docker.stop_pair()

    def _run(self, commit):
        marked = stopped = replacing = False
        previous = self.state['current']
        try:
            candidate = self.releases.verified(commit)
            if not compatible(previous, candidate) or not self.releases.upgrade(previous, candidate):
                raise ValueError('Release is not a compatible upgrade')
            self.docker.check_volumes()
            if not self.docker.ready(previous):
                raise RuntimeError('Installed pair does not match host state; inspect locally')
            self.cancelled()
            self.save('pulling', 'Downloading both images while Kadan keeps running…')
            self.docker.pull(candidate)
            self.cancelled()
            self.save('draining', 'Waiting for active requests, downloads and model loads to finish…')
            (self.root / 'control/maintenance').touch(mode=0o644)
            marked = True
            self.drain()
            with self.lock:
                self.cancelled()
                self.save('backup', 'Saving a database backup…')
            self.docker.backup()
            self.state['previous'] = previous
            self.state['target'] = candidate
            self.save('restarting', 'Restarting Kadan; this page will reconnect…')
            replacing = True
            self.docker.stop_pair()
            stopped = True
            self.docker.start_pair(candidate)
            self.save('verifying', 'Checking the paired release and database…')
            self.docker.wait(lambda: self.docker.ready(candidate), timeout=120)
            self.save('restoring', 'Restoring the selected model before accepting requests…')
            self.activate(candidate)
            self.state['current'] = candidate
            self.available = None
            self.save('complete', 'Update complete.')
        except Exception as exc:
            try:
                if stopped:
                    self.save('rollback', 'New release did not become ready. Restoring the previous image pair…')
                    # Admission remained closed; only same-schema code is restored.
                    (self.root / 'control/maintenance').touch(mode=0o644)
                    self.safe_stop()
                    self.docker.start_pair(previous)
                    self.activate(previous)
                    self.save('rolled_back', 'Previous version restored. The update failed readiness checks.')
                elif replacing:
                    raise RuntimeError('A container did not stop; refusing forced replacement')
                elif marked:
                    # Request cancellation of the wait without cancelling work/cleanup.
                    try:
                        self.docker.internal('resume', 'POST')
                    except OSError:
                        pass
                    self.docker.wait(lambda: self.docker.internal('drain')['state'] in
                                     ('drained', 'cancelled', 'busy'), timeout=150)
                    self.activate(previous)
                    self.save('failed', 'Update cancelled.' if isinstance(exc, Cancelled)
                              else 'Update stopped before restart; Kadan is still on its previous version.')
                else:
                    self.save('failed', 'Update cancelled.' if isinstance(exc, Cancelled)
                              else 'Update failed before restart. Check network, registry access and disk space.')
            except Exception:
                self.save('recovery_required', 'Update stopped safely. Inspect the host and run recover locally.',
                          'No forced container kill or database downgrade was attempted.')
        finally:
            self.operation.release()

    def recover(self):
        """Explicit local-owner recovery after an interrupted helper; never on startup."""
        if not self.operation.acquire(blocking=False):
            raise ValueError('An update is active')
        try:
            self.cancel.clear()
            previous = self.state['current']
            (self.root / 'control/maintenance').touch(mode=0o644)
            self.safe_stop()
            self.docker.start_pair(previous)
            self.activate(previous)
            self.save('rolled_back', 'Previous release restored by the server owner.')
        except Exception:
            self.save('recovery_required', 'Local recovery did not finish safely; inspect the host before retrying.')
            raise
        finally:
            self.operation.release()
