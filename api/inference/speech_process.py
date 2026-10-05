"""Kadan-owned POSIX worker groups; cleanup completes before resource release."""
import os
from pathlib import Path
import signal
import subprocess
import time


class OwnedSpeechProcess:
    def __init__(self):
        self.process = None

    def start(self, command, *, env, log):
        if os.name != 'posix':
            raise RuntimeError('Isolated speech workers require POSIX process-group ownership')
        self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
            stdout=log, stderr=log, env=env, start_new_session=True)

    def poll(self):
        return self.process.poll()

    def _signal(self, value):
        try:
            os.killpg(self.process.pid, value)
        except ProcessLookupError:
            pass

    def _alive(self):
        # Linux zombies hold no model allocations and may await an external
        # subreaper. Ignore only those; live descendants keep accounting owned.
        if Path('/proc/self/stat').exists():
            for path in Path('/proc').glob('[0-9]*/stat'):
                try:
                    fields = path.read_text().rsplit(')', 1)[1].split()
                    if int(fields[2]) == self.process.pid and fields[0] != 'Z':
                        return True
                except (FileNotFoundError, ProcessLookupError):
                    continue
            return False
        try:
            os.killpg(self.process.pid, 0)
            return True
        except ProcessLookupError:
            return False

    def stop(self):
        if self.process is None:
            return
        self._signal(signal.SIGTERM)
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._signal(signal.SIGKILL)
            self.process.wait(timeout=5)
        # Also terminate descendants after the direct worker has already exited.
        self._signal(signal.SIGKILL)
        deadline = time.monotonic() + 5
        while self._alive():
            if time.monotonic() >= deadline:
                raise RuntimeError('Speech descendants have not exited; retaining resource ownership')
            time.sleep(.01)
        self.process = None
