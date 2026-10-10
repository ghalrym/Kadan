"""Owned subprocess groups and bounded line-oriented inference transport."""
import os
import selectors
import signal
import subprocess
import time

MAX_FRAME = 4096


class LineProtocolError(RuntimeError):
    pass


def check_cancel(event):
    if event is not None and event.is_set():
        raise InterruptedError('Inference cancelled')


class LineProtocolProcess:
    """Single-owner bounded IPC, draining stderr while awaiting every reply."""
    def __init__(self):
        # Construct without side effects. The adapter must own this handle
        # before start() can spawn or perform fallible pipe/selector setup.
        self.process = self.selector = None
        self.buffer = bytearray()
        self.diagnostics = bytearray()
        self.closed = False
        self.io_ready = False

    def start(self, command, *, env=None):
        if self.process is not None or self.closed:
            raise RuntimeError('Inference process handle cannot be reused')
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, bufsize=0, start_new_session=True, env=env)
        self.selector = selectors.DefaultSelector()
        for stream, kind in ((self.process.stdout, 'out'), (self.process.stderr, 'err')):
            os.set_blocking(stream.fileno(), False)
            self.selector.register(stream, selectors.EVENT_READ, kind)
        self.io_ready = True

    def _pump(self, deadline, cancel):
        check_cancel(cancel)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Inference subprocess deadline expired')
        for key, _ in self.selector.select(min(.05, remaining)):
            data = os.read(key.fileobj.fileno(), 4096)
            if not data:
                self.selector.unregister(key.fileobj)
            elif key.data == 'err':
                self.diagnostics.extend(data)
                del self.diagnostics[:-8192]
            else:
                self.buffer.extend(data)
                if len(self.buffer) > MAX_FRAME:
                    raise LineProtocolError('Inference subprocess frame too large')

    def read(self, timeout, cancel=None):
        deadline = time.monotonic() + timeout
        while b'\n' not in self.buffer:
            if not self.selector.get_map():
                raise LineProtocolError('Inference subprocess exited without a complete reply')
            self._pump(deadline, cancel)
        line, _, rest = self.buffer.partition(b'\n')
        self.buffer = bytearray(rest)
        try:
            text = line.decode('ascii')
        except UnicodeDecodeError as error:
            raise LineProtocolError('Non-ASCII inference subprocess frame') from error
        if text.startswith('error '):
            raise LineProtocolError('Inference subprocess failed: ' + self.diagnostics.decode('utf-8', errors='replace')[-1000:])
        return text

    def exchange(self, command, timeout, cancel=None):
        check_cancel(cancel)
        if self.buffer or self.process.poll() is not None:
            raise LineProtocolError('Inference subprocess unavailable or sent unsolicited data')
        data = (command + '\n').encode('ascii')
        if len(data) > 128:
            raise LineProtocolError('Protocol command too large')
        # One small command at a time; no pipelining can fill this pipe.
        self.process.stdin.write(data)
        return self.read(timeout, cancel)

    def finish(self, timeout=5):
        deadline = time.monotonic() + timeout
        while self.process.poll() is None or self.selector.get_map():
            self._pump(deadline, None)
            if self.buffer:
                raise LineProtocolError('Unexpected trailing worker output')
        if self.process.returncode != 0:
            raise LineProtocolError('Inference subprocess exit was not successful')

    def stop(self):
        """Return only after owned child exit; on wait failure retain parent accounting.

        The reviewed inference subprocess never forks. Signal its owned process group
        while the leader is live; never signal a reaped/reusable PID.
        """
        if self.closed:
            return
        if self.process is None:
            self.closed = True
            return
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=5)  # Failure propagates: no release claim.
        finally:
            if self.process.poll() is not None:
                if self.selector is not None:
                    self.selector.close()
                for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                    stream.close()
                self.closed = True
