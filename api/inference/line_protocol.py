"""Owned subprocess groups and bounded line-oriented inference transport."""
import logging
import os
import re
import selectors
import signal
import subprocess
import time

from api.inference.progress import report

MAX_FRAME = 4096
log = logging.getLogger(__name__)
STAGE = re.compile(rb'(layer_completed|block_loaded|block_completed|refiner_completed|publish|tokenized|denoise_completed|audio_decoded|denoiser_block_loaded|tokenizer_loaded|text_loaded|denoiser_loaded|vae_loaded|worker_ready|generate_started|text_layer|conditioning|block|step_completed|vae_up_block) ([0-9]{1,6})')
WEIGHT_BYTES_STAGE = re.compile(rb'(checkpoint_tensor_bytes|expanded_f32_inventory_bytes|nonfloating_inventory_bytes|gpu_[01]_weight_(?:planned|allocated)_bytes|text_checkpoint_read_bytes|vae_checkpoint_read_bytes) ([0-9]{1,12})')


class LineProtocolError(RuntimeError):
    def __init__(self, message, *, diagnostics=b''):
        super().__init__(message)
        # Retain bounded local debugging context without exposing stderr in API errors.
        self.diagnostics = diagnostics[-8192:]



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
        self.stage_buffer = bytearray()
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

    def _capture_diagnostics(self, data):
        self.diagnostics.extend(data)
        del self.diagnostics[:-8192]
        self.stage_buffer.extend(data)
        while b'\n' in self.stage_buffer:
            line, _, rest = self.stage_buffer.partition(b'\n')
            self.stage_buffer = bytearray(rest)
            match = STAGE.fullmatch(line) or WEIGHT_BYTES_STAGE.fullmatch(line)
            if match:
                report(match[1].decode(), int(match[2]))
                # Receipt time, not an exact device completion timestamp. Never log
                # arbitrary worker text as a public stage or include prompt contents.
                log.info('worker_stage pid=%s observed_unix_ns=%s stage=%s index=%s',
                         self.process.pid, time.time_ns(), match[1].decode(), match[2].decode())
        if len(self.stage_buffer) > 8192:
            self.stage_buffer.clear()

    def _failure(self, message, error_type=LineProtocolError):
        # Read only already-available stderr, never wait for a failing child.
        if self.process is not None and self.process.stderr is not None:
            try:
                self._capture_diagnostics(os.read(self.process.stderr.fileno(), 8192))
            except (BlockingIOError, ValueError):
                pass
        error = error_type(message)
        error.diagnostics = bytes(self.diagnostics)
        return error

    def _pump(self, deadline, cancel):
        check_cancel(cancel)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise self._failure('Inference subprocess deadline expired', TimeoutError)
        for key, _ in self.selector.select(min(.05, remaining)):
            data = os.read(key.fileobj.fileno(), 4096)
            if not data:
                self.selector.unregister(key.fileobj)
            elif key.data == 'err':
                self._capture_diagnostics(data)
            else:
                self.buffer.extend(data)
                if len(self.buffer) > MAX_FRAME:
                    raise self._failure('Inference subprocess frame too large')

    def read(self, timeout, cancel=None):
        deadline = time.monotonic() + timeout
        while b'\n' not in self.buffer:
            if not self.selector.get_map():
                raise self._failure('Inference subprocess exited without a complete reply')
            self._pump(deadline, cancel)
        line, _, rest = self.buffer.partition(b'\n')
        self.buffer = bytearray(rest)
        try:
            text = line.decode('ascii')
        except UnicodeDecodeError as error:
            raise self._failure('Non-ASCII inference subprocess frame') from error
        if text == 'error' or text.startswith('error '):
            safe_codes = {f'image_{stage}_failed' for stage in
                          ('startup', 'load', 'generate', 'park', 'resume', 'cleanup')}
            code = text.partition(' ')[2]
            message = 'Inference subprocess failed'
            if code in safe_codes:
                message += ': ' + code
            raise self._failure(message)
        return text

    def exchange(self, command, timeout, cancel=None):
        check_cancel(cancel)
        if self.buffer or self.process.poll() is not None:
            raise self._failure('Inference subprocess unavailable or sent unsolicited data')
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
                raise self._failure('Unexpected trailing worker output')
        if self.process.returncode != 0:
            raise self._failure('Inference subprocess exit was not successful')

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
