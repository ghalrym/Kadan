"""Prepare verified native worker files on first load of a normal model download.

Conversion uses CPU serialization/tokenizer tooling only, never model inference.
The request owner waits for preparation and cleanup before another FIFO job runs.
"""
import fcntl
from pathlib import Path
import shutil
import tempfile
import time
from uuid import uuid4

from api.inference.failure_cleanup import clear_failure_frames
from api.inference.resources import ResourceCancelled


def check(cancel):
    if cancel is not None and cancel.is_set():
        raise ResourceCancelled('Native asset preparation cancelled')


def ensure_assets(kind, checkpoint, destination, resources, cancel=None, expected=None):
    checkpoint, destination = (Path(checkpoint), Path(destination))
    check(cancel)
    if destination.exists():
        return destination  # The adapter verifies manifests and all outputs before use.
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Shared across processes; cancellation remains responsive while another load exports.
    with (destination.parent / (destination.name + '.lock')).open('a+b') as lock:
        while True:
            check(cancel)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if cancel is not None:
                    cancel.wait(0.05)
                else:
                    time.sleep(0.05)
        if destination.exists():
            return destination
        size = checkpoint.stat().st_size if kind == 'whisper' else 0
        if size > 8 * 1024 ** 3:
            raise ValueError('Native conversion checkpoint exceeds its size bound')
        required = size + 64 * 1024 ** 2
        if shutil.disk_usage(destination.parent).free < required:
            raise ValueError('Insufficient disk space for native worker assets')
        admission = resources.reserve('native-assets-' + uuid4().hex, 'speech',
            host_bytes=2 * size + 1024**3, cancel_event=cancel)
        staging = None
        try:
            with admission.lease(cancel):
                staging = Path(tempfile.mkdtemp(prefix='.native-assets-', dir=destination.parent))
                output = staging / 'output'
                # Optional conversion dependencies import only inside admitted preparation;
                # ordinary API startup and native execution do not need their Python models.
                if kind == 'tts':
                    from api.inference.tts_assets import export
                    export(checkpoint, output)
                elif kind == 'whisper':
                    from api.inference.whisper_export import export
                    from api.inference.whisper_assets import export as assets
                    report = export(checkpoint, expected, output, cancel)
                    check(cancel)
                    assets(report['dimensions']['n_vocab'], output / 'assets')
                else:
                    raise ValueError('Unsupported native asset kind')
                check(cancel)
                # Same-filesystem publication under the process lock; never overwrite source.
                if destination.exists():
                    raise FileExistsError(destination)
                output.rename(destination)
        except BaseException as error:
            clear_failure_frames(error)
            raise
        finally:
            try:
                if staging is not None:
                    shutil.rmtree(staging)
            finally:
                # Disk cleanup errors propagate with their path, but cannot lose
                # the host reservation after conversion buffers are destroyed.
                admission.release()
    return destination
