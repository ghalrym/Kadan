"""Cancellable checkpoint serialization while the native FIFO awaits preparation."""
import fcntl
from pathlib import Path
import shutil
import tempfile

from api.inference.errors import ResourceCancelled


def prepare_assets(kind: str, checkpoint: Path, destination: Path, cancel, expected=None) -> Path:
    def check():
        if cancel.is_set():
            raise ResourceCancelled('Checkpoint preparation cancelled')

    check()
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / (destination.name + '.lock')).open('a+b') as lock:
        while True:
            check()
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                cancel.wait(.05)
        if destination.exists():
            return destination
        size = checkpoint.stat().st_size if kind == 'whisper' else 0
        if size > 8 * 1024**3 or shutil.disk_usage(destination.parent).free < size + 64 * 1024**2:
            raise ValueError('Insufficient space or oversized native checkpoint conversion')
        staging = Path(tempfile.mkdtemp(prefix='.native-assets-', dir=destination.parent))
        try:
            output = staging / 'output'
            # Optional CPU serialization dependencies are needed only for a first
            # export. Importing them at API startup would allocate unused runtimes.
            if kind == 'tts':
                from api.inference.tts_assets import export
                export(checkpoint, output)
            elif kind == 'whisper':
                from api.inference.whisper_export import export
                from api.inference.whisper_assets import export as export_assets
                report = export(checkpoint, expected, output, cancel)
                check()
                export_assets(report['dimensions']['n_vocab'], output / 'assets')
            else:
                raise ValueError('Unsupported native asset type')
            check()
            output.rename(destination)
        finally:
            shutil.rmtree(staging)
    return destination
