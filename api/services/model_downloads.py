"""Single-process local model store. No GPU dependencies or import-time writes.

Files stream into staging with pinned upstream size and SHA verification, then a
complete directory is atomically published. Completed directories are immutable.
Run one API worker; an OS lock prevents concurrent download writers even if a
second process is accidentally started. Interrupted staging is removed on retry.
"""
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
from urllib.request import urlopen

from api.services.model_catalog import CATALOG, CatalogEntry, allowed_asset


class BusyError(ValueError):
    pass


class Cancelled(Exception):
    pass


def _json_url(url: str):
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def _manifest(entry: CatalogEntry) -> list[dict]:
    data = _json_url(
        f'https://huggingface.co/api/models/{entry.repo_id}/revision/{entry.revision}?blobs=true'
    )
    if data.get('sha') != entry.revision:
        raise ValueError('Upstream revision does not match the pinned catalog')
    files = []
    for item in data['siblings']:
        name = item['rfilename']
        if not allowed_asset(name):
            continue
        lfs = item.get('lfs')
        size = lfs['size'] if lfs else item.get('size')
        digest = lfs.get('sha256') if lfs else item.get('blobId')
        if not isinstance(size, int) or size < 0 or not digest:
            raise ValueError(f'Upstream omitted integrity metadata for {name}')
        files.append({'name': name, 'size': size, 'digest': digest,
                      'algorithm': 'sha256' if lfs else 'git-sha1'})
    names = {item['name'] for item in files}
    if not {'config.json', 'tokenizer_config.json', 'model.safetensors.index.json'} <= names:
        raise ValueError('Checkpoint is missing required configuration or weight index')
    if not any(name.endswith('.safetensors') for name in names):
        raise ValueError('Checkpoint has no root safetensors weights')
    return files


class ModelManager:
    def __init__(self, root: Path | None = None):
        self.root = (root or Path(os.environ.get('KADAN_MODEL_DIR', '~/.local/share/kadan/models')).expanduser()).resolve()
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._in_use = False

    def _entry(self, model_id: str) -> CatalogEntry:
        if model_id not in CATALOG:
            raise ValueError('Unknown catalog model')
        return CATALOG[model_id]

    def close(self):
        """Stop a download at the next read boundary during API shutdown."""
        self._cancel.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=35)

    def _path(self, entry: CatalogEntry) -> Path:
        return self.root / f'{entry.id}-{entry.revision}'

    def _complete(self, entry: CatalogEntry) -> bool:
        path = self._path(entry)
        try:
            marker = json.loads((path / 'complete.json').read_text())
            files = marker['files']
            return (marker['revision'] == entry.revision and bool(files)
                    and all(allowed_asset(item['name']) and not (path / item['name']).is_symlink()
                            and (path / item['name']).stat().st_size == item['size'] for item in files))
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def _selected_id(self) -> str | None:
        try:
            return json.loads((self.root / 'selection.json').read_text())['model_id']
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _context_settings(self) -> dict:
        try:
            value = json.loads((self.root / 'context.json').read_text())
        except FileNotFoundError:
            return {}
        if not isinstance(value, dict) or any(
            key not in CATALOG or (limit is not None and (type(limit) is not int or not 1 <= limit <= 2**31 - 1))
            for key, limit in value.items()
        ):
            raise ValueError('Invalid persisted context settings; repair context.json before loading')
        return value

    def configured_context(self, model_id: str) -> int | None:
        with self._lock:
            self._entry(model_id)
            return self._context_settings().get(model_id)

    def architecture_context(self, model_id: str) -> int | None:
        entry = self._entry(model_id)
        if not self._complete(entry):
            return None
        config = json.loads((self._path(entry) / 'config.json').read_text())
        maximum = config.get('text_config', config).get('max_position_embeddings')
        if type(maximum) is not int or maximum <= 0:
            return None
        return maximum

    def set_context(self, model_id: str, context_limit: int | None):
        with self._lock:
            self._entry(model_id)
            if self._in_use:
                raise BusyError('Unload the running model before changing context limits')
            if context_limit is not None and (type(context_limit) is not int or not 1 <= context_limit <= 2**31 - 1):
                raise ValueError('Context limit must be a positive integer or null for architecture maximum')
            maximum = self.architecture_context(model_id)
            if context_limit is not None and maximum is not None and context_limit > maximum:
                raise ValueError(f'Context limit exceeds the checkpoint architecture maximum of {maximum}')
            settings = self._context_settings()
            settings[model_id] = context_limit
            self.root.mkdir(parents=True, exist_ok=True)
            temporary = self.root / 'context.tmp'
            temporary.write_text(json.dumps(settings))
            temporary.replace(self.root / 'context.json')

    def status(self) -> dict:
        with self._lock:
            rows = []
            for entry in CATALOG.values():
                state = dict(self._jobs.get(entry.id, {}))
                if self._complete(entry):
                    state.update(status='complete', error=None)
                elif state.get('status') == 'complete':
                    state = {'status': 'failed', 'error': 'Checkpoint files changed or are missing; inspect model storage'}
                elif not state:
                    state = {'status': 'not_downloaded'}
                rows.append({**asdict(entry), 'downloaded_bytes': 0, 'total_bytes': 0,
                             'error': None, **state, 'context_limit': self.configured_context(entry.id),
                             'architecture_context_limit': self.architecture_context(entry.id)})
            return {'models': rows, 'selected_model_id': self._selected_id()}

    def get_selected(self) -> tuple[CatalogEntry, Path]:
        with self._lock:
            selected = self._selected_id()
            if not selected:
                raise ValueError('No model selected; download and select a checkpoint in Settings')
            entry = self._entry(selected)
            if not self._complete(entry):
                raise ValueError('Selected model is not completely downloaded')
            return entry, self._path(entry)

    def get_runtime_model(self) -> tuple[str, Path]:
        entry, path = self.get_selected()
        return entry.id, path

    def acquire_runtime_model(self) -> tuple[CatalogEntry, Path]:
        with self._lock:
            if self._in_use:
                raise BusyError('A model is already loaded or loading; unload it first')
            selected = self.get_selected()
            self._in_use = True
            return selected

    def release_runtime_model(self):
        with self._lock:
            self._in_use = False

    def select(self, model_id: str):
        with self._lock:
            if self._in_use:
                raise BusyError('Unload the running model before changing selection')
            entry = self._entry(model_id)
            if not self._complete(entry):
                raise ValueError('Download this model completely before selecting it')
            self.root.mkdir(parents=True, exist_ok=True)
            temporary = self.root / 'selection.tmp'
            temporary.write_text(json.dumps({'model_id': model_id}))
            temporary.replace(self.root / 'selection.json')

    def start(self, model_id: str):
        with self._lock:
            entry = self._entry(model_id)
            if self._thread and self._thread.is_alive():
                raise BusyError('Another download is running; wait or cancel it first')
            if self._complete(entry):
                raise BusyError('This pinned checkpoint is already downloaded')
            self.root.mkdir(parents=True, exist_ok=True)
            lease = (self.root / '.download.lock').open('a')
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close()
                raise BusyError('Another API process owns the download store') from None
            self._cancel.clear()
            self._jobs[model_id] = {'status': 'downloading', 'downloaded_bytes': 0,
                                    'total_bytes': 0, 'error': None}
            self._thread = threading.Thread(target=self._download, args=(entry, lease), daemon=True)
            self._thread.start()

    def cancel(self, model_id: str):
        with self._lock:
            self._entry(model_id)
            if self._jobs.get(model_id, {}).get('status') not in ('downloading', 'cancelling'):
                raise BusyError('No active download for this model')
            self._jobs[model_id]['status'] = 'cancelling'
            self._cancel.set()

    def _update(self, entry: CatalogEntry, **values):
        with self._lock:
            self._jobs[entry.id].update(values)

    def _check_cancel(self):
        if self._cancel.is_set():
            raise Cancelled()

    def _download(self, entry: CatalogEntry, lease):
        stage = self.root / f'.{entry.id}.partial'
        try:
            if stage.exists():
                shutil.rmtree(stage)
            stage.mkdir()
            files = _manifest(entry)
            self._check_cancel()
            total = sum(item['size'] for item in files)
            if shutil.disk_usage(self.root).free < total + 1024**3:
                raise ValueError('Not enough disk space: checkpoint plus 1 GiB reserve required')
            self._update(entry, total_bytes=total)
            downloaded = 0
            for item in files:
                self._check_cancel()
                digest = hashlib.sha256() if item['algorithm'] == 'sha256' else hashlib.sha1()
                if item['algorithm'] == 'git-sha1':
                    digest.update(f"blob {item['size']}\0".encode())
                size = 0
                url = f"https://huggingface.co/{entry.repo_id}/resolve/{entry.revision}/{item['name']}"
                with urlopen(url, timeout=30) as source, (stage / item['name']).open('wb') as target:
                    while True:
                        self._check_cancel()
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > item['size']:
                            raise ValueError(f"Unexpected file size: {item['name']}")
                        target.write(chunk)
                        digest.update(chunk)
                        downloaded += len(chunk)
                        self._update(entry, downloaded_bytes=downloaded)
                if size != item['size'] or digest.hexdigest() != item['digest']:
                    raise ValueError(f"Integrity verification failed: {item['name']}")
            index = json.loads((stage / 'model.safetensors.index.json').read_text())
            if not set(index['weight_map'].values()) <= {item['name'] for item in files}:
                raise ValueError('Weight index references missing or disallowed shards')
            self._check_cancel()
            (stage / 'complete.json').write_text(json.dumps({'revision': entry.revision, 'files': files}))
            destination = self._path(entry)
            if destination.exists():
                raise ValueError('Checkpoint directory exists but is incomplete; inspect it manually')
            stage.rename(destination)
            self._update(entry, status='complete')
        except Cancelled:
            self._update(entry, status='cancelled', error=None)
        except Exception as exc:
            # No URLs/tokens from network exceptions are exposed in the API.
            message = str(exc) if isinstance(exc, ValueError) else f'Download failed ({type(exc).__name__}); check network and disk, then retry'
            self._update(entry, status='failed', error=message)
        finally:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)
            lease.close()


model_manager = ModelManager()
