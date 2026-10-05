"""Single-process local model store. No GPU dependencies or import-time writes.

Files stream into staging with pinned upstream size and SHA verification, then a
complete directory is atomically published. Completed directories are immutable.
Run one API worker; an OS lock prevents concurrent download writers even if a
second process is accidentally started. Interrupted staging is removed on retry.
"""
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import threading
from typing import IO, Literal
from typing_extensions import NotRequired, TypedDict
from urllib.request import urlopen

from pydantic import TypeAdapter, ValidationError

from api.services.model_catalog import CATALOG, CatalogEntry, allowed_asset


DEFAULT_CONTEXT_LIMIT = 65536


class BusyError(ValueError):
    pass


class Cancelled(Exception):
    pass


class UpstreamLargeFile(TypedDict):
    size: int
    sha256: str


class UpstreamFile(TypedDict):
    rfilename: str
    size: NotRequired[int]
    blobId: NotRequired[str]
    lfs: NotRequired[UpstreamLargeFile | None]


class UpstreamCheckpoint(TypedDict):
    sha: str
    siblings: list[UpstreamFile]


class ManifestFile(TypedDict):
    name: str
    size: int
    digest: str
    algorithm: Literal['sha256', 'git-sha1']


class CompletionMarker(TypedDict):
    revision: str
    files: list[ManifestFile]


DownloadStatus = Literal['not_downloaded', 'downloading', 'cancelling', 'cancelled', 'failed', 'complete']


@dataclass
class DownloadJob:
    status: DownloadStatus = 'downloading'
    downloaded_bytes: int = 0
    total_bytes: int = 0
    error: str | None = None


class ModelStatus(TypedDict):
    id: str
    repo_id: str
    revision: str
    license: str
    estimated_bytes: int
    status: DownloadStatus
    downloaded_bytes: int
    total_bytes: int
    error: str | None
    context_limit: int | None
    architecture_context_limit: int | None


class ModelsStatus(TypedDict):
    models: list[ModelStatus]
    selected_model_id: str | None


class WeightIndex(TypedDict):
    weight_map: dict[str, str]


def fetch_checkpoint_manifest(entry: CatalogEntry) -> list[ManifestFile]:
    """Fetch and validate allowed assets for a catalog entry's pinned revision.

    Returns size/digest metadata without downloading weights. Network errors and
    invalid upstream metadata propagate; duplicate or missing assets fail closed."""
    metadata_url = f'https://huggingface.co/api/models/{entry.repo_id}/revision/{entry.revision}?blobs=true'
    with urlopen(metadata_url, timeout=30) as response:
        checkpoint_metadata = TypeAdapter(UpstreamCheckpoint).validate_python(json.load(response), strict=True)
    if checkpoint_metadata['sha'] != entry.revision:
        raise ValueError('Upstream revision does not match the pinned catalog')
    manifest: list[ManifestFile] = []
    for upstream_file in checkpoint_metadata['siblings']:
        filename = upstream_file['rfilename']
        if not allowed_asset(filename):
            continue
        large_file = upstream_file.get('lfs')
        size = large_file['size'] if large_file else upstream_file.get('size')
        digest = large_file['sha256'] if large_file else upstream_file.get('blobId')
        digest_length = 64 if large_file else 40
        if size is None or size < 0 or digest is None or not re.fullmatch(f'[0-9a-f]{{{digest_length}}}', digest):
            raise ValueError(f'Upstream omitted valid integrity metadata for {filename}')
        manifest.append({'name': filename, 'size': size, 'digest': digest,
                         'algorithm': 'sha256' if large_file else 'git-sha1'})
    filenames = {item['name'] for item in manifest}
    if len(filenames) != len(manifest):
        raise ValueError('Upstream checkpoint contains duplicate filenames')
    if not {'config.json', 'tokenizer_config.json', 'model.safetensors.index.json'} <= filenames:
        raise ValueError('Checkpoint is missing required configuration or weight index')
    if not any(name.endswith('.safetensors') for name in filenames):
        raise ValueError('Checkpoint has no root safetensors weights')
    return manifest


class ModelManager:
    def __init__(self, root: Path | None = None) -> None:
        """Initialize in-memory coordination for a local checkpoint directory.

        Uses KADAN_MODEL_DIR when root is omitted. Resolves paths but creates no
        files or threads until an operation explicitly needs them."""
        self.root = (root or Path(os.environ.get('KADAN_MODEL_DIR', '~/.local/share/kadan/models')).expanduser()).resolve()
        self._lock = threading.RLock()
        self._jobs: dict[str, DownloadJob] = {}
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._in_use = False

    def _catalog_entry(self, model_id: str) -> CatalogEntry:
        """Resolve an approved model ID; raise ValueError for unknown IDs."""
        if model_id not in CATALOG:
            raise ValueError('Unknown catalog model')
        return CATALOG[model_id]

    def _checkpoint_directory(self, entry: CatalogEntry) -> Path:
        """Return the immutable destination path for this model and revision."""
        return self.root / f'{entry.id}-{entry.revision}'

    def _checkpoint_complete(self, entry: CatalogEntry) -> bool:
        """Check the completion marker and recorded file sizes without rehashing.

        Missing files, invalid markers, symlinks and size mismatches return False.
        Same-size content changes are not detected here; integrity hashes are
        verified during download, not on each status or selection check."""
        path = self._checkpoint_directory(entry)
        try:
            marker = TypeAdapter(CompletionMarker).validate_json((path / 'complete.json').read_text(), strict=True)
            files = marker['files']
            return (marker['revision'] == entry.revision and bool(files)
                    and all(allowed_asset(item['name']) and not (path / item['name']).is_symlink()
                            and (path / item['name']).stat().st_size == item['size'] for item in files))
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def _read_selected_model_id(self) -> str | None:
        """Return the persisted catalog ID, or None for absent/invalid selection."""
        try:
            selection = json.loads((self.root / 'selection.json').read_text())
            model_id = selection['model_id']
            return model_id if isinstance(model_id, str) and model_id in CATALOG else None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _read_context_limits(self) -> dict[str, int | None]:
        """Read per-model context overrides without mutating the store.

        An absent file means no overrides. Invalid values raise ValueError;
        unreadable files propagate OSError rather than silently resetting limits."""
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
        """Return a saved limit (including explicit None), or 64k when unset.

        Validates the model ID and reads persistence under the manager lock."""
        with self._lock:
            self._catalog_entry(model_id)
            return self._read_context_limits().get(model_id, DEFAULT_CONTEXT_LIMIT)

    def architecture_context(self, model_id: str) -> int | None:
        """Read the maximum context from a completed checkpoint's configuration.

        Returns None before download or when no positive maximum is declared.
        Malformed configuration raises ValueError; filesystem errors propagate."""
        entry = self._catalog_entry(model_id)
        if not self._checkpoint_complete(entry):
            return None
        checkpoint_config = json.loads((self._checkpoint_directory(entry) / 'config.json').read_text())
        if not isinstance(checkpoint_config, dict):
            raise ValueError('Checkpoint configuration must be a JSON object')
        text_config = checkpoint_config.get('text_config', checkpoint_config)
        if not isinstance(text_config, dict):
            raise ValueError('Checkpoint text configuration must be a JSON object')
        maximum = text_config.get('max_position_embeddings')
        if type(maximum) is not int or maximum <= 0:
            return None
        return maximum

    def set_context(self, model_id: str, context_limit: int | None) -> None:
        """Atomically persist a token limit; explicit None keeps architecture-maximum mode.

        Rejects invalid/excessive limits and raises BusyError while the runtime
        holds selection. Validation and replacement share the manager lock."""
        with self._lock:
            self._catalog_entry(model_id)
            if self._in_use:
                raise BusyError('Unload the running model before changing context limits')
            if context_limit is not None and (type(context_limit) is not int or not 1 <= context_limit <= 2**31 - 1):
                raise ValueError('Context limit must be a positive integer or null for architecture maximum')
            maximum = self.architecture_context(model_id)
            if context_limit is not None and maximum is not None and context_limit > maximum:
                raise ValueError(f'Context limit exceeds the checkpoint architecture maximum of {maximum}')
            settings = self._read_context_limits()
            settings[model_id] = context_limit
            self.root.mkdir(parents=True, exist_ok=True)
            temporary = self.root / 'context.tmp'
            temporary.write_text(json.dumps(settings))
            temporary.replace(self.root / 'context.json')

    def status(self) -> ModelsStatus:
        """Return catalog, download, selection, and context state under one lock.

        Rechecks checkpoint completeness without modifying jobs. Context-file and
        checkpoint-configuration errors propagate; an unreadable or invalid
        saved selection is represented as no selection."""
        with self._lock:
            context_limits = self._read_context_limits()
            models: list[ModelStatus] = []
            for entry in CATALOG.values():
                job = self._jobs.get(entry.id, DownloadJob(status='not_downloaded'))
                status, error = job.status, job.error
                if self._checkpoint_complete(entry):
                    status, error = 'complete', None
                elif status == 'complete':
                    status = 'failed'
                    error = 'Checkpoint files changed or are missing; inspect model storage'
                models.append({
                    'id': entry.id, 'repo_id': entry.repo_id, 'revision': entry.revision,
                    'license': entry.license, 'estimated_bytes': entry.estimated_bytes,
                    'status': status, 'downloaded_bytes': job.downloaded_bytes,
                    'total_bytes': job.total_bytes, 'error': error,
                    'context_limit': context_limits.get(entry.id, DEFAULT_CONTEXT_LIMIT),
                    'architecture_context_limit': self.architecture_context(entry.id),
                })
            return {'models': models, 'selected_model_id': self._read_selected_model_id()}

    def get_selected(self) -> tuple[CatalogEntry, Path]:
        """Return the selected catalog entry and absolute completed directory.

        Raises ValueError when nothing valid is selected or files are incomplete.
        This read does not reserve the model; use acquire_runtime_model to load."""
        with self._lock:
            selected = self._read_selected_model_id()
            if not selected:
                raise ValueError('No model selected; download and select a checkpoint in Settings')
            entry = self._catalog_entry(selected)
            if not self._checkpoint_complete(entry):
                raise ValueError('Selected model is not completely downloaded')
            return entry, self._checkpoint_directory(entry)

    def acquire_runtime_model(self) -> tuple[CatalogEntry, Path]:
        """Reserve selection for loading and return its entry and local directory.

        Raises BusyError for a second lease and ValueError for invalid selection.
        The caller must release the lease after unload or a failed load."""
        with self._lock:
            if self._in_use:
                raise BusyError('A model is already loaded or loading; unload it first')
            selected = self.get_selected()
            self._in_use = True
            return selected

    def release_runtime_model(self) -> None:
        """Release the runtime selection lease under the lock; repeated calls are safe.

        Does not unload tensors itself. The caller owns runtime cleanup."""
        with self._lock:
            self._in_use = False

    def select(self, model_id: str) -> None:
        """Atomically persist selection of a completely downloaded catalog model.

        Raises BusyError while a runtime lease is held, ValueError for unknown or
        incomplete models, and OSError when persistence fails. Does not load it."""
        with self._lock:
            if self._in_use:
                raise BusyError('Unload the running model before changing selection')
            entry = self._catalog_entry(model_id)
            if not self._checkpoint_complete(entry):
                raise ValueError('Download this model completely before selecting it')
            self.root.mkdir(parents=True, exist_ok=True)
            temporary = self.root / 'selection.tmp'
            temporary.write_text(json.dumps({'model_id': model_id}))
            temporary.replace(self.root / 'selection.json')

    def start(self, model_id: str) -> None:
        """Start one background download after acquiring the store's OS writer lock.

        Rejects unknown IDs, concurrent jobs, and completed checkpoints. Returns
        after thread startup, not download completion; progress/errors use status.
        Failed startup closes the writer lease. Existing model files are immutable."""
        with self._lock:
            entry = self._catalog_entry(model_id)
            if self._thread and self._thread.is_alive():
                raise BusyError('Another download is running; wait or cancel it first')
            if self._checkpoint_complete(entry):
                raise BusyError('This pinned checkpoint is already downloaded')
            self.root.mkdir(parents=True, exist_ok=True)
            lease = (self.root / '.download.lock').open('a')
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close()
                raise BusyError('Another API process owns the download store') from None
            except OSError:
                lease.close()
                raise
            self._cancel.clear()
            self._jobs[model_id] = DownloadJob()
            self._thread = threading.Thread(target=self._download, args=(entry, lease), daemon=True)
            try:
                self._thread.start()
            except Exception:
                lease.close()
                self._jobs[model_id] = DownloadJob(status='failed', error='Could not start download worker')
                raise

    def cancel(self, model_id: str) -> None:
        """Request cooperative cancellation of this model's active download.

        Raises BusyError if no matching job is active. The worker checks between
        reads and before publication; cancellation does not delete complete models."""
        with self._lock:
            self._catalog_entry(model_id)
            job = self._jobs.get(model_id)
            if job is None or job.status not in ('downloading', 'cancelling'):
                raise BusyError('No active download for this model')
            job.status = 'cancelling'
            self._cancel.set()

    def _download(self, entry: CatalogEntry, lease: IO[str]) -> None:
        """Download, verify, and atomically publish one pinned checkpoint.

        Owns the supplied filesystem lease until cleanup. Writes only staging
        files until size/digest/index validation succeeds; job updates and final
        publication use the manager lock so a cancellation accepted before
        publication cannot be recorded as a successful download.
        Records failures in the job, removes staging, and always closes the lease."""
        stage = self.root / f'.{entry.id}.partial'
        try:
            if stage.exists():
                shutil.rmtree(stage)
            stage.mkdir()
            files = fetch_checkpoint_manifest(entry)
            if self._cancel.is_set():
                raise Cancelled()
            total = sum(item['size'] for item in files)
            if shutil.disk_usage(self.root).free < total + 1024**3:
                raise ValueError('Not enough disk space: checkpoint plus 1 GiB reserve required')
            with self._lock:
                self._jobs[entry.id].total_bytes = total
            downloaded = 0
            for item in files:
                if self._cancel.is_set():
                    raise Cancelled()
                digest = hashlib.sha256() if item['algorithm'] == 'sha256' else hashlib.sha1()
                if item['algorithm'] == 'git-sha1':
                    digest.update(f"blob {item['size']}\0".encode())
                size = 0
                url = f"https://huggingface.co/{entry.repo_id}/resolve/{entry.revision}/{item['name']}"
                with urlopen(url, timeout=30) as source, (stage / item['name']).open('wb') as target:
                    while True:
                        if self._cancel.is_set():
                            raise Cancelled()
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > item['size']:
                            raise ValueError(f"Unexpected file size: {item['name']}")
                        target.write(chunk)
                        digest.update(chunk)
                        downloaded += len(chunk)
                        with self._lock:
                            self._jobs[entry.id].downloaded_bytes = downloaded
                if size != item['size'] or digest.hexdigest() != item['digest']:
                    raise ValueError(f"Integrity verification failed: {item['name']}")
            index = TypeAdapter(WeightIndex).validate_json((stage / 'model.safetensors.index.json').read_text(), strict=True)
            if not set(index['weight_map'].values()) <= {item['name'] for item in files}:
                raise ValueError('Weight index references missing or disallowed shards')
            # Commit completion while holding the same lock as cancel(). A
            # cancellation accepted before publication must never become success.
            with self._lock:
                if self._cancel.is_set():
                    raise Cancelled()
                completion_marker: CompletionMarker = {'revision': entry.revision, 'files': files}
                (stage / 'complete.json').write_text(json.dumps(completion_marker))
                destination = self._checkpoint_directory(entry)
                if destination.exists():
                    raise ValueError('Checkpoint directory exists but is incomplete; inspect it manually')
                stage.rename(destination)
                self._jobs[entry.id].status = 'complete'
        except Cancelled:
            with self._lock:
                self._jobs[entry.id].status = 'cancelled'
                self._jobs[entry.id].error = None
        except Exception as exc:
            # No URLs/tokens from network exceptions are exposed in the API.
            if isinstance(exc, ValidationError):
                message = 'Checkpoint metadata is malformed'
            elif isinstance(exc, ValueError):
                message = str(exc)
            else:
                message = f'Download failed ({type(exc).__name__}); check network and disk, then retry'
            with self._lock:
                self._jobs[entry.id].status = 'failed'
                self._jobs[entry.id].error = message
        finally:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)
            lease.close()


model_manager = ModelManager()
