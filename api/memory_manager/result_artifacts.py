"""Bounded, queue-owned disk transport for large synchronous image results.

Redis retains only a digest/length record. Files share the queue lock's ownership
and retention period; a full spool rejects new results instead of growing Redis.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import time

from api.inference.errors import InferenceFailure


class ResultArtifacts:
    def __init__(self, root, retention, *, max_file=128 * 1024**2, max_bytes=256 * 1024**2):
        self.root = Path(root)
        self.retention, self.max_file, self.max_bytes = retention, max_file, max_bytes

    def path(self, job_id):
        if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
            raise ValueError('Invalid result artifact identity')
        return self.root / (job_id + '.json')

    def store(self, job_id, encoded):
        data = encoded.encode('utf-8')
        if len(data) > self.max_file:
            raise ValueError('Image result exceeds the artifact size limit')
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError('Invalid result artifact directory')
        used = self.prune()
        if used + len(data) > self.max_bytes:
            raise ValueError('Image result artifact storage is full')
        path = self.path(job_id)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

    def prune(self):
        if not self.root.exists():
            return 0
        if self.root.is_symlink():
            raise ValueError('Invalid result artifact directory')
        used = 0
        cutoff = time.time() - self.retention
        for path in self.root.iterdir():
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError('Invalid result artifact entry')
            if info.st_mtime < cutoff:
                path.unlink()
            else:
                used += info.st_size
        return used

    def discard(self, job_id):
        self.path(job_id).unlink(missing_ok=True)

    def read(self, job_id, metadata):
        try:
            expected = metadata['bytes']
            if type(expected) is not int or not 0 < expected <= self.max_file:
                raise ValueError('Invalid result size')
            descriptor = os.open(self.path(job_id), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, 'rb') as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size != expected:
                    raise ValueError('Invalid result file')
                data = stream.read(expected + 1)
            if len(data) != expected or hashlib.sha256(data).hexdigest() != metadata['sha256']:
                raise ValueError('Result integrity mismatch')
            return json.loads(data)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise InferenceFailure('Image result artifact unavailable or invalid.', 503) from error
