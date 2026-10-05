"""Installed build identity and the deliberately conservative data boundary."""
import hashlib
import os
from pathlib import Path

# Bump for incompatible checkpoint selection/context/cache or other persisted
# formats. Website updates and rollback must not cross this boundary.
DATA_EPOCH = 1


def migration_fingerprint():
    root = Path(__file__).parents[1] / 'migrations'
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes())
    return digest.hexdigest()


def identity():
    return {'commit': os.environ.get('KADAN_COMMIT', 'development'),
            'data_epoch': DATA_EPOCH, 'migrations': migration_fingerprint()}
