"""Resolve opaque media IDs to uploaded local files, never caller-supplied paths."""
import os
from pathlib import Path
import re
import uuid

from collections.abc import AsyncIterator

SUFFIXES = {'image': {'image/png': '.png', 'image/jpeg': '.jpg', 'image/webp': '.webp'},
            'audio': {'audio/wav': '.wav', 'audio/x-wav': '.wav', 'audio/mpeg': '.mp3'},
            'video': {'video/mp4': '.mp4'}}


class VideoInputs:
    def __init__(self, root=None):
        self.root = Path(root or os.getenv('KADAN_VIDEO_INPUT_DIR', '~/.local/share/kadan/video-inputs')).expanduser()

    async def save(self, kind: str, chunks: AsyncIterator[bytes], content_type: str):
        """Stream a bounded upload and publish its opaque ID after the file is complete."""
        suffix = SUFFIXES.get(kind, {}).get(content_type)
        if suffix is None:
            raise ValueError('Unsupported video conditioning media type.')
        directory = self.root / kind
        directory.mkdir(parents=True, exist_ok=True)
        identifier = uuid.uuid4().hex
        temporary = directory / f'.{identifier}.partial'
        try:
            total = 0
            with temporary.open('xb') as target:
                async for chunk in chunks:
                    total += len(chunk)
                    if total > 100 * 1024 * 1024:
                        raise ValueError('Conditioning files must be at most 100 MiB.')
                    target.write(chunk)
            if total == 0:
                raise ValueError('Conditioning file is empty.')
            temporary.replace(directory / f'{identifier}{suffix}')
            return identifier
        finally:
            temporary.unlink(missing_ok=True)

    def resolve(self, identifier: str | None, kind: str):
        if identifier is None:
            return None
        if not re.fullmatch('[0-9a-f]{32}', identifier):
            raise ValueError('Invalid conditioning media ID.')
        for suffix in SUFFIXES[kind].values():
            path = self.root / kind / f'{identifier}{suffix}'
            if path.is_file() and not path.is_symlink():
                return str(path.resolve())
        raise ValueError('Conditioning media was not found. Upload it again.')


video_inputs = VideoInputs()
