"""Generate and atomically publish Torch/Diffusers images; never expose partial results."""
import base64
import binascii
from contextlib import ExitStack
from io import BytesIO
import os
from pathlib import Path
import secrets
import shutil
import threading
from uuid import UUID, uuid4

from PIL import Image, UnidentifiedImageError

from api.inference.image import model as qwen_image
from api.inference.decisions.model import clear_failure_frames
from api.inference.resources import ResourceBusy, ResourceCancelled, ResourceExhausted
from api.pydantic_models.media import ImageSet
from api.services.model_downloads import model_manager
from api.services.runtime import RuntimeFailure, runtime_manager

ASPECTS = {'1:1': 'square', '4:3': 'landscape', '3:4': 'portrait', '16:9': 'wide'}


def decode_source(source):
    """Decode an inline image; never fetch caller-controlled network or filesystem URLs."""
    try:
        header, encoded = source.split(',', 1)
        if header not in ('data:image/png;base64', 'data:image/jpeg;base64', 'data:image/webp;base64'):
            raise ValueError('Use a PNG, JPEG or WebP image upload')
        data = base64.b64decode(encoded, validate=True)
        if len(data) > 20 * 1024 ** 2:
            raise ValueError('Source image exceeds 20 MB')
        with Image.open(BytesIO(data)) as source_image:
            if source_image.width * source_image.height > 20_000_000:
                raise ValueError('Source image exceeds 20 million pixels')
            source_image.load()
            return source_image.convert('RGBA')
    except (binascii.Error, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError('Invalid source image') from exc


class ImageManager:
    def __init__(self, root=None, downloads=model_manager, runtime=runtime_manager, backend=None):
        self.root = Path(root or os.environ.get('KADAN_MEDIA_DIR', Path.home() / '.local/share/kadan/media')) / 'images'
        self.downloads, self.runtime, self.backend = downloads, runtime, backend
        self._gate = threading.Lock()
        self.native = None

    def preflight(self):
        try:
            entry, path = self.downloads.get_checkpoint(qwen_image.MODEL_ID)
        except ValueError as exc:
            raise RuntimeFailure('Download Qwen-Image-2.1 in Settings before generating images.', 409) from exc
        if entry.revision != qwen_image.REVISION:
            raise RuntimeFailure('Qwen Image checkpoint revision does not match this runtime', 409)
        return entry, path

    def history(self):
        """Read only atomically published result manifests."""
        records = []
        for manifest in self.root.glob('*/result.json'):
            try:
                records.append(ImageSet.model_validate_json(manifest.read_text()))
            except (OSError, ValueError):
                continue
        return sorted(records, key=lambda item: item.id, reverse=True)

    def file(self, identifier, index):
        """Resolve a published image by UUID and index, never by caller-supplied path."""
        try:
            identifier = str(UUID(identifier))
            result = ImageSet.model_validate_json((self.root / identifier / 'result.json').read_text())
            if not 0 <= index < len(result.seeds):
                raise ValueError('Unknown image index')
            path = self.root / identifier / f'{index}.png'
            if not path.is_file():
                raise ValueError('Missing image file')
            return path
        except (ValueError, OSError) as exc:
            raise RuntimeFailure('Image not found', 404) from exc

    def generate(self, prompt, aspect, count, seed, cancel, source=None):
        # Keep decoded input/output and PNG staging admitted even if a parked
        # pipeline is evicted between its forward and atomic publication.
        with ExitStack() as ownership:
            if self.backend is None:
                # Report a missing checkpoint before requesting any memory.
                try:
                    self.downloads.get_checkpoint(qwen_image.MODEL_ID)
                except ValueError as exc:
                    raise RuntimeFailure('Download Qwen-Image-2.1 in Settings before generating images.', 409) from exc
                try:
                    scratch = self.runtime.ensure_resources().reserve('image:output:' + uuid4().hex,
                        'image', host_bytes=qwen_image.GIB, cancel_event=cancel)
                except (ResourceBusy, ResourceCancelled, ResourceExhausted) as exc:
                    raise RuntimeFailure(str(exc), 503 if isinstance(exc, ResourceExhausted) else 409) from exc
                ownership.callback(scratch.release)
                ownership.enter_context(scratch.lease(cancel))
            try:
                return self._generate(prompt, aspect, count, seed, cancel, source)
            except BaseException as exc:
                clear_failure_frames(exc)
                raise

    def _generate(self, prompt, aspect, count, seed, cancel, source=None):
        """Hold process ownership through generation, cleanup and atomic publication."""
        if not self._gate.acquire(blocking=False):
            raise RuntimeFailure('Image generation is already active', 409)
        stage = None
        published = False
        try:
            image = decode_source(source) if source else None
            try:
                entry, path = self.downloads.get_checkpoint(qwen_image.MODEL_ID)
            except ValueError as exc:
                raise RuntimeFailure('Download Qwen-Image-2.1 in Settings before generating images.', 409) from exc
            if entry.revision != qwen_image.REVISION:
                raise RuntimeFailure('Qwen Image checkpoint revision does not match this runtime', 409)
            seeds = [((seed if seed is not None else secrets.randbits(53)) + index) % (2 ** 53) for index in range(count)]
            if self.backend is not None:
                pictures = self.backend(path, self.runtime.ensure_resources(), prompt, aspect, seeds, cancel, image=image)
            else:
                self._load(cancel)
                pictures = self.native.generate(prompt, aspect, seeds, cancel, image=image)
            if len(pictures) != count:
                raise RuntimeError('Image provider returned an unexpected result count')
            identifier = str(uuid4())
            self.root.mkdir(parents=True, exist_ok=True)
            stage = self.root / ('.' + identifier)
            stage.mkdir()
            for index, picture in enumerate(pictures):
                picture.save(stage / f'{index}.png', format='PNG')
            result = ImageSet(id=identifier, mode='Edit' if source else 'Generate', prompt=prompt,
                aspect=ASPECTS[aspect], seeds=seeds, meta=f'Qwen-Image-2.1 · {count} images',
                urls=[f'/v1/images/{identifier}/files/{index}' for index in range(count)])
            (stage / 'result.json').write_text(result.model_dump_json())
            if cancel.is_set():
                raise ResourceCancelled('Image generation cancelled')
            stage.rename(self.root / identifier)
            published = True
            return result
        except ResourceBusy as exc:
            raise RuntimeFailure(str(exc), 409) from exc
        except ResourceCancelled as exc:
            raise RuntimeFailure(str(exc), 409) from exc
        except (ResourceExhausted, ImportError, OSError, RuntimeError) as exc:
            if isinstance(exc, RuntimeFailure):
                raise
            raise RuntimeFailure(f'Image generation failed: {exc}') from exc
        except ValueError as exc:
            raise RuntimeFailure(str(exc), 422) from exc
        finally:
            try:
                if stage is not None and stage.exists():
                    shutil.rmtree(stage)
            finally:
                try:
                    if not published and self.native is not None:
                        self.native.close()
                finally:
                    self._gate.release()

    def load(self, cancel=None):
        with self._gate:
            return self._load(cancel)

    def _load(self, cancel=None):
        try:
            entry, path = self.downloads.get_checkpoint(qwen_image.MODEL_ID)
        except ValueError as exc:
            raise RuntimeFailure('Download Qwen-Image-2.1 in Settings before generating images.', 409) from exc
        if entry.revision != qwen_image.REVISION:
            raise RuntimeFailure('Qwen Image checkpoint revision does not match this runtime', 409)
        configured = os.environ.get('KADAN_IMAGE_DEVICE', os.environ.get('KADAN_GPU', 'auto'))
        device = 'cuda:' + configured if configured.isdecimal() else configured
        if self.native is not None and (self.native.path != path or self.native.requested != device):
            self.native.close()
            self.native = None
        if self.native is None:
            self.native = qwen_image.NativeImage(path, self.runtime.ensure_resources(), device=device)
        return self.native.load(cancel)

    def offload_to_ram(self, cancel=None):
        if self.native is not None:
            self.native.offload_to_ram(cancel)

    def close(self):
        if self.native is not None:
            self.native.close()
            self.native = None


image_manager = ImageManager()
