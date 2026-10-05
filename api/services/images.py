"""Generate and atomically publish native images; never expose partial results."""
import asyncio
import base64
import binascii
from io import BytesIO
import os
from pathlib import Path
import secrets
import shutil
import threading
from uuid import UUID, uuid4

from PIL import Image, UnidentifiedImageError

from api.inference import qwen_image, flux_klein, native_image
from api.inference.resources import ResourceBusy, ResourceCancelled, ResourceExhausted
from api.pydantic_models.media import ImageSet
from api.services.model_downloads import model_manager
from api.services.runtime import RuntimeFailure, runtime_manager, finish_cleanup

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

    def generate(self, prompt, aspect, count, seed, cancel, source=None, model=None):
        """Hold process ownership through generation, cleanup and atomic publication."""
        if not self._gate.acquire(blocking=False):
            raise RuntimeFailure('Image generation is already active', 409)
        stage = None
        try:
            image = decode_source(source) if source else None
            model = model or self.downloads.selected_image_model_id() or qwen_image.MODEL_ID
            recipe = ({qwen_image.MODEL_ID: qwen_image.RECIPE} | flux_klein.RECIPES).get(model)
            if recipe is None:
                raise RuntimeFailure('Unsupported image model', 422)
            try:
                entry, path = self.downloads.get_checkpoint(model)
            except (ValueError) as exc:
                raise RuntimeFailure(f'Download {model} in Settings before generating images.', 409) from exc
            if entry.revision != recipe.revision:
                raise RuntimeFailure('Image checkpoint revision does not match this runtime', 409)
            seeds = [((seed if seed is not None else secrets.randbits(53)) + index) % (2 ** 53) for index in range(count)]
            device = os.environ.get('KADAN_IMAGE_DEVICE', 'cuda:' + os.environ.get('KADAN_GPU', '0'))
            if device != 'cpu' and not (device.startswith('cuda:') and device[5:].isdecimal()):
                raise RuntimeFailure('KADAN_IMAGE_DEVICE must be cpu or cuda:<index>')
            options = dict(device=device, image=image)
            if self.backend is None:
                options['recipe'] = recipe
            pictures = (self.backend or native_image.generate)(path, self.runtime.ensure_resources(), prompt, aspect, seeds, cancel, **options)
            if len(pictures) != count:
                raise RuntimeError('Image provider returned an unexpected result count')
            identifier = str(uuid4())
            self.root.mkdir(parents=True, exist_ok=True)
            stage = self.root / ('.' + identifier)
            stage.mkdir()
            for index, picture in enumerate(pictures):
                picture.save(stage / f'{index}.png', format='PNG')
            result = ImageSet(id=identifier, mode='Edit' if source else 'Generate', prompt=prompt,
                aspect=ASPECTS[aspect], seeds=seeds, meta=f'{model} · {count} images',
                urls=[f'/v1/images/{identifier}/files/{index}' for index in range(count)])
            (stage / 'result.json').write_text(result.model_dump_json())
            if cancel.is_set():
                raise ResourceCancelled('Image generation cancelled')
            stage.rename(self.root / identifier)
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
                self._gate.release()

    async def run(self, request, body, source=None):
        """Forward disconnect cancellation and keep ownership until native cleanup finishes."""
        cancel = threading.Event()
        task = asyncio.create_task(asyncio.to_thread(self.generate, body.prompt, body.aspect,
            body.count, body.seed, cancel, source, getattr(body, 'model', None)))
        try:
            while not task.done():
                if await request.is_disconnected():
                    cancel.set()
                await asyncio.sleep(.1)
            return task.result()
        finally:
            cancel.set()
            await finish_cleanup(task)


image_manager = ImageManager()
