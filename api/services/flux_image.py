"""FLUX 3 Image hosted inference; no local generative weights are claimed."""
import base64
from io import BytesIO
from pathlib import Path
import tempfile

from PIL import Image

from api.inference.resources import ResourceCancelled
from api.services.bfl import BflClient

ASPECTS = frozenset(('21:9', '2:1', '16:9', '3:2', '7:5', '4:3', '5:4', '1:1',
                     '4:5', '3:4', '5:7', '2:3', '9:16', '1:2', '9:21'))
RESOLUTIONS = frozenset(('768sq', '1k', '1.5k', '2k', '4k'))


class FluxImageProvider:
    def __init__(self, client=None):
        self.client = client or BflClient()

    def validate(self, prompt, aspect, count, seed=None, resolution='1k'):
        """Reject unsupported controls before any paid submission."""
        if not prompt.strip() or aspect not in ASPECTS or type(count) is not int or not 1 <= count <= 4:
            raise ValueError('Unsupported FLUX 3 Image request')
        if seed is not None:
            raise ValueError('FLUX 3 Image does not expose a seed parameter.')
        if resolution not in RESOLUTIONS:
            raise ValueError('Unsupported FLUX 3 Image resolution')
        if not self.client.configured():
            raise RuntimeError('BFL_API_KEY is not configured.')

    def generate(self, prompt, aspect, count, cancel, source=None, seed=None, resolution='1k'):
        """Return decoded images for Kadan's existing atomic media publisher."""
        self.validate(prompt, aspect, count, seed, resolution)
        payload = dict(prompt=prompt, aspect_ratio=aspect, resolution=resolution)
        if source is not None:
            buffer = BytesIO()
            source.save(buffer, format='PNG')
            payload['images'] = ['data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')]
        images = []
        with tempfile.TemporaryDirectory(prefix='kadan-flux-image-') as directory:
            for index in range(count):
                if cancel.is_set():
                    raise ResourceCancelled('Image generation cancelled')
                url = self.client.generate('flux-3-image', payload, cancel)
                path = Path(directory) / f'{index}.image'
                self.client.download(url, path, cancel, 100 * 1024 ** 2)
                with Image.open(path) as image:
                    images.append(image.convert('RGB'))
        return images
