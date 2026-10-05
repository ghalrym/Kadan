"""Hosted FLUX 3 text-to-video with synchronized audio."""
from api.inference.resources import ResourceCancelled
from api.services.bfl import BflClient

RESOLUTIONS = {'720p': 'hd', '1080p': 'fhd', '1440p': 'qhd', '2160p': 'uhd'}
ASPECTS = frozenset(('21:9', '2:1', '16:9', '4:3', '1:1', '3:4', '9:16', '9:21'))


class FluxVideoProvider:
    def __init__(self, client=None):
        self.client = client or BflClient()

    def validate(self, spec):
        """Reject unsupported controls before submitting a billable request."""
        if not spec.prompt.strip() or spec.aspect not in ASPECTS or spec.resolution not in RESOLUTIONS:
            raise ValueError('Unsupported FLUX 3 Video request')
        if type(spec.duration) is not int or not 5 <= spec.duration <= 20 or spec.fps != 24:
            raise ValueError('FLUX 3 Video requires 5–20 whole seconds at 24 fps.')
        if spec.negative_prompt or spec.seed is not None:
            raise ValueError('FLUX 3 Video does not expose negative prompt or seed parameters.')
        if not self.client.configured():
            raise RuntimeError('BFL_API_KEY is not configured.')

    def generate(self, spec, output_path, cancellation):
        """Write a completed MP4 for Kadan's atomic publisher; allocate no GPU model."""
        self.validate(spec)
        payload = dict(prompt=spec.prompt, mode='t2v', duration=spec.duration,
                       resolution=RESOLUTIONS[spec.resolution], aspect_ratio=spec.aspect,
                       generate_audio=True)
        url = self.client.generate('flux-3-video', payload, cancellation)
        self.client.download(url, output_path, cancellation, 2 * 1024 ** 3)
        if cancellation.is_set():
            output_path.unlink(missing_ok=True)
            raise ResourceCancelled('Video download cancelled')
        with output_path.open('rb') as output:
            header = output.read(12)
        if len(header) < 12 or header[4:8] != b'ftyp':
            output_path.unlink(missing_ok=True)
            raise RuntimeError('BFL returned an invalid MP4 file.')
