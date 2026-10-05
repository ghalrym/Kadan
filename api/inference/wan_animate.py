"""Wan Animate-14B admission, sharing Kadan's native worker lifecycle."""
from pathlib import Path
import tempfile

from api.inference.wan import WanProvider


class WanAnimateProvider(WanProvider):
    worker_script = 'wan_animate.py'

    def __init__(self, **kwargs):
        """Bind this checkpoint to its preprocessing and animation worker."""
        super().__init__('wan22-animate-14b', 'animate-14B', **kwargs)

    def validate(self, spec):
        """Require raw driving video and reference image before queue admission."""
        if spec.animation_mode not in ('animate', 'replace'):
            raise ValueError('Choose animation or replacement mode')
        if not spec.image_path or not spec.video_path:
            raise ValueError('Wan Animate requires an uploaded image and driving video')
        if not Path(spec.image_path).is_file() or not Path(spec.video_path).is_file():
            raise ValueError('A conditioning upload is unavailable')
        if spec.audio_path:
            raise ValueError('Wan Animate does not generate an audio track')
        if spec.fps != 30 or spec.resolution != '720p' or spec.aspect not in ('16:9', '9:16'):
            raise ValueError('Wan Animate requires 30 fps, 720p, landscape or portrait')
        if not self.python or not Path(self.python).is_file():
            raise RuntimeError('The Kadan Wan Animate worker environment is not configured')
        self._checkpoint()

    def generate(self, spec, output_path, cancellation):
        """Own preprocessing scratch outside the child so cancellation always removes it."""
        with tempfile.TemporaryDirectory(prefix='kadan-wan-animate-') as directory:
            self.scratch_directory = directory
            super().generate(spec, output_path, cancellation)
