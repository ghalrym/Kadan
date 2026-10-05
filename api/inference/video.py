"""Provider-neutral video generation settings."""
from dataclasses import dataclass


@dataclass(frozen=True)
class VideoSpec:
    prompt: str
    negative_prompt: str = ''
    duration: int = 8
    fps: int = 24
    resolution: str = '720p'
    aspect: str = '16:9'
    seed: int = 42
    animation_mode: str = 'animate'
    image_path: str | None = None
    audio_path: str | None = None
    video_path: str | None = None

    @property
    def dimensions(self):
        """Return exact requested output dimensions, before provider padding."""
        height = int(self.resolution.removesuffix('p'))
        width = {480: 854, 720: 1280, 768: 1366, 1080: 1920}[height]
        if self.aspect == '1:1':
            return height, height
        return (width, height) if self.aspect == '16:9' else (height, width)
