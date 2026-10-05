"""Run only inside the isolated, pinned LTX environment; inputs are local paths."""
import json
import sys

import torch
from ltx_core.types import Audio
from ltx_core.model.video_vae import get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import OffloadMode


def crop_frames(video, width, height, count):
    """Crop model padding and the temporal alignment frame without buffering the clip."""
    chunks = (video,) if isinstance(video, torch.Tensor) else video
    remaining = count
    for chunk in chunks:
        if remaining <= 0:
            break
        top = (chunk.shape[1] - height) // 2
        left = (chunk.shape[2] - width) // 2
        selected = chunk[:remaining, top:top + height, left:left + width, :]
        remaining -= selected.shape[0]
        yield selected


@torch.inference_mode()
def main():
    """Generate synchronized audio/video with upstream distilled sampling defaults."""
    payload = json.load(sys.stdin)
    spec, assets = payload['spec'], payload['assets']
    upsampler = assets.pop('spatial_upsampler_path')
    pipeline = DistilledPipeline(model_paths=ModelPaths.from_split(**assets),
        spatial_upsampler_path=upsampler, loras=[], device=torch.device('cuda:0'),
        offload_mode=OffloadMode.CPU)
    frames = spec['duration'] * spec['fps']
    width, height = payload['width'], payload['height']
    result = pipeline(prompt=spec['prompt'], seed=spec['seed'], height=((height + 63) // 64) * 64,
        width=((width + 63) // 64) * 64, num_frames=((frames + 6) // 8) * 8 + 1,
        frame_rate=spec['fps'], images=[], enhance_prompt=False)
    audio = result.audio
    if audio is not None:
        audio = Audio(waveform=audio.waveform[..., :spec['duration'] * audio.sampling_rate],
                      sampling_rate=audio.sampling_rate)
    encode_video(video=crop_frames(result.video, width, height, frames), fps=spec['fps'], audio=audio,
        output_path=payload['output'], video_chunks_number=get_video_chunks_number(result.num_frames, result.tiling_config))


if __name__ == '__main__':
    main()
