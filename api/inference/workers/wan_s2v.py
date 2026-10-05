"""Native image-and-audio conditioned Wan S2V, without optional speech synthesis."""
from pathlib import Path
import subprocess
import tempfile
import wave

import imageio
import imageio_ffmpeg
from PIL import Image, ImageOps
import torch
import torch.nn.functional as functional
import wan
from wan.configs import WAN_CONFIGS


def generate_frames(payload, image_path, audio_path):
    """Use the official S2V API and its checkpoint-specific sampler settings."""
    spec = payload['spec']
    config = WAN_CONFIGS['s2v-14B']
    pipeline = wan.WanS2V(config=config, checkpoint_dir=payload['checkpoint'], device_id=0,
        rank=0, t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=True)
    return pipeline.generate(input_prompt=spec['prompt'], ref_image_path=str(image_path),
        audio_path=str(audio_path), enable_tts=False, tts_prompt_audio=None,
        tts_prompt_text=None, tts_text=None, num_repeat=1, pose_video=None,
        max_area=payload['width'] * payload['height'], infer_frames=spec['duration'] * 16,
        shift=config.sample_shift, sample_solver='unipc', sampling_steps=config.sample_steps,
        guide_scale=config.sample_guide_scale, n_prompt=spec['negative_prompt'],
        seed=spec['seed'], offload_model=True, init_first_frame=False)


@torch.inference_mode()
def render(payload):
    """Condition on uploaded image/audio, encode exact dimensions and require successful mux."""
    spec = payload['spec']
    width, height = payload['width'], payload['height']
    frames = spec['duration'] * 16
    with tempfile.TemporaryDirectory(prefix='kadan-wan-s2v-') as temporary:
        root = Path(temporary)
        reference, audio_path, silent = root / 'reference.png', root / 'audio.wav', root / 'silent.mp4'
        with Image.open(spec['image_path']) as source:
            ImageOps.fit(source.convert('RGB'), (width, height)).save(reference)
        with wave.open(spec['audio_path'], 'rb') as source:
            if source.getnframes() < source.getframerate() * spec['duration']:
                raise ValueError('Reference audio is shorter than the requested video.')
            with wave.open(str(audio_path), 'wb') as target:
                target.setparams(source.getparams())
                target.writeframes(source.readframes(source.getframerate() * spec['duration']))
        video = generate_frames(payload, reference, audio_path)
        if video.shape[1] < frames:
            raise RuntimeError('Wan S2V returned fewer frames than requested.')
        # S2V chooses an internal aligned area; preserve the requested display dimensions.
        with imageio.get_writer(str(silent), fps=16, codec='libx264', macro_block_size=1) as writer:
            for index in range(frames):
                frame = functional.interpolate(video[:, index].unsqueeze(0).float(),
                    size=(height, width), mode='bilinear', align_corners=False)[0]
                pixels = ((frame.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
                writer.append_data(pixels)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-y', '-i', str(silent),
            '-i', str(audio_path), '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy',
            '-c:a', 'aac', '-t', str(spec['duration']), '-shortest', payload['output']],
            check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
