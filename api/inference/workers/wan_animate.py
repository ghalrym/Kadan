"""Real pose/face/mask preprocessing followed by native Wan Animate inference."""
import json
from pathlib import Path
import subprocess
import sys

from decord import VideoReader
import imageio
import imageio_ffmpeg
from PIL import Image, ImageOps
import torch
import wan
from wan.configs import WAN_CONFIGS


def prepare_inputs(payload, scratch):
    """Normalize raw inputs, then run official preprocessing with local auxiliary weights."""
    spec = payload['spec']
    width, height = payload['width'], payload['height']
    frames = spec['duration'] * spec['fps']
    video = scratch / 'driving.mp4'
    reference = scratch / 'reference.png'
    processed = scratch / 'processed'
    with Image.open(spec['image_path']) as source:
        ImageOps.fit(ImageOps.exif_transpose(source).convert('RGB'), (width, height)).save(reference)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-y', '-i', spec['video_path'],
        '-an', '-vf', f'scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},fps=30',
        '-frames:v', str(frames), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(video)], check=True)
    if len(VideoReader(str(video))) < frames:
        raise ValueError('The driving video is shorter than the requested duration')
    script = Path(wan.__file__).parent / 'modules/animate/preprocess/preprocess_data.py'
    if not script.is_file():
        raise RuntimeError('The pinned Wan package is missing its preprocessing scripts')
    command = [sys.executable, str(script), '--ckpt_path',
        str(Path(payload['checkpoint']) / 'process_checkpoint'),
        '--video_path', str(video), '--refer_path', str(reference),
        '--save_path', str(processed), '--resolution_area', str(width), str(height), '--fps', '30']
    if spec['animation_mode'] == 'replace':
        command.append('--replace_flag')
    else:
        command.append('--retarget_flag')
    # The upstream script owns pose/face extraction and SAM2 masks. FLUX is not enabled.
    subprocess.run(command, check=True)
    required = ['src_pose.mp4', 'src_face.mp4', 'src_ref.png']
    if spec['animation_mode'] == 'replace':
        required += ['src_bg.mp4', 'src_mask.mp4']
    if not all((processed / name).is_file() for name in required):
        raise RuntimeError('Animate preprocessing did not produce the required conditioning assets')
    return processed


@torch.inference_mode()
def run(payload):
    """Keep preprocessing and generation in one leased worker process group."""
    spec = payload['spec']
    config = WAN_CONFIGS['animate-14B']
    replacement = spec['animation_mode'] == 'replace'
    processed = prepare_inputs(payload, Path(payload['scratch_directory']))
    pipeline = wan.WanAnimate(config=config, checkpoint_dir=payload['checkpoint'],
        device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False, use_sp=False,
        t5_cpu=True, convert_model_dtype=True, use_relighting_lora=replacement)
    video = pipeline.generate(src_root_path=str(processed), replace_flag=replacement,
        refert_num=1, clip_len=config.frame_num, shift=config.sample_shift,
        sample_solver='unipc', sampling_steps=config.sample_steps,
        guide_scale=config.sample_guide_scale, input_prompt=spec['prompt'],
        n_prompt=spec['negative_prompt'], seed=spec['seed'], offload_model=True)
    frame_count = spec['duration'] * spec['fps']
    if tuple(video.shape[2:]) != (payload['height'], payload['width']):
        raise RuntimeError('Animate returned a different canvas than requested')
    if video.shape[1] < frame_count:
        raise RuntimeError('Animate returned fewer frames than requested')
    with imageio.get_writer(payload['output'], fps=30, codec='libx264', macro_block_size=1) as writer:
        for index in range(frame_count):
            pixels = ((video[:, index].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
            writer.append_data(pixels)


if __name__ == '__main__':
    run(json.load(sys.stdin))
