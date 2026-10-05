"""Offline worker; its filename must not shadow the installed wan package."""
import json
import sys

import imageio
import torch
import wan
from PIL import Image, ImageOps
from wan.configs import WAN_CONFIGS


@torch.inference_mode()
def main():
    """Use the checkpoint's native sampler/shift/guidance and encode real frames."""
    payload = json.load(sys.stdin)
    spec, task = payload['spec'], payload['task']
    config = WAN_CONFIGS[task]
    factory = {'ti2v-5B': wan.WanTI2V, 't2v-A14B': wan.WanT2V,
               'i2v-A14B': wan.WanI2V}[task]
    width, height = payload['width'], payload['height']
    alignment = 32 if task == 'ti2v-5B' else 16
    padded = (((width + alignment - 1) // alignment) * alignment,
              ((height + alignment - 1) // alignment) * alignment)
    frame_count = spec['duration'] * spec['fps']
    options = dict(size=padded, frame_num=((frame_count + 2) // 4) * 4 + 1,
        shift=config.sample_shift, sample_solver='unipc', sampling_steps=config.sample_steps,
        guide_scale=config.sample_guide_scale, n_prompt=spec['negative_prompt'],
        seed=spec['seed'] if spec['seed'] is not None else 42, offload_model=True)
    if task in ('ti2v-5B', 'i2v-A14B'):
        image = None
        if spec.get('image_path'):
            with Image.open(spec['image_path']) as source:
                image = ImageOps.fit(ImageOps.exif_transpose(source).convert('RGB'), padded)
        options.update(img=image, max_area=padded[0] * padded[1])
        if task == 'i2v-A14B':
            if image is None:
                raise ValueError('Image-to-video requires an image.')
            options.pop('size')
            if spec['resolution'] == '480p':
                options['shift'] = 3.0
    pipeline = factory(config=config, checkpoint_dir=payload['checkpoint'], device_id=0,
        rank=0, t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=True)
    video = pipeline.generate(spec['prompt'], **options)
    # Native tensors are C,F,H,W in [-1,1]. Crop alignment padding before encoding.
    top, left = (video.shape[2] - height) // 2, (video.shape[3] - width) // 2
    with imageio.get_writer(payload['output'], fps=spec['fps'], codec='libx264', macro_block_size=1) as writer:
        for index in range(frame_count):
            frame = video[:, index, top:top + height, left:left + width]
            pixels = ((frame.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
            writer.append_data(pixels)


if __name__ == '__main__':
    if '--check-imports' in sys.argv:
        print('Wan native imports OK:', wan.__file__)
    else:
        main()
