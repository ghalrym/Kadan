"""Native Qwen Image pipeline with Kadan-owned admission and cleanup."""
import gc
import importlib
from pathlib import Path

from api.inference.resources import ResourceCancelled, ResourceExhausted

MODEL_ID = 'qwen-image-2.1'
REVISION = 'd26bb61231c349cf6b7896fa83353113880e1ba3'
SIZES = {'1:1': (2048, 2048), '4:3': (2400, 1792), '3:4': (1792, 2400), '16:9': (2752, 1536)}
GIB = 1024 ** 3


def native_modules():
    """Import optional native dependencies only when allocating an image workload."""
    return importlib.import_module('torch'), importlib.import_module('diffusers')


def generate(path: Path, resources, prompt, aspect, seeds, cancel, device='cuda:0', image=None, modules=native_modules):
    """Generate serially under one exclusive lease; release actual allocations before accounting."""
    if cancel.is_set():
        raise ResourceCancelled('Image generation cancelled')
    torch, diffusers = modules()
    pipeline_type = getattr(diffusers, 'QwenImage21Pipeline', None)
    if pipeline_type is None:
        raise RuntimeError('Install the pinned Qwen Image runtime requirements before generating images.')
    weights = sum(item.stat().st_size for item in path.rglob('*.safetensors'))
    if not weights:
        raise ValueError('The Qwen Image checkpoint has no weights')
    cpu = device == 'cpu'
    gpu = None if cpu else int(device.removeprefix('cuda:'))
    gpu_budget = 0 if cpu else resources.capacity.device_bytes.get(gpu, 0)
    if not cpu and not gpu_budget:
        raise ResourceExhausted('The selected CUDA device is unavailable')
    # Reserve all of the selected GPU budget: components/offload hooks share this
    # lease, and another workload cannot allocate alongside the image pipeline.
    host_budget = weights * (2 if cpu else 1) + 8 * GIB
    owner = 'qwen-image-2.1'
    pipeline = None
    reservation = None
    results = []
    with resources.exclusive(owner, cancel):
        try:
            reservation = resources.reserve(owner, 'image', host_bytes=host_budget,
                device_bytes={} if cpu else {gpu: gpu_budget}, cancel_event=cancel)
            with reservation.lease(cancel):
                pipeline = pipeline_type.from_pretrained(str(path), local_files_only=True,
                    torch_dtype=torch.float32 if cpu else torch.bfloat16, use_safetensors=True)
                if cancel.is_set():
                    raise ResourceCancelled('Image generation cancelled')
                if cpu or gpu_budget >= weights + 8 * GIB:
                    pipeline.to(device)
                else:
                    # Sequential offload supports devices that cannot hold a complete
                    # encoder/transformer; Kadan still owns their RAM and VRAM lease.
                    pipeline.enable_sequential_cpu_offload(gpu_id=gpu)
                pipeline.vae.enable_tiling()
                def checkpoint(_pipeline, _step, _timestep, values):
                    if cancel.is_set():
                        raise ResourceCancelled('Image generation cancelled')
                    return values
                width, height = SIZES[aspect]
                for seed in seeds:
                    if cancel.is_set():
                        raise ResourceCancelled('Image generation cancelled')
                    options = dict(prompt=prompt, width=width, height=height,
                        num_inference_steps=40, num_images_per_prompt=1,
                        generator=torch.Generator(device='cpu').manual_seed(seed),
                        callback_on_step_end=checkpoint)
                    if image is not None:
                        options['image'] = image
                    result = pipeline(**options)
                    if len(result.images) != 1:
                        raise RuntimeError('Qwen Image returned an unexpected image count')
                    results.append(result.images[0])
                if cancel.is_set():
                    raise ResourceCancelled('Image generation cancelled')
                return results
        finally:
            # Dropping the pipeline also drops Accelerate hooks and component
            # tensors. Never release a reservation while these remain reachable.
            pipeline = None
            gc.collect()
            if not cpu:
                with torch.cuda.device(gpu):
                    torch.cuda.empty_cache()
            if reservation is not None:
                reservation.release()
