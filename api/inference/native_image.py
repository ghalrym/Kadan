"""Shared native Diffusers image lifecycle; each checkpoint supplies an explicit recipe."""
from dataclasses import dataclass
import gc
import importlib
from pathlib import Path

from api.inference.resources import ResourceCancelled, ResourceExhausted
from api.services.decisions import clear_failure_frames

@dataclass(frozen=True)
class ImageRecipe:
    model_id: str
    revision: str
    pipeline: str
    sizes: dict[str, tuple[int, int]]
    steps: int
    guidance_scale: float | None = None
    quantization: str | None = None
    weight_filename: str | None = None


GIB = 1024 ** 3


def native_modules():
    """Import optional native dependencies only when allocating an image workload."""
    return importlib.import_module('torch'), importlib.import_module('diffusers')


def generate(path: Path, resources, prompt, aspect, seeds, cancel, device='cuda:0', image=None, modules=native_modules, *, recipe: ImageRecipe):
    """Generate serially under one exclusive lease; release actual allocations before accounting."""
    if cancel.is_set():
        raise ResourceCancelled('Image generation cancelled')
    torch, diffusers = modules()
    pipeline_type = getattr(diffusers, recipe.pipeline, None)
    if pipeline_type is None:
        raise RuntimeError(f'Install the pinned image runtime requirements for {recipe.model_id} before generating images.')
    weights = sum(item.stat().st_size for item in path.rglob('*.safetensors'))
    if not weights:
        raise ValueError('The image checkpoint has no weights')
    cpu = device == 'cpu'
    gpu = None if cpu else int(device.removeprefix('cuda:'))
    gpu_budget = 0 if cpu else resources.capacity.device_bytes.get(gpu, 0)
    if not cpu and not gpu_budget:
        raise ResourceExhausted('The selected CUDA device is unavailable')
    # Reserve all of the selected GPU budget: components/offload hooks share this
    # lease, and another workload cannot allocate alongside the image pipeline.
    host_budget = weights * (2 if cpu else 1) + 8 * GIB
    resident_weights = weights
    quant_loader = None
    if recipe.quantization is not None or recipe.weight_filename is not None:
        if recipe.quantization not in ('fp8', 'nvfp4') or not recipe.weight_filename:
            raise ValueError('Quantized recipes require an explicit format and local weight filename')
        quant_loader = importlib.import_module('api.inference.klein_quant')
        quant_path = quant_loader.transformer_path(path, recipe.weight_filename)
        dense_bytes = quant_loader.dense_size(quant_path, 4 if cpu else 2)
        # The input mapping, decoded state and Diffusers construction can coexist.
        # GPU placement decisions must use dense residency, not the download size.
        host_budget += dense_bytes * 2 + quant_loader.SCRATCH_BYTES
        resident_weights = weights - quant_path.stat().st_size + dense_bytes
    owner = recipe.model_id
    pipeline = None
    transformer = None
    overrides = None
    reservation = None
    results = []
    with resources.exclusive(owner, cancel):
        try:
            reservation = resources.reserve(owner, 'image', host_bytes=host_budget,
                device_bytes={} if cpu else {gpu: gpu_budget}, cancel_event=cancel)
            with reservation.lease(cancel):
                dtype = torch.float32 if cpu else torch.bfloat16
                overrides = {}
                if quant_loader is not None:
                    transformer = quant_loader.load_transformer(path, recipe.weight_filename,
                        recipe.quantization, dtype, diffusers, cancel)
                    overrides['transformer'] = transformer
                pipeline = pipeline_type.from_pretrained(str(path), local_files_only=True,
                    torch_dtype=dtype, use_safetensors=True, **overrides)
                if cancel.is_set():
                    raise ResourceCancelled('Image generation cancelled')
                if cpu or gpu_budget >= resident_weights + 8 * GIB:
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
                width, height = recipe.sizes[aspect]
                for seed in seeds:
                    if cancel.is_set():
                        raise ResourceCancelled('Image generation cancelled')
                    options = dict(prompt=prompt, width=width, height=height,
                        num_inference_steps=recipe.steps, num_images_per_prompt=1,
                        generator=torch.Generator(device='cpu').manual_seed(seed),
                        callback_on_step_end=checkpoint)
                    if recipe.guidance_scale is not None:
                        options['guidance_scale'] = recipe.guidance_scale
                    if image is not None:
                        options['image'] = image
                    result = pipeline(**options)
                    if len(result.images) != 1:
                        raise RuntimeError('The image pipeline returned an unexpected image count')
                    results.append(result.images[0])
                if cancel.is_set():
                    raise ResourceCancelled('Image generation cancelled')
                return results
        except BaseException as exc:
            clear_failure_frames(exc)
            raise
        finally:
            # Dropping the pipeline also drops Accelerate hooks and component
            # tensors. Never release a reservation while these remain reachable.
            pipeline = transformer = overrides = None
            gc.collect()
            if not cpu:
                with torch.cuda.device(gpu):
                    torch.cuda.empty_cache()
            if reservation is not None:
                reservation.release()
