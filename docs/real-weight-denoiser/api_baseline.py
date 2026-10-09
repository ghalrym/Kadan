"""Independent eager A/B oracle for MR125; no split adapter or trajectory snapshots.

One process executes exactly one frozen case. The outer reviewed launcher owns
thermal/physical guards, container limits, queue lease and API restoration.
"""
import argparse
from functools import wraps
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import time
from uuid import UUID

from trajectory_contracts import CRITERIA, REVISION, StepOrder

PROTOCOL = 'api-isolated-image-baseline-v1'
CASES = {
    'A': dict(prompt='A single red apple on a plain white table, soft natural daylight, realistic still-life photograph.', seed=42),
    'B': dict(prompt='A blue ceramic teapot beside a yellow lemon on a dark wooden table, soft window light, realistic still-life photograph.', seed=314159),
}
PNG_CAP = 24 * 1024**2
RUN_CAP = 64 * 1024**2


def canonical_device_uuid(value):
    return 'GPU-'+str(UUID(str(value).removeprefix('GPU-')))


def settings(case):
    return dict(CASES[case], checkpoint=REVISION, width=2048, height=2048, steps=40,
        dtype='bfloat16', true_cfg_scale=1.0, use_kv_cache=True,
        compilation=False, component_offload=True, vae_tiling=True)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def generate(pipeline, torch, case):
    """Independent eager pipeline with only order/finiteness guards, no split hooks."""
    order = StepOrder()
    pipeline.scheduler = type(pipeline.scheduler).from_config(pipeline.scheduler.config)
    scheduler_config = dict(pipeline.scheduler.config)
    original_forward = pipeline.transformer.forward
    original_step = pipeline.scheduler.step
    original_post = pipeline.image_processor.postprocess
    finite_checked = False
    timesteps = []

    @wraps(original_forward)
    def forward(*args, **kwargs):
        order.prediction(order.next, kwargs['kv_cache_mode'])
        return original_forward(*args, **kwargs)

    @wraps(original_step)
    def step(prediction, timestep, latent, *args, **kwargs):
        order.scheduler()
        timesteps.append(float(timestep))
        return original_step(prediction, timestep, latent, *args, **kwargs)

    @wraps(original_post)
    def postprocess(image, *args, **kwargs):
        nonlocal finite_checked
        order.complete()
        if not bool(torch.isfinite(image).all()):
            raise ValueError('Nonfinite raw VAE output before normalization/clipping')
        finite_checked = True
        return original_post(image, *args, **kwargs)

    pipeline.transformer.forward = forward
    pipeline.scheduler.step = step
    pipeline.image_processor.postprocess = postprocess
    started = time.monotonic()
    try:
        with torch.no_grad():
            result = pipeline(prompt=CASES[case]['prompt'], width=2048, height=2048,
                num_inference_steps=40, num_images_per_prompt=1,
                generator=torch.Generator(device='cpu').manual_seed(CASES[case]['seed']),
                true_cfg_scale=1.0, use_kv_cache=True, output_type='pil')
        order.complete()
        if not finite_checked or len(result.images) != 1:
            raise ValueError('Missing finite complete image output')
        image = result.images[0]
        if image.mode != 'RGBA' or image.size != (2048, 2048):
            raise ValueError('Expected one 2048-square RGBA image')
        return image, dict(request_seconds=time.monotonic()-started, completed_steps=order.next,
            raw_vae_finite=finite_checked, scheduler_config=scheduler_config, timesteps=timesteps)
    finally:
        pipeline.transformer.forward = original_forward
        pipeline.scheduler.step = original_step
        pipeline.image_processor.postprocess = original_post


def publish(image, output):
    target = output / 'output.png'
    temporary = output / 'output.partial.png'
    # Exclusive create plus hard-link publication prevents overwriting a baseline.
    try:
        with temporary.open('xb') as stream:
            image.save(stream, format='PNG')
            if stream.tell() > PNG_CAP:
                raise ValueError('PNG exceeds fixed artifact cap')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        return dict(file=target.name, bytes=target.stat().st_size, sha256=digest(target))
    finally:
        temporary.unlink(missing_ok=True)


def verify(root, case, commit):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if (manifest.get('protocol') != PROTOCOL or manifest.get('case') != case
            or manifest.get('source_commit') != commit or manifest.get('settings') != settings(case)
            or manifest.get('criteria') != CRITERIA or manifest.get('status') != 'passed'
            or manifest.get('raw_vae_finite') is not True or manifest.get('completed_steps') != 40):
        raise ValueError('Baseline identity or completion mismatch')
    artifact = manifest['artifact']
    path = root/'output.png'
    if (artifact['file'] != 'output.png' or path.is_symlink() or path.stat().st_size > PNG_CAP
            or path.stat().st_size != artifact['bytes'] or digest(path) != artifact['sha256']):
        raise ValueError('Baseline PNG changed or exceeds cap')
    return manifest


def compare(candidate, baseline, case, commit):
    verify(baseline, case, commit)
    # Optional pixel dependencies are used only by explicit offline comparison.
    np = importlib.import_module('numpy')
    image_module = importlib.import_module('PIL.Image')
    values = []
    for path in (Path(baseline)/'output.png', Path(candidate)):
        if path.is_symlink() or path.stat().st_size > PNG_CAP:
            raise ValueError('Unsafe or oversized comparison PNG')
        with image_module.open(path) as image:
            if image.size != (2048, 2048) or image.mode != 'RGBA':
                raise ValueError('Comparison requires 2048-square RGBA')
            values.append(np.asarray(image).astype(np.int16))
    delta = np.abs(values[0]-values[1])
    maximum = int(delta.max())
    rgb_mae = float(delta[..., :3].mean())
    alpha_mae = float(delta[..., 3].mean())
    return dict(case=case, candidate_sha256=digest(candidate), baseline_sha256=digest(Path(baseline)/'output.png'),
        pixel_max=maximum, rgb_mae=rgb_mae, alpha_mae=alpha_mae,
        bitwise_pixels=maximum==0, passed=maximum<=CRITERIA['pixel_max']
        and rgb_mae<=CRITERIA['pixel_mae'] and alpha_mae<=CRITERIA['pixel_mae'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--compare', type=Path, help='Offline comparison only; --output is the baseline directory')
    args = parser.parse_args()
    if not re.fullmatch('[a-f0-9]{40}', args.commit):
        raise ValueError('Exact source SHA required')
    if args.compare is not None:
        result = compare(args.compare,args.output,args.case,args.commit)
        print(json.dumps(result))
        if not result['passed']: raise SystemExit(1)
        return
    if not args.output.is_dir() or any(args.output.iterdir()):
        raise ValueError('Baseline output directory must exist and be empty')
    # Optional GPU imports only after CLI admission. CPU unit tests import this
    # module without initializing CUDA or loading optional model dependencies.
    torch = importlib.import_module('torch')
    diffusers = importlib.import_module('diffusers')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    device = torch.cuda.get_device_properties(0)
    device_uuid = canonical_device_uuid(device.uuid)
    if device_uuid != os.environ['KADAN_EXPECTED_GPU_UUID']:
        raise ValueError(f'Baseline physical GPU mapping differs: {device_uuid}')
    torch.cuda.set_per_process_memory_fraction(22*1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    pipeline = None
    report = dict(protocol=PROTOCOL, case=args.case, source_commit=args.commit,
        settings=settings(args.case), criteria=CRITERIA, status='incomplete', production_activation=False)
    try:
        started = time.monotonic()
        pipeline = diffusers.QwenImage21Pipeline.from_pretrained('/models/qwen-image-2.1-'+REVISION,
            local_files_only=True, torch_dtype=torch.bfloat16, use_safetensors=True)
        if not pipeline.transformer.config.causal_condition or len(pipeline.transformer.transformer_blocks) != 32:
            raise ValueError('Unsupported transformer configuration')
        pipeline.vae.enable_tiling()
        pipeline.enable_model_cpu_offload(gpu_id=0)
        report['load_seconds'] = time.monotonic()-started
        report['sources'] = {name:digest(inspect.getfile(type(component))) for name,component in
            [('pipeline',pipeline),('transformer',pipeline.transformer),('scheduler',pipeline.scheduler),('vae',pipeline.vae)]}
        report['device_uuid'] = device_uuid
        report['runtime'] = dict(torch=torch.__version__, diffusers=diffusers.__version__, cuda=torch.version.cuda)
        report['precision'] = dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            bf16_reduced_precision=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            cudnn_tf32=torch.backends.cudnn.allow_tf32)
        torch.cuda.synchronize()
        image, details = generate(pipeline, torch, args.case)
        torch.cuda.synchronize()
        report.update(details, artifact=publish(image,args.output), status='passed')
    except BaseException as exc:
        report.update(status='failed', error=dict(type=type(exc).__name__, message=str(exc)[:2000]))
        raise
    finally:
        (args.output/'manifest.json').write_text(json.dumps(report,indent=2))
        if pipeline is not None:
            pipeline.remove_all_hooks()


if __name__ == '__main__':
    main()
