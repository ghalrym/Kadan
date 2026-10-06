"""Offline H3 generation inside the API, under Kadan resource ownership."""
from pathlib import Path
import shutil
import tempfile
import threading

from api.inference.resources import ResourceCancelled, ResourceExhausted
from api.services.model_downloads import model_manager
from api.services.model_catalog import H3_INT8_REVISION
from api.services.runtime import runtime_manager as runtime

H3_REVISION = H3_INT8_REVISION
SGLANG_REVISION = 'f048d5aa4bc1bcad7fa2c60d067590d83d6dbe4a'
GIB = 1024 ** 3


def check_media_tools():
    """Reject missing native media executables before reserving model memory."""
    missing = [name for name in ('ffmpeg', 'ffprobe') if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f'H3 requires {", ".join(missing)}; install FFmpeg before generation')


def sampling_arguments(spec, output: Path) -> dict:
    """Use Turbo with four denoiser forwards (five sigma points) and joint audio/video."""
    return dict(prompt=spec.prompt, task='t2va', conditions=[],
                target=dict(short_edge=int(spec.resolution.removesuffix('p')), aspect_ratio=spec.aspect,
                            duration_seconds=spec.duration),
                seed=spec.seed, num_inference_steps=5, flow_shift=12.0,
                audio_flow_shift=3.0,
                save_output=True, output_path=str(output.parent),
                output_file_name=output.name)


class H3Provider:
    def __init__(self, model_id='h3-fl2va-int8-turbo'):
        """Select a checkpoint without loading tensors or starting processes."""
        self.model_id = model_id

    def validate(self, spec):
        """Reject settings the native base model cannot honor before job admission."""
        if self.model_id != 'h3-fl2va-int8-turbo':
            raise ValueError('H3 Ref2VA requires reference inputs; select H3 FL2VA for text generation')
        if spec.fps != 24 or not 4 <= spec.duration <= 15:
            raise ValueError('H3 requires 24 fps and a duration between 4 and 15 seconds')
        if spec.resolution not in ('480p', '768p') or spec.aspect not in ('16:9', '9:16', '1:1'):
            raise ValueError('H3 requires 480p or 768p and a supported aspect ratio')
        if spec.negative_prompt.strip():
            raise ValueError('The CFG-distilled H3 checkpoint does not support negative prompts')

    def generate(self, spec, output_path: Path, cancellation: threading.Event):
        """Hold shared RAM/VRAM admission until synchronous pipeline cleanup completes."""
        self.validate(spec)
        check_media_tools()
        entry, checkpoint = model_manager.get_checkpoint(self.model_id)
        if entry.revision != H3_REVISION:
            raise ValueError('H3 checkpoint revision does not match the native provider')
        resources = runtime.ensure_resources()
        devices = resources.capacity.device_bytes
        if not devices:
            raise ResourceExhausted('H3 native inference requires a CUDA GPU')
        # Each CUDA device has an independent budget. INT8 weights live on the
        # host; layerwise streaming bounds residency independently of disk size.
        # A direct pipeline uses one GPU; no tensor-parallel worker processes.
        selected = sorted(devices, key=devices.get, reverse=True)[:1]
        if devices[selected[0]] < 12 * GIB:
            raise ResourceExhausted('H3 needs at least 12 GiB of available GPU budget')
        host_bytes = entry.estimated_bytes * 2 + 8 * GIB
        device_bytes = {device: devices[device] for device in selected}
        owner = 'video:h3'
        with resources.exclusive(owner, cancellation):
            reservation = resources.reserve(owner, 'video', host_bytes,
                                             device_bytes, cancel_event=cancellation)
            try:
                with reservation.lease(cancellation):
                    self._run(spec, checkpoint, output_path, cancellation, selected)
            finally:
                reservation.release()

    def _run(self, spec, checkpoint, output, cancellation, devices):
        """Render synchronously; keep the lease until in-process cleanup completes."""
        # Native imports must follow platform activation, and remain lazy for API startup.
        from api.inference.h3_pipeline import render, check_cancel

        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.kadan-h3-', dir=output.parent) as scratch:
            rendered = Path(scratch) / 'video.mp4'
            check_cancel(cancellation)
            render(checkpoint, sampling_arguments(spec, rendered), devices[0], cancellation)
            check_cancel(cancellation)
            if rendered.is_symlink() or not rendered.is_file() or not rendered.stat().st_size:
                raise RuntimeError('H3 produced no validated audiovisual output')
            rendered.replace(output)
