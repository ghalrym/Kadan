"""Offline H3 generation in a Kadan-owned, isolated native worker."""
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
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


def stop_process_group(process):
    """Reap the worker and stop its native scheduler children before releasing memory."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # The main worker may exit before its scheduler children; kill the entire group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


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
        """Hold exclusive RAM/VRAM admission until the isolated generation process exits."""
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
        # TP=2 is valid for both H3 and Qwen3VL ConvRot group dimensions.
        selected = sorted(devices, key=devices.get, reverse=True)[:2]
        if any(devices[device] < 12 * GIB for device in selected):
            selected = selected[:1]
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
        """Pass local paths only; cancellation terminates the worker's entire process group."""
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   HF_DATASETS_OFFLINE='1')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        env['CUDA_VISIBLE_DEVICES'] = ','.join(visible[device] if visible != [''] else str(device)
                                              for device in devices)
        python = os.environ.get('KADAN_H3_PYTHON', '/opt/kadan-h3/bin/python')
        if not Path(python).is_file():
            raise RuntimeError('The installed H3 runtime is missing; rebuild the standard Kadan API image')
        # Keep the renderer's sanitizer-safe filename separate from the queue's
        # hidden staging name. The same filesystem permits atomic publication.
        with tempfile.TemporaryDirectory(prefix='.kadan-h3-', dir=output.parent) as scratch:
            rendered = Path(scratch) / 'video.mp4'
            payload = dict(checkpoint=str(checkpoint.resolve()), num_gpus=len(devices),
                           sampling=sampling_arguments(spec, rendered))
            request = Path(scratch) / 'request.json'
            request.write_text(json.dumps(payload))
            with (Path(scratch) / 'worker.log').open('w+b') as log:
                if cancellation.is_set():
                    raise ResourceCancelled('H3 generation cancelled')
                process = subprocess.Popen([python, '-m', 'api.inference.h3_worker', str(request)],
                                           cwd=Path(__file__).resolve().parents[2], env=env,
                                           stdout=log, stderr=log, start_new_session=True)
                try:
                    while process.poll() is None:
                        if cancellation.wait(.1):
                            raise ResourceCancelled('H3 generation cancelled')
                    if cancellation.is_set():
                        raise ResourceCancelled('H3 generation cancelled')
                    if process.returncode:
                        log.seek(0)
                        diagnostic = output.with_suffix('.worker.log')
                        diagnostic.write_bytes(log.read())
                        raise RuntimeError(f'H3 native worker failed (exit {process.returncode}); '
                                           f'diagnostics: {diagnostic.name}')
                    if not rendered.is_file() or rendered.stat().st_size == 0:
                        raise RuntimeError('H3 worker produced no audiovisual output')
                except BaseException:
                    output.unlink(missing_ok=True)
                    raise
                finally:
                    stop_process_group(process)
                if cancellation.is_set():
                    raise ResourceCancelled('H3 generation cancelled')
                rendered.replace(output)
