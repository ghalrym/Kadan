"""Offline H3 generation in a Kadan-owned, isolated native worker."""
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import threading

from api.inference.resources import ResourceCancelled, ResourceExhausted
from api.services.model_downloads import model_manager
from api.services.runtime import runtime_manager as runtime

H3_REVISION = '42ed227ee7df40d41602854ae760620d6eb651fe'
SGLANG_REVISION = 'f048d5aa4bc1bcad7fa2c60d067590d83d6dbe4a'
GIB = 1024 ** 3


def check_media_tools():
    """Reject missing native media executables before reserving model memory."""
    missing = [name for name in ('ffmpeg', 'ffprobe') if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f'H3 requires {", ".join(missing)}; install FFmpeg before generation')


def sampling_arguments(spec, output: Path) -> dict:
    """Preserve the official CFG-distilled base schedule and audiovisual output."""
    return dict(prompt=spec.prompt, task='t2va', conditions=[],
                target=dict(short_edge=768, aspect_ratio=spec.aspect,
                            duration_seconds=spec.duration),
                seed=spec.seed, num_inference_steps=50, flow_shift=12.0,
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
    def __init__(self, model_id='h3-fl2va'):
        """Select a checkpoint without loading tensors or starting processes."""
        self.model_id = model_id

    def validate(self, spec):
        """Reject settings the native base model cannot honor before job admission."""
        if self.model_id != 'h3-fl2va':
            raise ValueError('H3 Ref2VA requires reference inputs; select H3 FL2VA for text generation')
        if spec.fps != 24 or not 4 <= spec.duration <= 15:
            raise ValueError('H3 requires 24 fps and a duration between 4 and 15 seconds')
        if spec.resolution != '768p' or spec.aspect not in ('16:9', '9:16', '1:1'):
            raise ValueError('H3 requires 768p and a supported aspect ratio')
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
        device = max(devices, key=devices.get)
        owner = 'video:h3'
        # Conservative host budget includes staging of the full unpruned checkpoint.
        host_bytes = int(os.getenv('KADAN_H3_RAM_BYTES', str(entry.estimated_bytes * 2 + 16 * GIB)))
        device_bytes = int(os.getenv('KADAN_H3_VRAM_BYTES', str(entry.estimated_bytes + 16 * GIB)))
        if host_bytes <= 0 or device_bytes <= 0:
            raise ValueError('H3 RAM/VRAM admission budgets must be positive byte counts')
        with resources.exclusive(owner, cancellation):
            reservation = resources.reserve(owner, 'video', host_bytes,
                                             {device: device_bytes}, cancel_event=cancellation)
            try:
                with reservation.lease(cancellation):
                    self._run(spec, checkpoint, output_path, cancellation, device)
            finally:
                reservation.release()

    def _run(self, spec, checkpoint, output, cancellation, device):
        """Pass local paths only; cancellation terminates the worker's entire process group."""
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   HF_DATASETS_OFFLINE='1')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        env['CUDA_VISIBLE_DEVICES'] = visible[device] if visible != [''] else str(device)
        python = os.environ.get('KADAN_H3_PYTHON', sys.executable)
        # Keep the renderer's sanitizer-safe filename separate from the queue's
        # hidden staging name. The same filesystem permits atomic publication.
        with tempfile.TemporaryDirectory(prefix='.kadan-h3-', dir=output.parent) as scratch:
            rendered = Path(scratch) / 'video.mp4'
            payload = dict(checkpoint=str(checkpoint.resolve()),
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
                        raise RuntimeError('H3 native worker failed; verify the pinned optional runtime and GPU capacity')
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
