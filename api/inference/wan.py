"""Native Wan2.2 inference in a Kadan-owned, dependency-isolated process."""
import json
import os
from pathlib import Path
import subprocess
import signal
import threading
import uuid

from api.inference.resources import ResourceCancelled
from api.services.model_downloads import model_manager
from api.services.runtime import runtime_manager

SOURCE_REVISION = '1ea34ff48f87168174e12956e200b1d908b1c5ff'


class WanProvider:
    def __init__(self, model_id, task, manager=None, resources=None, python=None):
        self.model_id = model_id
        self.task = task
        self.manager = manager or model_manager
        self.resources = resources
        self.python = python or os.getenv('KADAN_WAN_PYTHON')

    def validate(self, spec):
        """Reject unsupported controls before creating a job or allocating memory."""
        if self.task not in ('ti2v-5B', 't2v-A14B'):
            raise ValueError('Unsupported Wan generation task.')
        required_fps = 24 if self.task == 'ti2v-5B' else 16
        if spec.fps != required_fps:
            raise ValueError(f'This Wan2.2 checkpoint generates at {required_fps} fps.')
        if spec.audio_path or spec.video_path:
            raise ValueError('This Wan checkpoint does not accept audio or video conditioning.')
        if self.task == 't2v-A14B' and spec.image_path:
            raise ValueError('The text-to-video checkpoint does not accept an image.')
        spec.dimensions
        if not self.python or not Path(self.python).is_file():
            raise RuntimeError('The Kadan Wan worker environment is not configured.')
        self._checkpoint()

    def _checkpoint(self):
        """Use only a complete Kadan-managed checkpoint; never download during inference."""
        entry, path = self.manager.get_checkpoint(self.model_id)
        if entry.kind != 'video':
            raise ValueError('The selected checkpoint is not a video model.')
        return path

    def generate(self, spec, output_path: Path, cancellation: threading.Event):
        """Hold shared RAM/VRAM ownership until the native process exits, even on cancellation."""
        self.validate(spec)
        checkpoint = self._checkpoint()
        resources = self.resources or runtime_manager.ensure_resources()
        if not resources.capacity.device_bytes:
            raise RuntimeError('Wan2.2 requires a CUDA device.')
        device = max(resources.capacity.device_bytes, key=resources.capacity.device_bytes.get)
        owner = f'video-wan-{uuid.uuid4().hex}'
        # Conservative CPU-offload admission, not a measured peak-memory claim.
        host_bytes = sum(file.stat().st_size for file in checkpoint.rglob('*') if file.is_file()) * 2
        worker = Path(__file__).parent / 'workers' / 'wan.py'
        payload = dict(spec=vars(spec), checkpoint=str(checkpoint), task=self.task,
                       output=str(output_path), width=spec.dimensions[0], height=spec.dimensions[1])
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(device), 'HF_HUB_OFFLINE': '1',
               'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
        with resources.exclusive(owner, cancellation):
            reservation = resources.reserve(owner, 'video', host_bytes,
                {device: resources.capacity.device_bytes[device]}, cancel_event=cancellation)
            try:
                with reservation.lease(cancellation):
                    process = subprocess.Popen([self.python, str(worker)], stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, env=env, start_new_session=True)
                    try:
                        process.stdin.write(json.dumps(payload))
                        process.stdin.close()
                        while process.poll() is None:
                            if cancellation.wait(.1):
                                raise ResourceCancelled('Video generation cancelled')
                        if process.returncode != 0:
                            raise RuntimeError('Wan2.2 inference failed; verify worker dependencies and memory.')
                        if cancellation.is_set():
                            raise ResourceCancelled('Video generation cancelled')
                        if not output_path.is_file() or output_path.stat().st_size == 0:
                            raise RuntimeError('Wan2.2 did not produce a video.')
                    finally:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
            finally:
                reservation.release()
