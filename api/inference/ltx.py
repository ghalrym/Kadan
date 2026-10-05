"""Native LTX-2.5 inference in a Kadan-owned, dependency-isolated process."""
import json
import os
from pathlib import Path
import subprocess
import threading
import uuid

from api.inference.resources import ResourceCancelled
from api.inference.processes import visible_cuda_device
from api.services.model_downloads import model_manager
from api.services.runtime import runtime_manager

MODEL_ID = 'ltx-2.5-distilled'
SOURCE_REVISION = '9ec55f9f22798a3198d9c923856824821bc3317e'
MODEL_REVISION = 'da86602791e888542b1c755f0b5df376fdb1a3af'
ASSETS = {
    'transformer_path': 'diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors',
    'text_encoder_path': 'text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors',
    'video_vae_path': 'vae/ltx-2.5-video-vae-bf16.safetensors',
    'audio_vae_path': 'vae/ltx-2.5-audio-vae-bf16.safetensors',
    'spatial_upsampler_path': 'latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors',
}


class LTXProvider:
    def __init__(self, manager=None, resources=None, python=None):
        self.manager = manager or model_manager
        self.resources = resources
        self.python = python or os.getenv('KADAN_LTX_PYTHON')

    def validate(self, spec):
        """Reject unsupported controls before creating a job or allocating memory."""
        if spec.negative_prompt.strip():
            raise ValueError('LTX-2.5 distilled does not support a negative prompt.')
        spec.dimensions
        if not self.python or not Path(self.python).is_file():
            raise RuntimeError('The Kadan LTX worker environment is not configured.')
        self._checkpoint()

    def _checkpoint(self):
        """Use only a complete Kadan-managed checkpoint; never download during inference."""
        entry, path = self.manager.get_checkpoint(MODEL_ID)
        if entry.revision != MODEL_REVISION:
            raise RuntimeError('Download the pinned LTX-2.5 checkpoint in Settings first.')
        if any(not (path / name).is_file() for name in ASSETS.values()):
            raise RuntimeError('LTX-2.5 checkpoint components are missing.')
        return path

    def generate(self, spec, output_path: Path, cancellation: threading.Event):
        """Hold shared RAM/VRAM ownership until the native process exits, even on cancellation."""
        self.validate(spec)
        checkpoint = self._checkpoint()
        resources = self.resources or runtime_manager.ensure_resources()
        if not resources.capacity.device_bytes:
            raise RuntimeError('LTX-2.5 requires a CUDA device.')
        device = max(resources.capacity.device_bytes, key=resources.capacity.device_bytes.get)
        owner = f'video-ltx-{uuid.uuid4().hex}'
        # Conservative CPU-offload admission, not a measured peak-memory claim.
        host_bytes = sum((checkpoint / file).stat().st_size for file in ASSETS.values()) * 2
        worker = Path(__file__).parent / 'workers' / 'ltx.py'
        payload = dict(spec=vars(spec), assets={key: str(checkpoint / name) for key, name in ASSETS.items()},
                       output=str(output_path), width=spec.dimensions[0], height=spec.dimensions[1])
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': visible_cuda_device(device), 'HF_HUB_OFFLINE': '1',
               'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
        with resources.exclusive(owner, cancellation):
            reservation = resources.reserve(owner, 'video', host_bytes,
                {device: resources.capacity.device_bytes[device]}, cancel_event=cancellation)
            try:
                with reservation.lease(cancellation):
                    process = subprocess.Popen([self.python, str(worker)], stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, env=env)
                    try:
                        process.stdin.write(json.dumps(payload))
                        process.stdin.close()
                        while process.poll() is None:
                            if cancellation.wait(.1):
                                raise ResourceCancelled('Video generation cancelled')
                        if process.returncode != 0:
                            raise RuntimeError('LTX-2.5 inference failed; verify worker dependencies and memory.')
                        if cancellation.is_set():
                            raise ResourceCancelled('Video generation cancelled')
                        if not output_path.is_file() or output_path.stat().st_size == 0:
                            raise RuntimeError('LTX-2.5 did not produce a video.')
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
            finally:
                reservation.release()
