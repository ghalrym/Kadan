"""Resident Torch/Diffusers Qwen Image adapter; ResourceManager owns all admission."""
import gc
import importlib
import threading
from uuid import uuid4

from PIL import Image

from api.inference.decisions.model import clear_failure_frames
from api.inference.resources import ResourceBusy, ResourceCancelled, ResourceExhausted

MODEL_ID = 'qwen-image-2.1'
REVISION = 'd26bb61231c349cf6b7896fa83353113880e1ba3'
SIZES = {'1:1': (2048, 2048), '4:3': (2400, 1792), '3:4': (1792, 2400), '16:9': (2752, 1536)}
GIB = 1024 ** 3
WORKSPACE = 8 * GIB
CONTEXT_BYTES = 512 * 1024**2


def native_modules():
    """Keep optional native imports out of API startup and catalog inspection."""
    return importlib.import_module('torch'), importlib.import_module('diffusers')


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise ResourceCancelled('Image generation cancelled')


class NativeImage:
    def __init__(self, path, resources, device='auto', modules=native_modules, offload_mode='sequential'):
        self.path, self.resources, self.requested, self.modules = path, resources, device, modules
        if offload_mode not in ('sequential', 'component'):
            raise ValueError('Image offload mode must be sequential or component')
        self.offload_mode = offload_mode
        self.component_weights = {name: sum(item.stat().st_size for item in (path/name).glob('*.safetensors'))
            for name in ('text_encoder', 'transformer', 'vae')}
        self.owner = 'image:qwen:' + uuid4().hex
        self.pipeline = self.host = self.gpu = self.torch = None
        self.device = None
        self.offloaded = False
        self.cleanup_failed = False
        self.weights = sum(item.stat().st_size for item in path.rglob('*.safetensors'))
        if not self.weights:
            raise ValueError('The Qwen Image checkpoint has no weights')
        self.gate = threading.RLock()

    def _plan(self):
        if self.requested == 'cpu':
            return 'cpu', 0, False
        capacities = self.resources.capacity.device_bytes
        if self.requested != 'auto':
            if not self.requested.startswith('cuda:') or not self.requested[5:].isdecimal():
                raise ValueError('KADAN_IMAGE_DEVICE must be auto, cpu or cuda:<index>')
            candidates = [int(self.requested[5:])]
        else:
            candidates = sorted(capacities, key=capacities.get, reverse=True)
        # Use the shared-placement view when available. On master, reserve()
        # remains authoritative and rechecks live capacity before any allocation.
        available = getattr(self.resources, 'available_devices', None)
        if available is not None:
            free = available()
        else:
            snapshot = self.resources.snapshot()
            free = {gpu: max(0, size - sum(row['device_bytes'].get(gpu, 0)
                for row in snapshot['reservations'].values())) for gpu, size in capacities.items()}
        reserved = self.resources.snapshot()['reservations']
        overhead = {gpu: 0 if f'framework-context:{gpu}' in reserved else CONTEXT_BYTES for gpu in candidates}
        free = {gpu: max(0, free.get(gpu, 0) - overhead[gpu]) for gpu in candidates}
        capacities = {gpu: max(0, capacities.get(gpu, 0) - CONTEXT_BYTES) for gpu in candidates}
        if self.offload_mode == 'component':
            if not all(self.component_weights.values()):
                raise ValueError('Component offload requires all three pinned checkpoint components')
            # Experimental comparison mode: reserve the entire logical phase
            # envelope, including activations, retained embeddings, allocator
            # workspace and transfer peaks. Never reuse the 8 GiB leaf envelope.
            for gpu in candidates:
                held = sum(row['device_bytes'].get(gpu, 0) for row in reserved.values()
                    if row['active_leases'] or not row['offload_on_handoff'])
                budget = self.resources.capacity.device_bytes.get(gpu, 0) - held - overhead[gpu]
                if budget > max(self.component_weights.values()):
                    return f'cuda:{gpu}', budget, True
            raise ResourceExhausted('No single-device phase envelope fits the largest image component')
        whole = self.weights + WORKSPACE
        for gpu in candidates:
            if free.get(gpu, 0) >= whole:
                return f'cuda:{gpu}', whole, False
        # A single-GPU checkpoint may fit after pressure eviction. Do not add
        # separate cards' capacity or pretend this pipeline supports sharding.
        for gpu in candidates:
            if capacities.get(gpu, 0) >= whole:
                return f'cuda:{gpu}', whole, False
        for gpu in candidates:
            if capacities.get(gpu, 0) >= WORKSPACE:
                return f'cuda:{gpu}', WORKSPACE, True
        raise ResourceExhausted('Qwen Image requires at least 8 GiB on one GPU, or explicit CPU execution')

    def load(self, cancel=None):
        with self.gate:
            check_cancel(cancel)
            if self.cleanup_failed:
                raise ResourceBusy('Image cleanup is uncertain; close before reuse')
            if self.pipeline is not None:
                return self
            self.device, _, _ = self._plan()
            self.torch, diffusers = self.modules()
            pipeline_type = getattr(diffusers, 'QwenImage21Pipeline', None)
            if pipeline_type is None:
                raise RuntimeError('Install the pinned Qwen Image runtime requirements')
            try:
                # Include CPU construction, parked weights, edit input and CPU
                # output staging. GPU execution gets a separate reservation.
                self.host = self.resources.reserve(self.owner + ':host', 'image',
                    host_bytes=self.weights * 2 + WORKSPACE, evict=self.close, cancel_event=cancel)
                with self.host.lease(cancel):
                    self.pipeline = pipeline_type.from_pretrained(str(self.path), local_files_only=True,
                        torch_dtype=self.torch.float32 if self.device == 'cpu' else self.torch.bfloat16,
                        use_safetensors=True)
                    self.pipeline.vae.enable_tiling()
                    check_cancel(cancel)
                return self
            except BaseException as exc:
                clear_failure_frames(exc)
                self._close()
                raise

    def _sync(self):
        if self.gpu is not None and self.device != 'cpu' and self.torch is not None:
            self.torch.cuda.synchronize(int(self.device[5:]))

    def _park(self):
        if self.gpu is None:
            return
        self._sync()
        if self.pipeline is not None:
            if self.offloaded:
                # Accelerate detach restores original CPU tensors from its map.
                self.pipeline.remove_all_hooks()
            self.pipeline.to('cpu')
        self._sync()
        self.offloaded = False
        with self.torch.cuda.device(int(self.device[5:])):
            self.torch.cuda.empty_cache()
        self.gpu.release()
        self.gpu = None

    def offload_to_ram(self, cancel=None):
        if not self.gate.acquire(blocking=False):
            raise ResourceBusy('Image pipeline is active')
        try:
            self._park()
        except BaseException as exc:
            clear_failure_frames(exc)
            self.cleanup_failed = True
            raise
        finally:
            self.gate.release()

    def _close(self):
        try:
            self._close_allocations()
            self.cleanup_failed = False
        except BaseException as exc:
            clear_failure_frames(exc)
            self.cleanup_failed = True
            raise

    def _close_allocations(self):
        # Clear the pipeline and exception frames before releasing accounting.
        self.pipeline = None
        gc.collect()
        self._sync()
        if self.gpu is not None:
            with self.torch.cuda.device(int(self.device[5:])):
                self.torch.cuda.empty_cache()
            self.gpu.release()
            self.gpu = None
        if self.host is not None:
            self.host.release()
            self.host = None
        self.offloaded = False

    def close(self):
        if not self.gate.acquire(blocking=False):
            raise ResourceBusy('Image pipeline is active')
        try:
            self._close()
        finally:
            self.gate.release()

    def generate(self, prompt, aspect, seeds, cancel, image=None):
        with self.gate:
            result = None
            try:
                self.load(cancel)
                with self.host.lease(cancel):
                    if self.device != 'cpu' and self.gpu is None:
                        self.device, budget, self.offloaded = self._plan()
                        self.resources.framework_context(int(self.device[5:]), CONTEXT_BYTES)
                        self.gpu = self.resources.reserve(self.owner + ':gpu', 'image',
                            device_bytes={int(self.device[5:]): budget}, evict=self.offload_to_ram,
                            cancel_event=cancel)
                    # Host-only execution uses an additional lease on the same
                    # reservation, preserving one ownership path for both modes.
                    with (self.gpu or self.host).lease(cancel):
                        if self.gpu is not None:
                            if self.offloaded:
                                if self.offload_mode == 'component':
                                    if self.pipeline.model_cpu_offload_seq != 'text_encoder->transformer->vae':
                                        raise RuntimeError('Unexpected component offload order')
                                    self.pipeline.enable_model_cpu_offload(gpu_id=int(self.device[5:]))
                                else:
                                    self.pipeline.enable_sequential_cpu_offload(gpu_id=int(self.device[5:]))
                            else:
                                self.pipeline.to(self.device)
                        def checkpoint(_pipeline, _step, _timestep, values):
                            check_cancel(cancel)
                            return values
                        width, height = SIZES[aspect]
                        pictures = []
                        for seed in seeds:
                            check_cancel(cancel)
                            options = dict(prompt=prompt, width=width, height=height,
                                num_inference_steps=40, num_images_per_prompt=1,
                                generator=self.torch.Generator(device='cpu').manual_seed(seed),
                                callback_on_step_end=checkpoint)
                            if image is not None:
                                options['image'] = image
                            result = self.pipeline(**options)
                            if len(result.images) != 1 or not isinstance(result.images[0], Image.Image):
                                raise RuntimeError('Qwen Image returned an unexpected image count')
                            pictures.append(result.images[0])
                            result = None
                        self._sync()
                        check_cancel(cancel)
                    # Sequential hooks are per-request; restore a reusable CPU
                    # pipeline. Fully resident mode stays until pressure/offload.
                    if self.offloaded:
                        self._park()
                    return pictures
            except BaseException as exc:
                result = None
                clear_failure_frames(exc)
                self._close()
                raise
