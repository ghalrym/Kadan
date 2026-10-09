"""Opt-in two-rank Torch image adapter; the Redis consumer owns all scheduling."""
import json
import logging
import os
import time

from PIL import Image

from api.inference.image.qwen_image_pipeline import GIB, CONTEXT_BYTES, check_cancel
from api.inference.image.execution_policy import ImageExecutionPolicy
from api.inference.image.rank_residency import ImageRankBudget, ImageRankResidency
from api.inference.image.rank_processes import ImageRankProcesses
from api.inference.resources import ResourceExhausted


class TwoRankQwenImage:
    requested = 'dual'
    offload_mode = 'component'

    def __init__(self, path, resources, *, transport_factory=ImageRankProcesses, devices=None):
        self.path, self.resources = path, resources
        devices = json.loads(os.environ.get('KADAN_IMAGE_DEVICES', '[0,1]')) if devices is None else devices
        if not isinstance(devices, (list, tuple)):
            raise ValueError('KADAN_IMAGE_DEVICES must be a JSON pair of logical device IDs')
        self.policy = ImageExecutionPolicy.from_environment()
        self.weights = sum(p.stat().st_size for p in path.rglob('*.safetensors'))
        self.budget = ImageRankBudget(self.policy.host_budget(self.weights, 3), tuple(devices),
            CONTEXT_BYTES, self.policy.execution_bytes)
        self._plan()
        self.transport = transport_factory(path, self.budget, self.policy)
        self.session = ImageRankResidency(resources, self.transport, self.budget, enabled=True,
            operation_timeout=self.policy.operation_seconds, cleanup_timeout=self.policy.cleanup_seconds)
        self.last_timing = None

    def _plan(self):
        # Reject deployment errors before the FIFO parks another resident model.
        # ImageRankProcesses.start rechecks the mask in case it changes after preflight.
        self.policy.affinity(os.sched_getaffinity(0))
        if not self.weights:
            raise ResourceExhausted('The two-rank checkpoint has no weights')
        required_host = self.budget.host_bytes + GIB
        if self.resources.capacity.host_bytes < required_host:
            raise ResourceExhausted(f'Two-rank image execution requires {required_host} host bytes including publication')
        required_device = self.budget.context_bytes + self.budget.execution_bytes + CONTEXT_BYTES
        if any(self.resources.capacity.device_bytes.get(device, 0) < required_device
                for device in self.budget.devices):
            raise ResourceExhausted(f'Two-rank images require {required_device} bytes on each visible device; capacities cannot be pooled')

    @staticmethod
    def validate(prompt, aspect, count, image=None):
        if image is not None or aspect != '1:1' or count != 1:
            raise ValueError('Two-rank images support one 2048-square text-to-image output; edits and other sizes are unavailable')
        if not isinstance(prompt,str) or not 0 < len(prompt) <= 2000:
            raise ValueError('Two-rank prompt must contain 1 to 2000 characters')

    def load(self, cancel=None):
        check_cancel(cancel)
        self._plan()
        return self  # Actual construction belongs to the admitted FIFO execute.

    def generate(self, prompt, aspect, seeds, cancel, image=None, *, job_id=None):
        self.validate(prompt, aspect, len(seeds), image)
        if job_id is None:
            raise ValueError('Two-rank execution requires the global FIFO job ID')
        for device in self.budget.devices:
            self.resources.framework_context(device, CONTEXT_BYTES)
        started = time.monotonic()
        previous = self.session.state
        receipts = self.session.execute(job_id, cancel, payload=dict(prompt=prompt,seed=seeds[0]))
        try:
            check_cancel(cancel)
            path = self.transport.output()
            if path.is_symlink() or path.stat().st_size > 24*1024**2:
                raise ValueError('Invalid bounded rank image output')
            with Image.open(path) as image:
                if image.mode != 'RGBA' or image.size != (2048,2048):
                    raise ValueError('Invalid rank image dimensions or format')
                image.load()
                picture = image.copy()
            path.unlink()
            self.last_timing = dict(previous_state=previous, total_seconds=time.monotonic()-started,
                rank_request_seconds=[r['request_seconds'] for r in receipts],
                load_or_restore_seconds=self.session.prepare_seconds)
            logging.getLogger(__name__).info("dual_image_timing %s", json.dumps(dict(job=job_id, **self.last_timing)))
            return [picture]
        except BaseException:
            self.session.close()
            raise

    def offload_to_ram(self, cancel=None):
        check_cancel(cancel)
        self.session.park(cancel)

    def close(self):
        self.session.close()
