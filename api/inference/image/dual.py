"""Opt-in two-rank Torch image adapter; the Redis consumer owns all scheduling."""
import json
import logging
import os
import time

from PIL import Image

from api.inference.image.model import GIB, CONTEXT_BYTES, check_cancel
from api.inference.image.rank_session import RankBudget, RankSession
from api.inference.image.rank_transport import ProcessRanks
from api.inference.resources import ResourceExhausted


class DualImage:
    requested = 'dual'
    offload_mode = 'component'

    def __init__(self, path, resources, *, transport_factory=ProcessRanks, devices=None):
        self.path, self.resources = path, resources
        devices = json.loads(os.environ.get('KADAN_IMAGE_DEVICES', '[0,1]')) if devices is None else devices
        if not isinstance(devices, (list, tuple)):
            raise ValueError('KADAN_IMAGE_DEVICES must be a JSON pair of logical device IDs')
        self.budget = RankBudget(128*GIB, tuple(devices), CONTEXT_BYTES, 20*GIB)
        self.weights = sum(p.stat().st_size for p in path.rglob('*.safetensors'))
        self._plan()
        self.transport = transport_factory(path, self.budget)
        self.session = RankSession(resources, self.transport, self.budget, enabled=True,
            operation_timeout=900, cleanup_timeout=30)
        self.last_timing = None

    def _plan(self):
        if not self.weights or self.weights*3 + 8*GIB > self.budget.host_bytes:
            raise ResourceExhausted('Two-rank checkpoint/staging exceeds the fixed 128 GiB host envelope')
        if self.resources.capacity.host_bytes < self.budget.host_bytes + GIB:
            raise ResourceExhausted('Two-rank image execution requires 129 GiB host admission including publication')
        if any(self.resources.capacity.device_bytes.get(d,0) < self.budget.context_bytes+self.budget.execution_bytes+CONTEXT_BYTES for d in self.budget.devices):
            raise ResourceExhausted('Two-rank images require 21 GiB on each of two visible devices; capacities cannot be pooled')

    @staticmethod
    def validate(prompt, aspect, count, image=None):
        if image is not None or aspect != '1:1' or count != 1:
            raise ValueError('Experimental two-rank images support one 2048-square text-to-image output; edits and other sizes are unavailable')
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
