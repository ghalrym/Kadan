"""Native image IPC and retained CPU ownership; model execution stays in C++."""
from contextlib import ExitStack
import base64
import io
import json
import os
from pathlib import Path
import secrets
import shutil
import tempfile
import threading
import uuid

from PIL import Image

from api.inference.errors import InferenceFailure
from api.inference.native_compute import NativeCompute, native_compute
from api.inference.line_protocol import LineProtocolError, LineProtocolProcess
from api.inference.resources import ResourceBusy, ResourceCancelled
from api.services.model_downloads import model_manager

MODEL = 'qwen-image-2.1'
REVISION = 'd26bb61231c349cf6b7896fa83353113880e1ba3'
HOST_BUDGET = 80 * 1024**3
PROCESS_BUDGET = HOST_BUDGET + 256 * 1024**2
SIZES = {'1:1': (2048, 2048), '4:3': (2400, 1792), '3:4': (1792, 2400), '16:9': (2752, 1536)}


def check_cancel(cancel):
    if cancel.is_set():
        raise ResourceCancelled('Image generation cancelled')


def validate(request):
    if (not request.prompt.strip() or len(request.prompt.encode('utf-8')) > 8000
            or request.aspect not in SIZES or request.count not in (1, 2, 4)
            or not 2 <= getattr(request, "steps", 50) <= 100
            or (request.seed is not None and not 0 <= request.seed < 2**64)):
        raise InferenceFailure('Unsupported native image request bounds.', 422)


def prepare():
    value = os.environ.get('KADAN_NATIVE_IMAGE_WORKER') or '/opt/kadan/bin/kadan-image-worker'
    binary = Path(value)
    if not binary.is_absolute() or not binary.is_file():
        raise InferenceFailure('Configure an absolute native image worker path.')
    try:
        entry, checkpoint = model_manager.get_checkpoint(MODEL)
    except ValueError as error:
        raise InferenceFailure(str(error)) from error
    if entry.revision != REVISION:
        raise InferenceFailure('Native image checkpoint revision mismatch.')
    return Path(checkpoint), binary


class NativeImageSession:
    def __init__(self, checkpoint, binary, process_factory=LineProtocolProcess):
        self.checkpoint, self.binary, self.process_factory = checkpoint, binary, process_factory
        self.child = self.workspace = self.baseline = None
        self.quarantined = False
        self.compute = NativeCompute("IMAGE")
        self.parked = False

    def check_execution_state(self):
        if self.quarantined:
            raise InferenceFailure('Native image cleanup is unconfirmed; retry unload before another job.')

    def load(self, cancel):
        self.check_execution_state()
        check_cancel(cancel)
        self.workspace = Path(tempfile.mkdtemp(prefix='kadan-image-'))
        self.child = self.process_factory()
        self.child.start([str(self.binary), str(self.checkpoint), str(self.workspace)],
                         env=self.compute.environment())
        try:
            ready = self.child.read(1200, cancel).split()
        except InterruptedError as error:
            raise ResourceCancelled('Image loading cancelled') from error
        if len(ready) != 3 or ready[:2] != ['ready', '1'] or not ready[2].isdigit() or not 0 < int(ready[2]) <= HOST_BUDGET:
            raise LineProtocolError('Invalid native image ready response')
        self.baseline = int(ready[2])

    def restore(self, cancel):
        if self.parked:
            if self.child.exchange('resume', 60, cancel).split() != ['resumed']:
                raise LineProtocolError('Native image resume was not acknowledged')
            self.parked = False

    def offload_to_ram(self):
        if self.compute.devices and not self.parked:
            if self.child.exchange('park', 60, threading.Event()).split() != ['parked']:
                raise LineProtocolError('Native image GPU cleanup was not acknowledged')
            self.parked = True

    def generate(self, request, cancel):
        self.check_execution_state()
        validate(request)
        check_cancel(cancel)
        width, height = SIZES[request.aspect]
        steps = getattr(request, 'steps', 50)
        seed = request.seed if request.seed is not None else secrets.randbits(32)
        images, seeds = [], []
        for index in range(request.count):
            check_cancel(cancel)
            current = (seed + index) % 2**64
            (self.workspace / 'request.json').write_text(json.dumps(dict(prompt=request.prompt,
                width=width, height=height, steps=steps, seed=current)), encoding='utf-8')
            output = self.workspace / 'image.png'
            output.unlink(missing_ok=True)
            try:
                reply = self.child.exchange('generate', 7200, cancel).split()
            except InterruptedError as error:
                raise ResourceCancelled('Image generation cancelled') from error
            if reply != ['done', str(self.baseline)]:
                raise LineProtocolError('Invalid native image completion response')
            if output.is_symlink() or not output.is_file() or not 0 < output.stat().st_size <= width * height * 4 + 1024**2:
                raise LineProtocolError('Invalid native image artifact')
            data = output.read_bytes()
            with Image.open(io.BytesIO(data)) as image:
                if image.format != 'PNG' or image.mode != 'RGBA' or image.size != (width, height):
                    raise LineProtocolError('Invalid native image dimensions')
                image.load()
            check_cancel(cancel)
            images.append(base64.b64encode(data).decode('ascii'))
            seeds.append(current)
        return {'image': dict(id=uuid.uuid4().hex, mode='Generate', prompt=request.prompt,
            aspect={'1:1': 'square', '4:3': 'landscape', '3:4': 'portrait', '16:9': 'wide'}[request.aspect],
            seeds=seeds, meta=f'{width}×{height} · {steps} steps · native {self.compute.mode.upper()}',
            images_base64=images, mime_type='image/png')}

    def unload(self):
        if self.child is not None:
            try:
                self.child.stop()
            except BaseException:
                self.quarantined = True
                raise
        if self.workspace is not None:
            shutil.rmtree(self.workspace)
        self.child = self.workspace = self.baseline = None
        self.quarantined = False
        self.parked = False


class NativeImageRuntime:
    """One resident under the API's shared ResourceManager and existing FIFO."""
    def __init__(self, resources, prepare_plan=prepare, session_factory=NativeImageSession):
        self.resources, self.prepare_plan, self.session_factory = resources, prepare_plan, session_factory
        self.session = self.reservation = self.identity = self.cancel = None
        self.gate = threading.Lock()
        self.device_reservation = None

    def check_execution_state(self):
        if self.session is not None:
            self.session.check_execution_state()

    def _unload(self):
        if self.session is not None:
            self.session.unload()
        if self.device_reservation is not None:
            self.device_reservation.release()
            self.device_reservation = None
        if self.reservation is not None:
            self.reservation.release()
        self.session = self.reservation = self.identity = None

    def _park(self):
        if not self.gate.acquire(blocking=False):
            raise ResourceBusy('Image worker is active')
        try:
            self.session.offload_to_ram()
            if self.device_reservation is not None:
                self.device_reservation.release()
                self.device_reservation = None
        finally:
            self.gate.release()

    def offload_to_ram(self, cancel=None):
        self.resources().offload_workload_devices('image', cancel)

    def _evict(self):
        if not self.gate.acquire(blocking=False):
            raise ResourceBusy('Image worker is active')
        try:
            self._unload()
        finally:
            self.gate.release()

    def unload(self):
        while not self.gate.acquire(timeout=.05):
            if self.cancel is not None:
                self.cancel.set()
        try:
            self._unload()
        finally:
            self.gate.release()

    def run(self, request, cancel):
        if request is not None:
            validate(request)
        if not self.gate.acquire(blocking=False):
            raise ResourceBusy('Image worker is active')
        self.cancel = cancel
        try:
            check_cancel(cancel)
            self.check_execution_state()
            plan = self.prepare_plan()
            resources = self.resources()
            compute = native_compute('IMAGE', resources)
            identity = (plan, compute.identity)
            resources.offload_inactive_devices('image', cancel)
            if self.identity != identity:
                self._unload()
                self.reservation = resources.reserve('image-host-' + uuid.uuid4().hex, 'image',
                    host_bytes=PROCESS_BUDGET, evict=self._evict, cancel_event=cancel)
            try:
                with ExitStack() as leases:
                    leases.enter_context(self.reservation.lease(cancel))
                    if compute.devices and self.device_reservation is None:
                        self.device_reservation = resources.reserve('image-device-' + uuid.uuid4().hex, 'image',
                            device_bytes=compute.device_bytes, evict=self._park, cancel_event=cancel)
                    if self.device_reservation is not None:
                        leases.enter_context(self.device_reservation.lease(cancel))
                    if self.session is None:
                        self.session = self.session_factory(*plan)
                        self.session.compute = compute
                        self.session.load(cancel)
                        self.identity = identity
                    restore = getattr(self.session, 'restore', None)
                    if restore is not None:
                        restore(cancel)
                    check_cancel(cancel)
                    return self.session.generate(request, cancel) if request is not None else None
            except BaseException:
                self._unload()
                raise
        finally:
            self.cancel = None
            self.gate.release()
