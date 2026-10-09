"""Offline H3 generation inside the API, under Kadan resource ownership."""
from contextlib import ExitStack
import logging
from pathlib import Path
import shutil
import tempfile
import threading

from api.inference.placement import select_device
from api.inference.resources import ResourceBusy, ResourceCancelled, ResourceExhausted
from api.services.model_downloads import model_manager
from api.services.model_catalog import H3_INT8_REVISION
from api.services.runtime import runtime_manager as runtime
from api.inference.decisions.laya_python import clear_failure_frames

H3_REVISION = H3_INT8_REVISION
SGLANG_REVISION = 'f048d5aa4bc1bcad7fa2c60d067590d83d6dbe4a'
GIB = 1024 ** 3
CONTEXT_BYTES = GIB
log = logging.getLogger(__name__)


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
        self._lock = threading.Lock()
        self._host = self._context = self._device = None
        self._session = None
        self._selected_device = None
        self._full_resident = False

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

    def _dispose_locked(self):
        if self._session is not None:
            self._session.close()
            self._session = None
        for name in ('_device', '_context', '_host'):
            reservation = getattr(self, name)
            if reservation is not None:
                reservation.release()
                setattr(self, name, None)
        self._selected_device = None
        self._full_resident = False

    def _evict(self):
        if not self._lock.acquire(blocking=False):
            raise ResourceBusy('H3 is active')
        try:
            self._dispose_locked()
        finally:
            self._lock.release()

    def close(self):
        with self._lock:
            self._dispose_locked()

    def _park(self):
        if not self._lock.acquire(blocking=False):
            raise ResourceBusy('H3 is active')
        try:
            try:
                if self._session is not None:
                    self._session.park(CONTEXT_BYTES)
            except Exception as exc:
                # An unsupported native cache must not become unaccounted VRAM.
                # Dispose honestly under pressure; the next request reloads.
                log.exception('H3 could not retain a bounded idle session; disposing it')
                clear_failure_frames(exc)
                self._dispose_locked()
                return
            if self._device is not None:
                self._device.release()
                self._device = None
        finally:
            self._lock.release()

    def load(self, cancellation):
        self._use(None, None, cancellation)

    def offload_to_ram(self, cancellation=None):
        runtime.ensure_resources().offload_workload_devices('video', cancellation)

    def generate(self, spec, output_path: Path, cancellation: threading.Event):
        self._use(spec, output_path, cancellation)

    def _use(self, spec, output_path, cancellation):
        """Lease retained host weights, bounded native context and streamed GPU work."""
        if spec is not None:
            self.validate(spec)
        check_media_tools()
        if not self._lock.acquire(blocking=False):
            raise ResourceBusy('H3 is active')
        try:
            entry, checkpoint = model_manager.get_checkpoint(self.model_id)
            if entry.revision != H3_REVISION:
                raise ValueError('H3 checkpoint revision does not match the native provider')
            resources = runtime.ensure_resources()
            devices = resources.capacity.device_bytes
            if not devices:
                raise ResourceExhausted('H3 native inference requires a CUDA GPU')
            selected = self._selected_device
            if selected is None:
                try:
                    selected = int(select_device(resources, entry.estimated_bytes + 12 * GIB + CONTEXT_BYTES)[5:])
                    self._full_resident = True
                except ResourceExhausted:
                    selected = int(select_device(resources, 12 * GIB + CONTEXT_BYTES)[5:])
                    self._full_resident = False
            if devices[selected] < 12 * GIB + CONTEXT_BYTES:
                raise ResourceExhausted('H3 needs 12 GiB of GPU execution budget plus its native context')
            # A single GPU streams the real INT8 host banks. Other inactive model
            # allocations park through their own callbacks before admission.
            with ExitStack() as leases:
                if self._host is None:
                    self._host = resources.reserve('video:h3:host', 'video',
                        host_bytes=entry.estimated_bytes * 2 + 8 * GIB,
                        evict=self._evict, cancel_event=cancellation)
                leases.enter_context(self._host.lease(cancellation))
                if self._context is None:
                    self._context = resources.reserve('video:h3:context', 'video',
                        device_bytes={selected: CONTEXT_BYTES}, evict=self._evict,
                        cancel_event=cancellation, offload_on_handoff=False)
                leases.enter_context(self._context.lease(cancellation))
                if self._device is None:
                    self._device = resources.reserve('video:h3:device', 'video',
                        device_bytes={selected: devices[selected] - CONTEXT_BYTES},
                        evict=self._park, cancel_event=cancellation)
                leases.enter_context(self._device.lease(cancellation))
                self._selected_device = selected
                if spec is None:
                    # Optional native dependencies remain lazy until explicitly loaded.
                    from api.inference.video.h3_pipeline import H3Session
                    if self._session is None:
                        self._session = H3Session(checkpoint, selected)
                    self._session.full_resident = self._full_resident
                    self._session.load(cancellation)
                else:
                    self._run(spec, checkpoint, output_path, cancellation, [selected])
        except BaseException:
            self._dispose_locked()
            raise
        finally:
            self._lock.release()

    def _run(self, spec, checkpoint, output, cancellation, devices):
        """Render synchronously; keep the lease until in-process cleanup completes."""
        # Native imports must follow platform activation, and remain lazy for API startup.
        from api.inference.video.h3_pipeline import H3Session, check_cancel

        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.kadan-h3-', dir=output.parent) as scratch:
            rendered = Path(scratch) / 'video.mp4'
            check_cancel(cancellation)
            if self._session is None:
                self._session = H3Session(checkpoint, devices[0])
            self._session.full_resident = self._full_resident
            self._session.render(sampling_arguments(spec, rendered), cancellation)
            check_cancel(cancellation)
            if rendered.is_symlink() or not rendered.is_file() or not rendered.stat().st_size:
                raise RuntimeError('H3 produced no validated audiovisual output')
            rendered.replace(output)
