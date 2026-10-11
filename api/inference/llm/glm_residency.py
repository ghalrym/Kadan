"""GLM-specific planning with the shared resident C++ protocol and global FIFO."""
from contextlib import contextmanager
import re

from api.inference.llm.context import ContextLimitError, ContextMemoryError, resolve_context
from api.inference.llm.qwen_residency import ResidentQwenAdapter
from api.inference.llm.qwen_subprocess import MIB, PYTHON_HOST_BYTES, numbers
from api.inference.line_protocol import LineProtocolError, LineProtocolProcess
from api.inference.resources import ResourceBusy, ResourceExhausted


class ResidentGlmAdapter(ResidentQwenAdapter):
    # Preserve the checkpoint's reasoning template and make its already-open
    # reasoning span explicit in streamed/non-streamed text. No silent template edits.
    output_prefix = '<think>'
    chat_template_options = {}
    checkpoint_id = 'large'
    model_type = 'glm5_next'
    config_byte_limit = 16 * MIB
    binary_environment = 'KADAN_GLM_WORKER'
    default_binary = '/opt/kadan/bin/kadan-glm-worker'

    def _configure_context_locked(self, configured):
        self.check_execution_state()
        if self._closed:
            raise RuntimeError('Native adapter is closed')
        supported, effective = resolve_context(self.config, configured)
        if effective > 1048576:
            raise ContextLimitError('Native GLM supports up to the checkpoint maximum of 1048576 context tokens.')
        if self.worker is not None:
            if effective != self.capacity:
                raise ContextLimitError('Unload the native worker before changing context')
            return
        self.configured_context_limit = configured
        self.supported_context_limit, self.effective_context_limit = supported, effective
        self.capacity = effective
        # Exact fixed-shape state plan, mirrored and checked by header-only C++ planning.
        host_bytes = ((512 + 128) * MIB + 34 * (64*128*128 + 3*8192*4)*4
                      + 11 * (effective*512 + ((effective+3)//4)*128 + 8*128)*4)
        try:
            self.host = self.resources.reserve(self.owner + ':host', 'llm',
                host_bytes=host_bytes + PYTHON_HOST_BYTES, evict=self._evict, cancel_event=self.cancel)
            with self.host.lease(self.cancel):
                limits = dict(self.resources.capacity.device_bytes)
                if self.device != 'auto':
                    if self.device not in ('cuda:0', 'cuda:1'):
                        raise ValueError('GLM requires auto, cuda:0 or cuda:1')
                    index = int(self.device[-1])
                    limits = {index: limits.get(index, 0)}
                available = self.resources.available_devices(reclaim=True)
                budgets = {d: min(b, available.get(d, 0)) - 64*MIB for d, b in limits.items()
                           if d in (0, 1) and min(b, available.get(d, 0)) >= 832*MIB}
                if not budgets:
                    budgets = {d: min(b, 768*MIB) for d, b in limits.items() if d in (0, 1) and b >= 768*MIB}
                if not budgets:
                    raise ResourceExhausted('No CUDA device can hold GLM projection scratch and context.')
                arg = ','.join(f'{d}:{b}' for d, b in sorted(budgets.items()))
                self.worker = LineProtocolProcess()
                self.worker.start([str(self.binary), '--plan-glm', str(self.root), str(effective), arg])
                line = self.worker.read(self.load_timeout, self.cancel)
                fields = numbers(line, ['plan', '5'], len(line.split()) - 2)
                self.worker.finish()
                self.worker.stop()
                self.worker = None
                if len(fields) < 9:
                    raise LineProtocolError('Invalid GLM plan')
                host, arena, vocab, capacity, packed, tensors, count = fields[:7]
                if (host != host_bytes or vocab != 154880 or capacity != effective or arena <= 0
                        or not 0 < packed <= 1024**4 or not 0 < tensors <= 131072
                        or count != len(budgets) or len(fields) != 7 + 2*count
                        or dict(zip(fields[7::2], fields[8::2])) != budgets):
                    raise LineProtocolError('Native GLM plan violates its admitted bounds')
                self.execution_bytes = budgets
                self.vocabulary, self.arena = vocab, arena
                self.packed_weight_bytes, self.packed_tensor_count = packed, tensors
                self.cache_capacity = 0  # direct bounded disk reads; no mandatory RAM cache
                self._reserve_arena(self.cancel)
                with self.reservation.lease(self.cancel):
                    self.tokenizer = self.tokenizer_factory(self.root)
                    self.worker = LineProtocolProcess()
                    self.worker.start([str(self.binary), '--serve-glm', str(self.root), str(effective), arg,
                                       f'{host}:{packed}:{tensors}:{arena}'])
                    ready = self.worker.read(self.load_timeout, self.cancel).split()
                    if (len(ready) != 5 or ready[:2] != ['ready', '2']
                            or not re.fullmatch('[0-9a-f]{32}', ready[2])
                            or numbers(' '.join(ready[3:]), [], 2) != [vocab, effective]):
                        raise LineProtocolError('Native GLM readiness differs from the admitted plan')
                    self.session = ready[2]
                    self._read_cache_stats(self.cancel)
                    self.last_id = self.request_id = 0
                    self.parked = False
        except BaseException:
            self._close_locked()
            raise

    @contextmanager
    def _leases(self, cancel):
        with self.host.lease(cancel), self.reservation.lease(cancel):
            yield

    def _restore_locked(self, cancel):
        self.check_execution_state()
        if self.parked and self.worker_alive:
            with self.host.lease(cancel):
                try:
                    self._reserve_arena(cancel)
                except (ResourceBusy, ResourceExhausted) as error:
                    raise ContextMemoryError(str(error)) from error
            self.parked = False


def build_resident_glm(entry, path, resources, device='auto', cancel_event=None):
    return ResidentGlmAdapter(entry, path, resources, device, cancel_event)
