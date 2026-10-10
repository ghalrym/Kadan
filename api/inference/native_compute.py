"""Explicit, frozen native-worker placement; no model execution or CUDA calls."""
from dataclasses import dataclass
import os

from api.inference.errors import InferenceFailure

DEFAULT_DEVICE_BYTES = 2 * 1024**3 + 384 * 1024**2


@dataclass(frozen=True)
class NativeCompute:
    feature: str
    devices: tuple[int, ...] = ()
    budget: int = DEFAULT_DEVICE_BYTES

    @property
    def device_bytes(self):
        return {device: self.budget for device in self.devices}

    @property
    def mode(self):
        return 'cuda' if self.devices else 'cpu'

    @property
    def identity(self):
        return (self.mode, ','.join(map(str, self.devices)), str(self.budget))

    def environment(self):
        return {**os.environ, f'KADAN_{self.feature}_DEVICES': ','.join(map(str, self.devices)),
                'KADAN_NATIVE_GPU_BUDGET_BYTES': str(self.budget),
                'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}


def native_compute(feature, resources=None):
    selection = os.environ.get(f'KADAN_{feature}_DEVICES') or 'auto'
    raw = os.environ.get('KADAN_NATIVE_GPU_BUDGET_BYTES', str(DEFAULT_DEVICE_BYTES))
    if not raw.isascii() or not raw.isdecimal() or not 400 * 1024**2 <= int(raw) <= 24 * 1024**3:
        raise InferenceFailure('Invalid native worker GPU budget.')
    budget = int(raw)
    capacity = resources.snapshot()['device_capacity_bytes'] if resources is not None else None
    if selection == 'auto':
        if capacity is None:
            raise InferenceFailure('Automatic native placement requires the shared resource manager.')
        # Freeze a feasible placement; admission waits for transient pressure.
        devices = tuple(d for d in (0, 1) if capacity.get(d, 0) >= budget)
    elif selection in ('cpu', '0', '1', '0,1', '1,0'):
        devices = tuple(map(int, selection.split(','))) if selection != 'cpu' else ()
    else:
        raise InferenceFailure('Native devices must be 0, 1, 0,1, 1,0 auto or cpu.')
    if capacity is not None and any(capacity.get(d, 0) < budget for d in devices):
        raise InferenceFailure('Selected native devices exceed the shared device budgets.')
    return NativeCompute(feature, devices, budget)
