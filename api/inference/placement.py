"""Whole-model device selection; reservations remain authoritative."""
from api.inference.resources import ResourceExhausted


def select_device(resources, required_bytes, requested='auto', *, allow_cpu=False, retained=None):
    """Select a supported whole-model target; account retained ownership and fresh free bytes.

    Prefer an already resident model, then available GPU space, then reclaimable
    idle allocations. Selection does not evict or reserve; the adapter must admit
    before allocating and must honor admission failure if physical memory changes.
    """
    if type(required_bytes) is not int or required_bytes <= 0:
        raise ValueError('Execution budget must be positive integer bytes')
    if requested == 'cpu':
        if not allow_cpu:
            raise ResourceExhausted('This adapter requires CUDA')
        return 'cpu'
    if requested != 'auto':
        if not requested.startswith('cuda:') or not requested[5:].isdecimal():
            raise ValueError('Device must be auto, cpu, or cuda:N')
        index = int(requested[5:])
        if resources.capacity.device_bytes.get(index, 0) < required_bytes:
            raise ResourceExhausted('Execution exceeds the explicitly selected GPU budget')
        return requested
    for reclaim in (False, True):
        available = resources.available_devices(reclaim=reclaim)
        if retained is not None:
            index, accounted = retained
            if index in available:
                available[index] = min(resources.capacity.device_bytes[index], available[index] + accounted)
                if available[index] >= required_bytes:
                    return f'cuda:{index}'
        choices = [i for i, n in available.items() if n >= required_bytes]
        if choices:
            return f'cuda:{max(choices, key=lambda i: (available[i], -i))}'
    # Feasible capacity under transient contention waits in reserve(), rather
    # than becoming a permanent adapter-capability failure before admission.
    choices = [i for i, n in resources.capacity.device_bytes.items() if n >= required_bytes]
    if choices:
        return f'cuda:{max(choices, key=lambda i: (resources.capacity.device_bytes[i], -i))}'
    if allow_cpu:
        return 'cpu'
    raise ResourceExhausted('No GPU can hold this adapter’s whole-model execution budget; its tensor splitting is unsupported')
