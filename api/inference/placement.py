"""Pure placement policy shared by native adapters; reservations remain authoritative."""
from dataclasses import dataclass

from api.inference.resources import ResourceExhausted


@dataclass(frozen=True)
class PackedPlacement:
    assignments: dict
    capacities: dict[int, int]
    fully_resident: bool


def place_packed(sizes, available, primary, headroom):
    """Prefer one GPU, then distribute indivisible packed entries; spill excess through LRU.

    Each device keeps its own scratch/activation headroom. This never treats two
    cards as one address space or splits an individual tensor across devices.
    """
    if any(type(size) is not int or size <= 0 for size in sizes.values()):
        raise ValueError('Packed entry sizes must be positive integer bytes')
    caps = {i: max(0, n - headroom.get(i, 0)) for i, n in available.items()}
    total = sum(sizes.values())
    whole = [i for i, n in caps.items() if n >= total]
    if whole:
        selected = primary if primary in whole else max(whole, key=lambda i: (caps[i], -i))
        return PackedPlacement(dict.fromkeys(sizes, selected), {selected: caps[selected]}, True)
    remaining, assigned, full = dict(caps), {}, True
    for key, size in sorted(sizes.items(), key=lambda item: -item[1]):
        choices = [i for i, free in remaining.items() if free >= size]
        if choices:
            selected = max(choices, key=lambda i: (remaining[i], i == primary, -i))
            remaining[selected] -= size
        else:
            choices = [i for i, capacity in caps.items() if capacity >= size]
            if not choices:
                raise ResourceExhausted('No GPU can hold one packed entry plus execution headroom')
            selected = max(choices, key=lambda i: (caps[i], i == primary, -i))
            full = False
        assigned[key] = selected
    return PackedPlacement(assigned, {i: caps[i] for i in set(assigned.values())}, full)


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
    if allow_cpu:
        return 'cpu'
    raise ResourceExhausted('No GPU can hold this adapter’s whole-model execution budget; its tensor splitting is unsupported')
