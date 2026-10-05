"""Helpers for Kadan-owned isolated inference processes."""
import os


def visible_cuda_device(logical_index: int) -> str:
    """Keep parent CUDA device remapping when a child is restricted to one GPU."""
    visibility = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visibility is None:
        return str(logical_index)
    devices = [device.strip() for device in visibility.split(',')]
    if logical_index >= len(devices) or not devices[logical_index]:
        raise RuntimeError('The selected CUDA device is no longer visible.')
    return devices[logical_index]
