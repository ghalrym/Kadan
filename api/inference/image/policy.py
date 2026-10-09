"""Validated deployment settings for image admission and rank execution.

Byte settings are integer bytes, deadlines are finite positive seconds. Defaults
retain the supported model's execution envelopes; overrides may increase them,
not claim that unvalidated smaller envelopes work. CPU affinity and thread counts
inherit the deployment unless explicitly set. CUDA wait scheduling is unchanged
unless blocking sync is requested. These settings do not broaden model support.
"""
from dataclasses import dataclass
import json
import math
import os
from typing import Mapping

GIB = 1024 ** 3


def byte_budget(environment: Mapping[str, str], name: str, minimum: int) -> int:
    """Read an integer reservation no smaller than the supported model estimate."""
    raw = environment.get(name, str(minimum))
    if not raw.isascii() or not raw.isdecimal() or int(raw) < minimum:
        raise ValueError(f'{name} must be an integer byte budget of at least {minimum}')
    return int(raw)


@dataclass(frozen=True)
class ImagePolicy:
    workspace_bytes: int
    host_bytes: int | None
    execution_bytes: int
    cpus: tuple[int, ...] | None
    threads: int | None
    operation_seconds: float
    collective_seconds: float
    cleanup_seconds: float
    blocking_sync: bool

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> 'ImagePolicy':
        """Validate KADAN_IMAGE_* settings before reserving memory or starting ranks.

        WORKSPACE_BYTES defaults to 8 GiB. HOST_BYTES, when set, must cover the
        checkpoint-derived host estimate. DUAL_EXECUTION_BYTES defaults to 20 GiB
        per rank, excluding context. CPUS is a JSON array of distinct allowed CPU
        IDs; THREADS is a positive per-rank Torch/BLAS count. Unset values inherit.
        OPERATION_SECONDS/COLLECTIVE_SECONDS/CLEANUP_SECONDS default to 900/120/30;
        the collective bound cannot exceed the encompassing operation deadline.
        CUDA_WAIT accepts default or blocking; default preserves driver policy.
        """
        env = os.environ if environment is None else environment
        workspace = byte_budget(env, 'KADAN_IMAGE_WORKSPACE_BYTES', 8 * GIB)
        host = byte_budget(env, 'KADAN_IMAGE_HOST_BYTES', 1) if 'KADAN_IMAGE_HOST_BYTES' in env else None
        execution = byte_budget(env, 'KADAN_IMAGE_DUAL_EXECUTION_BYTES', 20 * GIB)
        cpus = None
        if 'KADAN_IMAGE_CPUS' in env:
            try:
                values = json.loads(env['KADAN_IMAGE_CPUS'])
                if (not isinstance(values, list) or not values
                        or any(type(cpu) is not int or cpu < 0 for cpu in values)
                        or len(set(values)) != len(values)):
                    raise ValueError()
                cpus = tuple(sorted(values))
            except (ValueError, TypeError) as exc:
                raise ValueError('KADAN_IMAGE_CPUS must be a nonempty JSON array of distinct CPU IDs') from exc
        threads = byte_budget(env, 'KADAN_IMAGE_THREADS', 1) if 'KADAN_IMAGE_THREADS' in env else None
        seconds = []
        for suffix, default in [('OPERATION', 900), ('COLLECTIVE', 120), ('CLEANUP', 30)]:
            name = f'KADAN_IMAGE_{suffix}_SECONDS'
            try:
                value = float(env.get(name, str(default)))
                if not math.isfinite(value) or value <= 0:
                    raise ValueError()
            except ValueError as exc:
                raise ValueError(f'{name} must be finite positive seconds') from exc
            seconds.append(value)
        if seconds[1] > seconds[0]:
            raise ValueError('KADAN_IMAGE_COLLECTIVE_SECONDS cannot exceed KADAN_IMAGE_OPERATION_SECONDS')
        wait = env.get('KADAN_IMAGE_CUDA_WAIT', 'default')
        if wait not in ('default', 'blocking'):
            raise ValueError('KADAN_IMAGE_CUDA_WAIT must be default or blocking')
        return cls(workspace, host, execution, cpus, threads, *seconds, wait == 'blocking')

    def host_budget(self, weights: int, copies: int) -> int:
        """Include checkpoint construction/staging and workspace in host admission."""
        required = weights * copies + self.workspace_bytes
        if self.host_bytes is not None and self.host_bytes < required:
            raise ValueError(f'KADAN_IMAGE_HOST_BYTES must cover the {required}-byte checkpoint/staging estimate')
        return required if self.host_bytes is None else self.host_bytes

    def affinity(self, allowed: set[int]) -> list[int]:
        """Never widen the process/cgroup CPU mask or silently clamp a request."""
        selected = allowed if self.cpus is None else set(self.cpus)
        if not selected or not selected <= allowed:
            raise ValueError('KADAN_IMAGE_CPUS must be within the current process CPU affinity')
        return sorted(selected)
