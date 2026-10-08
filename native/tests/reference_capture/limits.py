"""Explicit test-reference ledger limits; independent of container RSS limits."""
from dataclasses import dataclass
from .artifacts import require


@dataclass(frozen=True)
class Limits:
    host_bytes: int = 128 * 1024**2
    request_bytes: int = 4 * 1024**2
    owner: str = 'synthetic-reference'

    def __post_init__(self):
        require(type(self.host_bytes) is int and type(self.request_bytes) is int
                and 0 < self.request_bytes < self.host_bytes <= 40 * 1024**3,
                'reference_limits')
        require(self.request_bytes <= 512 * 1024**2, 'request_limit')
        require(self.owner in ('synthetic-reference', 'actual-reference'), 'reference_owner')


ACTUAL_LIMITS = Limits(40 * 1024**3, 512 * 1024**2, 'actual-reference')
