"""Native T3-style coding Work harness contract, adapters, folder browser, and runtime."""

from .contract import (
    HarnessAdapter,
    HarnessAuthStatus,
    HarnessAvailability,
    HarnessError,
    HarnessModelInfo,
    HarnessSession,
    NormalizedEvent,
    SessionConfig,
)
from .inkling_adapter import InklingHarnessAdapter, InklingHarnessSession
from .registry import HarnessRegistry, default_registry

__all__ = [
    "HarnessAdapter",
    "HarnessAuthStatus",
    "HarnessAvailability",
    "HarnessError",
    "HarnessModelInfo",
    "HarnessRegistry",
    "HarnessSession",
    "InklingHarnessAdapter",
    "InklingHarnessSession",
    "NormalizedEvent",
    "SessionConfig",
    "default_registry",
]
