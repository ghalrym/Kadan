"""Extensible registry of coding harness adapters for Kadan native Work.

Bundled adapters: Inkling (Local), Codex, Claude Code, Google Antigravity, and xAI Grok.
Additional harnesses register through `register(...)`.
"""

from __future__ import annotations

from ..api.config import ServiceError
from .antigravity_adapter import AntigravityHarnessAdapter
from .claude_adapter import ClaudeHarnessAdapter
from .codex_adapter import CodexHarnessAdapter
from .contract import HarnessAdapter
from .grok_adapter import GrokHarnessAdapter
from .inkling_adapter import InklingHarnessAdapter


class HarnessRegistry:
    """Ordered registry of available `HarnessAdapter` implementations."""

    def __init__(self, adapters: list[HarnessAdapter] | None = None):
        self._adapters: dict[str, HarnessAdapter] = {}
        for adapter in adapters or []:
            self.register(adapter)

    def register(self, adapter: HarnessAdapter) -> None:
        self._adapters[adapter.harness_id] = adapter

    def get(self, harness_id: str) -> HarnessAdapter:
        adapter = self._adapters.get(harness_id)
        if adapter is None:
            raise ServiceError(
                f"Unknown coding harness '{harness_id}'.",
                404,
                "unknown_harness",
            )
        return adapter

    def list(self) -> list[HarnessAdapter]:
        return list(self._adapters.values())


def default_registry() -> HarnessRegistry:
    """Create the default registry with Inkling (Local) and all bundled coding harnesses."""
    return HarnessRegistry(
        [
            InklingHarnessAdapter(),
            CodexHarnessAdapter(),
            ClaudeHarnessAdapter(),
            AntigravityHarnessAdapter(),
            GrokHarnessAdapter(),
        ]
    )
