"""Provider-agnostic coding harness adapter contract for Kadan native Work.

This module defines the stable interface implemented by all bundled coding harnesses
(Codex, Claude Code, Google Antigravity, xAI Grok) and future harnesses such as
Inkling 2 Mini (#530).

To add a new harness:
1. Subclass `HarnessSession` to drive one live turn/session in a working directory.
2. Subclass `HarnessAdapter` to report availability, UI sign-in/auth, model discovery,
   and open/resume `HarnessSession` instances.
3. Register the adapter with `HarnessRegistry.register(...)`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

if TYPE_CHECKING:
    from ..api.services import LocalServices


class HarnessError(Exception):
    """Plain-language failure raised by a coding harness adapter or session."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


AuthState = Literal[
    "ready",
    "sign_in_required",
    "starting",
    "waiting",
    "verifying",
    "unavailable",
    "error",
]
LoginMethod = Literal[
    "device_code", "oauth_code", "api_key", "browser", "local", "none"
]
EventKind = Literal[
    "session_started",
    "text_delta",
    "assistant_message",
    "reasoning_delta",
    "tool_call",
    "tool_result",
    "file_change",
    "pending_question",
    "pending_approval",
    "status",
    "error",
    "turn_completed",
]


@dataclass(frozen=True)
class HarnessModelInfo:
    """One selectable model offered by a harness."""

    id: str
    label: str
    description: str = ""
    options: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class HarnessAuthStatus:
    """Current sign-in and credential state surfaced to Kadan's Harnesses UI."""

    status: AuthState
    login_method: LoginMethod = "none"
    authorize_url: str | None = None
    user_code: str | None = None
    needs_code: bool = False
    account_label: str | None = None
    message: str = ""


@dataclass
class HarnessAvailability:
    """Discovery snapshot for one coding harness."""

    harness_id: str
    display_name: str
    description: str
    installed: bool
    binary_path: str | None
    auth: HarnessAuthStatus
    models: list[HarnessModelInfo] = field(default_factory=list)
    default_model: str | None = None
    supports_steer: bool = False
    supports_resume: bool = True
    supports_approvals: bool = True
    supports_questions: bool = True


@dataclass(frozen=True)
class SessionConfig:
    """Inputs used to open or resume a coding harness session."""

    thread_id: UUID
    working_dir: str
    model: str | None = None
    model_options: dict[str, Any] = field(default_factory=dict)
    harness_session_id: str | None = None
    permission_mode: Literal["ask", "auto_edit", "full_auto"] = "ask"
    env_overrides: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedEvent:
    """Uniform event emitted by every coding harness adapter."""

    kind: EventKind
    text: str = ""
    call_id: str | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    is_error: bool = False
    file_path: str | None = None
    change_type: str | None = None
    request_id: str | None = None
    title: str = ""
    options: list[dict[str, Any]] = field(default_factory=list)
    session_id: str | None = None
    terminal_status: (
        Literal["completed", "failed", "interrupted", "cancelled"] | None
    ) = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class HarnessSession(ABC):
    """A live coding session bound to a project root or isolated Git worktree."""

    @property
    @abstractmethod
    def session_id(self) -> str | None:
        """Durable harness session/thread identifier for later resume, if known."""

    @abstractmethod
    def start_turn(self, prompt: str) -> AsyncIterator[NormalizedEvent]:
        """Run one user prompt turn and yield normalized activity & terminal events."""

    async def steer(self, prompt: str) -> bool:
        """Send a mid-turn steering update if the harness supports live steering.

        Returns True when accepted inline, or False so the runtime queues a follow-up turn.
        """
        return False

    @abstractmethod
    async def interrupt(self) -> None:
        """Request immediate cancellation of the active turn."""

    @abstractmethod
    async def respond_pending(self, request_id: str, response: dict[str, Any]) -> None:
        """Deliver a user's answer or permission decision back to the harness."""

    @abstractmethod
    async def close(self) -> None:
        """Terminate child processes and clean up temporary resources."""


class HarnessAdapter(ABC):
    """Provider-agnostic interface for one coding harness."""

    harness_id: str
    display_name: str
    description: str

    @abstractmethod
    async def check_availability(self, services: LocalServices) -> HarnessAvailability:
        """Inspect binary installation, auth state, and selectable models."""

    @abstractmethod
    async def start_sign_in(
        self, services: LocalServices, payload: dict[str, Any] | None = None
    ) -> HarnessAuthStatus:
        """Begin a UI-driven sign-in or credential setup flow."""

    async def sign_in_status(self, services: LocalServices) -> HarnessAuthStatus:
        """Check in-progress sign-in status (defaults to current availability auth)."""
        return (await self.check_availability(services)).auth

    async def submit_sign_in(
        self, services: LocalServices, payload: dict[str, Any]
    ) -> HarnessAuthStatus:
        """Submit a verification code or credential during UI-driven sign-in."""
        raise HarnessError(
            f"{self.display_name} does not accept a sign-in code in this step."
        )

    @abstractmethod
    async def sign_out(self, services: LocalServices) -> HarnessAuthStatus:
        """Clear saved Kadan credentials for this harness."""

    @abstractmethod
    async def open_session(
        self, config: SessionConfig, services: LocalServices
    ) -> HarnessSession:
        """Start or resume a live coding session in `config.working_dir`."""
