"""Native single-pass Decisions API guardrails for the Inkling 2 Mini coding harness.

Uses Hikari's zero-decode `POST /v1/decisions` endpoint with `weight_variant="base"`
so guardrails share the resident Inkling 2 Mini engine without evicting the live
coding KV cache or switching away from base weights:
1. Command safety / approval gating (`classify_command_approval`)
2. Loop / stuck detection (`evaluate_loop_stuck`)
3. Verification / completion gating (`evaluate_completion_verification`)
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from ..runner.agent_inference import resolve_task_weight_variant

logger = logging.getLogger(__name__)
DECISIONS_TIMEOUT = httpx.Timeout(20.0, connect=3.0)
HARNESS_WEIGHT_VARIANT = resolve_task_weight_variant(task_class="harness")

# Commands that always require explicit user approval regardless of session setting.
_HARD_APPROVAL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\brm\s+-[a-zA-Z]*[rf][a-zA-Z]*\b"),
        "recursive or forced file deletion",
    ),
    (
        re.compile(r"\bgit\s+push\b.*(--force|-f\b|--force-with-lease)"),
        "force pushing Git history",
    ),
    (re.compile(r"\bgit\s+reset\s+--hard\b"), "hard-resetting Git working tree"),
    (
        re.compile(r"\bgit\s+clean\s+-[a-zA-Z]*[fd]"),
        "deleting untracked files with git clean",
    ),
    (re.compile(r"\bsudo\b|\bsu\s+"), "running a privileged command"),
    (
        re.compile(r"\b(mkfs|dd\s+if=|shutdown|reboot|halt|poweroff)\b"),
        "destructive system command",
    ),
    (
        re.compile(r"\b(curl|wget)\b[^|]*\|\s*(sh|bash|zsh)\b"),
        "piping remote script to a shell",
    ),
    (
        re.compile(r"\bchmod\s+-R\b|\bchown\s+-R\b"),
        "recursive permission or ownership change",
    ),
)

# Read-only or standard local verification prefixes safe without shell chaining/redirection.
_SHELL_META_RE = re.compile(r"[|;&`><]|\$\(|\n")
_SAFE_READONLY_PREFIXES: tuple[str, ...] = (
    "pytest",
    "python -m pytest",
    ".venv/bin/pytest",
    ".venv/bin/python -m pytest",
    "ruff check",
    "ruff format --check",
    ".venv/bin/ruff check",
    ".venv/bin/ruff format --check",
    "npm test",
    "npm run lint",
    "npm run build",
    "node --test",
    "git status",
    "git diff",
    "git log",
    "git branch",
    "ls",
    "pwd",
    "rg ",
    "grep ",
)


async def evaluate_base_decisions(
    client: httpx.AsyncClient,
    hikari_url: str,
    state: str,
    questions: Sequence[dict[str, Any]],
    *,
    weight_variant: str | None = HARNESS_WEIGHT_VARIANT,
) -> dict[str, float]:
    """Call Hikari `POST /v1/decisions` using the active Inkling weight variant."""
    cleaned_state = state.strip()
    if not cleaned_state or not questions:
        return {}
    payload_questions = [dict(q) for q in questions]
    trimmed_state = cleaned_state[:7500]
    body: dict[str, Any] = {
        "state": trimmed_state,
        "questions": payload_questions,
        "weight_variant": weight_variant or HARNESS_WEIGHT_VARIANT,
    }
    serialized_len = len(json.dumps(body, ensure_ascii=False))
    if serialized_len > 15_800:
        overflow = serialized_len - 15_800
        trimmed_state = trimmed_state[
            : max(100, len(trimmed_state) - overflow - 64)
        ].rstrip()
        body["state"] = trimmed_state

    response = await client.post(
        hikari_url.rstrip("/") + "/v1/decisions",
        json=body,
        timeout=DECISIONS_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    scores: dict[str, float] = {}
    for answer in data.get("answers", []):
        key = str(answer["key"])
        val = float(answer["value"])
        scores[key] = max(0.0, min(1.0, val))
    return scores


def _matches_hard_approval(command: str) -> str | None:
    for pattern, label in _HARD_APPROVAL_PATTERNS:
        if pattern.search(command):
            return label
    return None


def _is_simple_safe_command(command: str) -> bool:
    stripped = command.strip()
    if not stripped or _SHELL_META_RE.search(stripped):
        return False
    return any(
        stripped == prefix.strip() or stripped.startswith(prefix)
        for prefix in _SAFE_READONLY_PREFIXES
    )


@dataclass(frozen=True)
class CommandApprovalDecision:
    requires_approval: bool
    reason: str
    source: Literal["policy", "decisions", "fallback"]
    probability: float | None = None


async def classify_command_approval(
    command: str,
    *,
    cwd: str,
    permission_mode: Literal["ask", "auto_edit", "full_auto"],
    session_approved: bool,
    client: httpx.AsyncClient,
    hikari_url: str,
    weight_variant: str | None = HARNESS_WEIGHT_VARIANT,
) -> CommandApprovalDecision:
    """Decide whether a proposed shell command needs explicit user approval."""
    hard_reason = _matches_hard_approval(command)
    if hard_reason is not None:
        return CommandApprovalDecision(
            requires_approval=True,
            reason=f"Requires approval: {hard_reason}.",
            source="policy",
            probability=1.0,
        )
    if session_approved or permission_mode == "full_auto":
        return CommandApprovalDecision(
            requires_approval=False,
            reason="Allowed by session approval setting.",
            source="policy",
            probability=0.0,
        )
    if permission_mode == "auto_edit" and _is_simple_safe_command(command):
        return CommandApprovalDecision(
            requires_approval=False,
            reason="Standard local inspection or verification command.",
            source="policy",
            probability=0.0,
        )

    state = (
        f"Proposed workspace shell command:\n{command[:2000]}\n\n"
        f"Working directory: {cwd}\n"
        f"Permission mode: {permission_mode}"
    )
    questions = [
        {
            "key": "requires_approval",
            "type": "Noul",
            "instructions": (
                "Does this shell command have risky side effects that require explicit "
                "user approval before executing?"
            ),
            "trueWhen": (
                "The command deletes or overwrites files, pushes or rewrites Git history, "
                "installs packages, accesses the network, modifies environment/system state, "
                "or runs an unfamiliar script with unclear side effects."
            ),
            "falseWhen": (
                "The command is a purely local read-only inspection, linter, type-check, "
                "build, or unit test run with no destructive or external side effects."
            ),
        }
    ]
    try:
        scores = await evaluate_base_decisions(
            client, hikari_url, state, questions, weight_variant=weight_variant
        )
        prob = scores.get("requires_approval")
        if prob is not None:
            needs = prob >= 0.5
            return CommandApprovalDecision(
                requires_approval=needs,
                reason=(
                    "Flagged for confirmation before running."
                    if needs
                    else "Classified as low-risk local command."
                ),
                source="decisions",
                probability=prob,
            )
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        logger.debug("Command approval decision fallback: %s", exc)

    # Fail-safe fallback when Decisions is unavailable.
    if _is_simple_safe_command(command):
        return CommandApprovalDecision(
            requires_approval=False,
            reason="Standard local verification command.",
            source="fallback",
        )
    return CommandApprovalDecision(
        requires_approval=True,
        reason="Confirm before running this command.",
        source="fallback",
    )


@dataclass(frozen=True)
class StepRecord:
    tool_name: str
    signature: str
    is_error: bool
    summary: str


@dataclass(frozen=True)
class LoopGuardrailDecision:
    action: Literal["continue", "nudge", "ask_user"]
    message: str = ""
    is_stuck_score: float | None = None
    should_ask_score: float | None = None


def detect_repetition_trigger(recent_steps: Sequence[StepRecord]) -> str | None:
    """Fast local pre-filter so Decisions is only called when a loop pattern emerges."""
    if len(recent_steps) < 3:
        return None
    tail3 = recent_steps[-3:]
    if all(step.is_error for step in tail3):
        return "Three consecutive tool steps failed."
    window = recent_steps[-6:]
    sig_counts: dict[str, int] = {}
    for step in window:
        sig_counts[step.signature] = sig_counts.get(step.signature, 0) + 1
    most_common = max(sig_counts.values())
    if most_common >= 3:
        return "The same tool call pattern repeated 3+ times in recent steps."
    return None


async def evaluate_loop_stuck(
    recent_steps: Sequence[StepRecord],
    *,
    client: httpx.AsyncClient,
    hikari_url: str,
    weight_variant: str | None = HARNESS_WEIGHT_VARIANT,
) -> LoopGuardrailDecision:
    """Detect stuck or repetitive failing cycles using a single-pass Decisions check."""
    trigger = detect_repetition_trigger(recent_steps)
    if trigger is None:
        return LoopGuardrailDecision(action="continue")

    history_lines = [
        f"{idx + 1}. {step.tool_name} [{'ERROR' if step.is_error else 'OK'}]: {step.summary[:240]}"
        for idx, step in enumerate(recent_steps[-8:])
    ]
    state = f"Trigger: {trigger}\n\nRecent coding steps:\n" + "\n".join(history_lines)
    questions = [
        {
            "key": "is_stuck",
            "type": "Noul",
            "instructions": (
                "Is the coding assistant stuck in an unproductive or failing loop?"
            ),
            "trueWhen": (
                "It keeps repeating the same failing edit, invalid arguments, or unchanged "
                "command without making meaningful progress."
            ),
            "falseWhen": (
                "Each step is making deliberate progress on different files or resolving "
                "distinct errors."
            ),
        },
        {
            "key": "should_ask_user",
            "type": "Noul",
            "instructions": (
                "Does getting unstuck require asking the user for clarification or a decision?"
            ),
            "trueWhen": (
                "Requirements are ambiguous, a missing credential/environment detail is needed, "
                "or repeated attempts cannot resolve the blocker autonomously."
            ),
            "falseWhen": (
                "The assistant can recover by reading the actual file, inspecting the error, "
                "or changing its technical approach."
            ),
        },
    ]
    try:
        scores = await evaluate_base_decisions(
            client, hikari_url, state, questions, weight_variant=weight_variant
        )
        stuck = scores.get("is_stuck", 0.0)
        ask = scores.get("should_ask_user", 0.0)
        if stuck >= 0.65 and ask >= 0.70:
            return LoopGuardrailDecision(
                action="ask_user",
                message=(
                    "I hit repeated issues trying the same steps and need your guidance "
                    f"to continue ({trigger}). How would you like me to proceed?"
                ),
                is_stuck_score=stuck,
                should_ask_score=ask,
            )
        if stuck >= 0.60:
            return LoopGuardrailDecision(
                action="nudge",
                message=(
                    f"Recovery check: {trigger} Stop repeating the same call. "
                    "Re-read the relevant file lines or full error message, change strategy, "
                    "or call ask_user if you are blocked on a decision."
                ),
                is_stuck_score=stuck,
                should_ask_score=ask,
            )
        return LoopGuardrailDecision(
            action="continue", is_stuck_score=stuck, should_ask_score=ask
        )
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        logger.debug("Loop detection decision fallback: %s", exc)
        # Deterministic fallback when Decisions endpoint is unreachable.
        consecutive_errors = 0
        for step in reversed(recent_steps):
            if step.is_error:
                consecutive_errors += 1
            else:
                break
        if consecutive_errors >= 5:
            return LoopGuardrailDecision(
                action="ask_user",
                message=(
                    "Several consecutive steps failed and I need your input on how to proceed."
                ),
            )
        return LoopGuardrailDecision(
            action="nudge",
            message=(
                f"Recovery check: {trigger} Pause, inspect the exact error or file contents, "
                "and try a different approach."
            ),
        )


@dataclass(frozen=True)
class VerificationDecision:
    needs_verification: bool
    nudge_message: str = ""
    score: float | None = None


async def evaluate_completion_verification(
    modified_files: dict[str, str],
    draft_reply: str,
    verified_checks: Sequence[dict[str, Any]],
    *,
    client: httpx.AsyncClient,
    hikari_url: str,
    weight_variant: str | None = HARNESS_WEIGHT_VARIANT,
) -> VerificationDecision:
    """Check whether edited files need a verification pass before finishing the turn."""
    if not modified_files:
        return VerificationDecision(needs_verification=False)

    files_summary = ", ".join(
        f"{path} ({kind})" for path, kind in sorted(modified_files.items())[:20]
    )
    checks_summary = (
        "; ".join(f"{c['command']} -> {c['status']}" for c in verified_checks[-5:])
        if verified_checks
        else "None run after the latest edit."
    )
    state = (
        f"Modified files: {files_summary}\n"
        f"Checks run: {checks_summary}\n\n"
        f"Assistant's proposed final reply:\n{draft_reply[:2500]}"
    )
    questions = [
        {
            "key": "needs_verification",
            "type": "Noul",
            "instructions": (
                "Should the assistant run a quick verification check (such as unit tests, "
                "lint, or type-check) before closing out this turn?"
            ),
            "trueWhen": (
                "Code or configuration files were modified and no test, lint, or build "
                "check has been run since the last edit."
            ),
            "falseWhen": (
                "Only pure prose/documentation changed, or the user explicitly asked not "
                "to run checks, or checks already verified the final edits."
            ),
        }
    ]
    nudge = (
        "Verification check: you modified files without running a check after your last edit. "
        "Run the relevant targeted unit tests or lint check now using run_command, "
        "or briefly note why no check applies before concluding."
    )
    try:
        scores = await evaluate_base_decisions(
            client, hikari_url, state, questions, weight_variant=weight_variant
        )
        score = scores.get("needs_verification")
        if score is not None:
            return VerificationDecision(
                needs_verification=score >= 0.65,
                nudge_message=nudge if score >= 0.65 else "",
                score=score,
            )
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        logger.debug("Completion verification decision fallback: %s", exc)

    # Deterministic fallback: nudge when non-markdown source files were changed.
    code_changed = any(
        not path.lower().endswith((".md", ".txt")) for path in modified_files
    )
    return VerificationDecision(
        needs_verification=code_changed,
        nudge_message=nudge if code_changed else "",
    )
