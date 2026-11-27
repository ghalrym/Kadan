"""Explicit token budgeting and structured working-memory compaction for Inkling Work.

Tracks estimated token consumption against Hikari's active `HIKARI_LLM_CONTEXT_LIMIT`
before every step, never silently truncating or dropping messages. When a coding
thread approaches context headroom, compacts older turns into a structured
working-memory handoff (goal, modified files, verified checks, open blockers, next
steps) and resets cleanly onto the byte-identical immutable static prefix.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

DEFAULT_CONTEXT_LIMIT = 65_536
MIN_CONTEXT_LIMIT = 4_096
MAX_CONTEXT_LIMIT = 1_048_576

# Conservative code/JSON token estimate so we compact before hitting Hikari's hard limit.
CHARS_PER_TOKEN = 3.2
IMAGE_TOKEN_ESTIMATE = 768
MESSAGE_OVERHEAD_TOKENS = 12
DEFAULT_OUTPUT_RESERVE_TOKENS = 4_096
COMPACTION_TRIGGER_RATIO = 0.72
COMPACTION_MAX_SUMMARY_CHARS = 6_000
COMPACTION_RECENT_TAIL_GROUPS = 2

COMPACTION_HANDOFF_PREFIX = (
    "[Working-memory compaction checkpoint]\n"
    "Earlier steps in this coding thread were compacted to preserve context headroom. "
    "Continue from this verified state without repeating completed work:\n\n"
)

COMPACTION_SYSTEM_PROMPT = (
    "You summarize an in-progress coding session into a compact, factual working-memory "
    "handoff for Inkling 2 Mini. Return ONLY concise Markdown with these exact headings:\n"
    "## Goal\n"
    "## Modified files\n"
    "## Verified checks\n"
    "## Open blockers\n"
    "## Next steps\n"
    "Be concrete: cite exact file paths, line numbers, command outcomes, and unresolved "
    "errors. Omit chatter and keep the total under 3,500 characters."
)


def clamp_context_limit(context_limit: int | None) -> int:
    if context_limit is None or context_limit <= 0:
        return DEFAULT_CONTEXT_LIMIT
    return max(MIN_CONTEXT_LIMIT, min(MAX_CONTEXT_LIMIT, int(context_limit)))


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def estimate_message_tokens(message: dict[str, Any]) -> int:
    tokens = MESSAGE_OVERHEAD_TOKENS
    content = message.get("content")
    if isinstance(content, str):
        tokens += estimate_text_tokens(content)
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                tokens += estimate_text_tokens(str(part.get("text", "")))
            elif part_type == "image_url":
                tokens += IMAGE_TOKEN_ESTIMATE
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        tokens += estimate_text_tokens(
            json.dumps(tool_calls, separators=(",", ":"), ensure_ascii=False)
        )
    if message.get("tool_call_id"):
        tokens += 6
    return tokens


def estimate_messages_tokens(
    messages: Sequence[dict[str, Any]], *, tools_tokens: int = 0
) -> int:
    return tools_tokens + sum(estimate_message_tokens(msg) for msg in messages)


@dataclass(frozen=True)
class BudgetSnapshot:
    context_limit: int
    prompt_tokens: int
    tools_tokens: int
    reserved_output_tokens: int
    available_prompt_tokens: int
    headroom_tokens: int
    utilization: float
    needs_compaction: bool
    exceeds_limit: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_limit": self.context_limit,
            "prompt_tokens": self.prompt_tokens,
            "tools_tokens": self.tools_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "available_prompt_tokens": self.available_prompt_tokens,
            "headroom_tokens": self.headroom_tokens,
            "utilization": round(self.utilization, 3),
            "needs_compaction": self.needs_compaction,
            "exceeds_limit": self.exceeds_limit,
        }


def compute_budget_snapshot(
    messages: Sequence[dict[str, Any]],
    *,
    context_limit: int,
    tools_tokens: int,
    reserved_output_tokens: int = DEFAULT_OUTPUT_RESERVE_TOKENS,
    compaction_ratio: float = COMPACTION_TRIGGER_RATIO,
) -> BudgetSnapshot:
    limit = clamp_context_limit(context_limit)
    reserve = min(reserved_output_tokens, max(512, limit // 4))
    available = max(512, limit - reserve)
    prompt_tokens = estimate_messages_tokens(messages, tools_tokens=tools_tokens)
    headroom = available - prompt_tokens
    utilization = prompt_tokens / available
    return BudgetSnapshot(
        context_limit=limit,
        prompt_tokens=prompt_tokens,
        tools_tokens=tools_tokens,
        reserved_output_tokens=reserve,
        available_prompt_tokens=available,
        headroom_tokens=headroom,
        utilization=utilization,
        needs_compaction=utilization >= compaction_ratio,
        exceeds_limit=prompt_tokens > available,
    )


@dataclass
class WorkingMemoryTracker:
    """Structured coding-thread state preserved across explicit compactions."""

    goal: str = ""
    modified_files: dict[str, str] = field(default_factory=dict)
    verified_checks: list[dict[str, Any]] = field(default_factory=list)
    open_blockers: list[str] = field(default_factory=list)
    last_summary: str | None = None
    compaction_count: int = 0
    unverified_edits: bool = False

    def record_goal_if_empty(self, prompt: str) -> None:
        cleaned = prompt.strip()
        if cleaned and not self.goal:
            self.goal = cleaned[:1000]

    def record_file_change(self, path: str, change_type: str) -> None:
        self.modified_files[path] = change_type
        self.unverified_edits = True

    def record_check(self, command: str, exit_code: int | None) -> None:
        status = "passed" if exit_code == 0 else f"failed (exit {exit_code})"
        self.verified_checks.append({"command": command[:200], "status": status})
        self.verified_checks = self.verified_checks[-12:]
        if exit_code == 0:
            self.unverified_edits = False

    def record_blocker(self, note: str) -> None:
        cleaned = note.strip()[:300]
        if cleaned and cleaned not in self.open_blockers:
            self.open_blockers.append(cleaned)
            self.open_blockers = self.open_blockers[-8:]

    def clear_blockers(self) -> None:
        self.open_blockers.clear()

    def format_deterministic_summary(
        self, dropped_messages: Sequence[dict[str, Any]]
    ) -> str:
        """Build a guaranteed local working-memory handoff without needing an LLM call."""
        lines: list[str] = [
            "## Goal",
            self.goal or "Continue the current coding task.",
            "",
            "## Modified files",
        ]
        if self.modified_files:
            for path, kind in sorted(self.modified_files.items()):
                lines.append(f"- `{path}` ({kind})")
        else:
            lines.append("- None recorded yet.")

        lines.extend(["", "## Verified checks"])
        if self.verified_checks:
            for item in self.verified_checks[-6:]:
                lines.append(f"- `{item['command']}`: {item['status']}")
        else:
            lines.append("- None run yet.")

        lines.extend(["", "## Open blockers"])
        if self.open_blockers:
            for blocker in self.open_blockers[-5:]:
                lines.append(f"- {blocker}")
        else:
            lines.append("- None noted.")

        # Extract brief highlights from the dropped messages for continuity.
        recent_notes: list[str] = []
        for msg in dropped_messages[-6:]:
            role = msg.get("role")
            content = _plain_message_text(msg)
            if role in {"user", "assistant"} and content:
                recent_notes.append(f"- ({role}) {content[:220]}")
        lines.extend(["", "## Next steps"])
        if self.unverified_edits:
            lines.append("- Verify modified files with relevant tests/lint checks.")
        if recent_notes:
            lines.append("- Recent compacted context:")
            lines.extend(recent_notes[-4:])
        else:
            lines.append("- Continue directly from the latest user instruction.")
        summary = "\n".join(lines).strip()
        return summary[:COMPACTION_MAX_SUMMARY_CHARS]

    def build_compaction_messages(
        self, dropped_messages: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Build a stateless prompt for Inkling 2 Mini to synthesize structured working memory."""
        transcript_lines: list[str] = []
        for msg in dropped_messages:
            role = str(msg.get("role", "unknown"))
            text = _plain_message_text(msg)
            if msg.get("tool_calls"):
                names = [
                    c.get("function", {}).get("name", "tool")
                    for c in msg["tool_calls"]
                    if isinstance(c, dict)
                ]
                call_note = f"[tool_calls: {', '.join(names)}]"
                text = f"{text} {call_note}".strip()
            if text:
                transcript_lines.append(f"{role.upper()}: {text[:800]}")
        tracked = self.format_deterministic_summary([])
        parts = [
            "Tracked factual state:\n" + tracked,
        ]
        if self.last_summary:
            parts.append("Previous compacted summary:\n" + self.last_summary[:2000])
        parts.append(
            "Older turns to fold into working memory:\n"
            + "\n".join(transcript_lines)[-16_000:]
        )
        return [
            {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)},
        ]


def _plain_message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    pieces.append(str(part.get("text", "")))
                elif part.get("type") == "image_url":
                    pieces.append("[attached image]")
        return " ".join(p for p in pieces if p).strip()
    return ""


def _group_atomic_messages(
    history: Sequence[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group messages so assistant tool_calls and their tool results are never split."""
    groups: list[list[dict[str, Any]]] = []
    idx = 0
    while idx < len(history):
        msg = history[idx]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            group = [msg]
            idx += 1
            while idx < len(history) and history[idx].get("role") == "tool":
                group.append(history[idx])
                idx += 1
            groups.append(group)
        else:
            groups.append([msg])
            idx += 1
    return groups


def partition_messages_for_compaction(
    messages: Sequence[dict[str, Any]],
    *,
    keep_tail_groups: int = COMPACTION_RECENT_TAIL_GROUPS,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split `[system, ...]` into `(system_msg, to_compact, recent_tail)`.

    `system_msg` is always preserved verbatim at index 0. Tool-call groups are kept
    atomic so Hikari never receives an orphaned `tool` result message.
    """
    if not messages or messages[0].get("role") != "system":
        raise ValueError("Conversation must begin with the immutable system message.")
    system_msg = messages[0]
    groups = _group_atomic_messages(messages[1:])
    if len(groups) <= keep_tail_groups:
        return system_msg, [], list(messages[1:])
    compact_groups = groups[:-keep_tail_groups] if keep_tail_groups > 0 else groups
    tail_groups = groups[-keep_tail_groups:] if keep_tail_groups > 0 else []
    to_compact = [msg for group in compact_groups for msg in group]
    recent_tail = [msg for group in tail_groups for msg in group]
    return system_msg, to_compact, recent_tail


def rebuild_compacted_messages(
    system_msg: dict[str, Any],
    summary_text: str,
    recent_tail: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rebuild conversation onto the exact immutable static system prompt."""
    cleaned = summary_text.strip()[:COMPACTION_MAX_SUMMARY_CHARS]
    handoff_msg = {
        "role": "user",
        "content": COMPACTION_HANDOFF_PREFIX + cleaned,
    }
    return [system_msg, handoff_msg, *recent_tail]
