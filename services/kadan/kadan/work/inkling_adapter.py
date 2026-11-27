"""First-party local coding harness adapter optimized for Inkling 2 Mini (#530).

Connects directly to Hikari's native C++ Strata MoE engine (`hikari-worker-llm`) via
`POST /v1/chat/completions` and `POST /v1/decisions` with:
- Explicit base Inkling 2 Mini weight selection (`weight_variant="base"` via #528).
- Priority-aware scheduling (`background` for autonomous/agent-launched threads,
  `interactive` when user-driven).
- Strictly byte-identical static system prompt + tool schema prefix across every step
  and thread so `hikari-worker-llm`'s GPU boundary snapshots (`checkpoint_hit` /
  `static_prefix_hit`) stay warm.
- Explicit token budgeting against `HIKARI_LLM_CONTEXT_LIMIT` and structured
  working-memory compaction without silent truncation.
- Single-pass Decisions guardrails for command approval, loop recovery, and
  verification gating.
- Native multimodal image support via Inkling 2 Mini's built-in `mmproj` encoder.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

import httpx
from pydantic import ValidationError

from ..api.config import ServiceError
from ..api.pydantic_models.inference import Checkpoint, HikariModels
from ..runner.agent_inference import (
    INFERENCE_TIMEOUT,
    parse_call,
    resolve_task_weight_variant,
    stream_data,
    unique_fields,
)
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
from .inkling_budget import (
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_OUTPUT_RESERVE_TOKENS,
    BudgetSnapshot,
    WorkingMemoryTracker,
    clamp_context_limit,
    compute_budget_snapshot,
    estimate_text_tokens,
    partition_messages_for_compaction,
    rebuild_compacted_messages,
)
from .inkling_decisions import (
    StepRecord,
    classify_command_approval,
    evaluate_completion_verification,
    evaluate_loop_stuck,
)
from .inkling_tools import (
    INKLING_TOOL_SCHEMAS,
    AskUserArgs,
    InklingToolset,
    RunCommandArgs,
    canonical_tool_schemas_json,
    validate_tool_arguments,
)

if TYPE_CHECKING:
    from ..api.services import LocalServices

logger = logging.getLogger(__name__)

INKLING_HARNESS_ID = "inkling"
INKLING_BASE_MODELS: tuple[str, ...] = (
    "inkling-2-mini-iq4-nl",
    "inkling-2-mini-q8-0",
)
INKLING_ABLITERATED_MODELS: tuple[str, ...] = (
    "inkling-2-mini-abliterated-iq4-nl",
    "inkling-2-mini-abliterated-q8-0",
)
INKLING_DEFAULT_MODEL_ID = "inkling-2-mini-iq4-nl"
MAX_STEPS_PER_TURN = 32
MAX_CONSECUTIVE_VALIDATION_ERRORS = 4
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_REASONING_TOKENS = 2048

# Strictly static, byte-immutable system prompt shared across every step and thread.
# Never interpolate timestamps, working directories, git state, or dynamic file trees here.
STATIC_SYSTEM_PROMPT = (
    "You are Inkling, Kadan's local coding assistant working directly inside a project repository.\n"
    "Rules:\n"
    "1. Inspect before editing: use find_files, search_text, read_file, and git_inspect to understand the code.\n"
    "2. Prefer targeted edit_file replacements over rewriting entire files; use write_file only for new files or full rewrites.\n"
    "3. Keep commands focused and bounded. Never run destructive or external commands without need.\n"
    "4. Verify code changes with relevant targeted unit tests or lint checks using run_command before concluding.\n"
    "5. If requirements are ambiguous or you are blocked on a user decision, call ask_user with clear plain-language options.\n"
    "6. Never fabricate file contents, tool outputs, or test results. Report concisely what changed and what was verified."
)

_STATIC_TOOLS_JSON = canonical_tool_schemas_json()
_STATIC_TOOLS_TOKENS = estimate_text_tokens(_STATIC_TOOLS_JSON)


def compute_static_prefix_digest(
    system_prompt: str = STATIC_SYSTEM_PROMPT,
    tools_json: str = _STATIC_TOOLS_JSON,
) -> str:
    """Return the SHA-256 digest of the immutable system prompt + tool schema prefix."""
    hasher = hashlib.sha256()
    hasher.update(system_prompt.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(tools_json.encode("utf-8"))
    return hasher.hexdigest()


STATIC_PREFIX_DIGEST = compute_static_prefix_digest()


def _companion_base_model_id(selected_model_id: str | None) -> str:
    """Map any Inkling 2 Mini checkpoint ID to its same-precision base checkpoint ID."""
    if selected_model_id in ("inkling-2-mini-q8-0", "inkling-2-mini-abliterated-q8-0"):
        return "inkling-2-mini-q8-0"
    return "inkling-2-mini-iq4-nl"


def _select_base_checkpoint(catalog: HikariModels) -> Checkpoint | None:
    preferred_id = _companion_base_model_id(catalog.selected_model_id)
    by_id = {item.id: item for item in catalog.models}
    if preferred_id in by_id:
        return by_id[preferred_id]
    for candidate_id in INKLING_BASE_MODELS:
        if candidate_id in by_id:
            return by_id[candidate_id]
    return None


def _select_inkling_checkpoint(catalog: HikariModels) -> Checkpoint | None:
    """Select the ready Inkling 2 Mini checkpoint on Kadan Server.

    Prefers a downloaded base checkpoint when present so coding runs use base weights,
    and otherwise falls back to Kadan Server's active Inkling 2 Mini LLM checkpoint so
    Inkling works out of the box against whatever variant the server is serving.
    """
    base_ckpt = _select_base_checkpoint(catalog)
    if (
        base_ckpt is not None
        and base_ckpt.inference_available
        and base_ckpt.status == "complete"
    ):
        return base_ckpt
    if catalog.selected_model_id:
        selected = next(
            (item for item in catalog.models if item.id == catalog.selected_model_id),
            None,
        )
        if selected is not None and (
            selected.kind == "llm" or selected.id.startswith("inkling-2-mini-")
        ):
            if selected.inference_available and selected.status == "complete":
                return selected
            if base_ckpt is None:
                return selected
    llm_ckpt = next(
        (
            item
            for item in catalog.models
            if item.kind == "llm" or item.id.startswith("inkling-2-mini-")
        ),
        None,
    )
    return base_ckpt or llm_ckpt


def _checkpoint_weight_variant(model_id: str | None) -> Literal["base", "abliterated"]:
    if model_id in INKLING_ABLITERATED_MODELS:
        return "abliterated"
    return resolve_task_weight_variant(task_class="harness") or "base"


def _model_label(model_id: str) -> str:
    if model_id in ("inkling-2-mini-q8-0", "inkling-2-mini-abliterated-q8-0"):
        return "Inkling 2 Mini (Q8)"
    return "Inkling 2 Mini"


def _friendly_hikari_error(status_code: int, detail: str) -> str:
    """Translate Hikari HTTP/worker failures into plain-language actionable guidance."""
    lower = detail.lower()
    if (
        "not downloaded" in lower
        or "missing" in lower
        and "checkpoint" in lower
        or "weight_variant" in lower
        and "download" in lower
    ):
        return (
            "The standard Inkling 2 Mini model is not downloaded yet. "
            "Open Kadan Server, go to Hikari → Settings, and download Inkling 2 Mini to continue."
        )
    if "context" in lower and (
        "limit" in lower or "exceed" in lower or "fit" in lower or "too long" in lower
    ):
        return (
            "This conversation is too large for Inkling 2 Mini's current memory window. "
            "Try a shorter message, fewer attached images, or start a new Work thread."
        )
    if (
        "vram" in lower
        or "memory" in lower
        or "admission" in lower
        or "out of memory" in lower
        or status_code == 507
    ):
        return (
            "Not enough GPU or system memory is available to run Inkling 2 Mini right now. "
            "Wait for other local AI tasks to finish or lower the context window in Hikari Settings."
        )
    if status_code in (502, 503, 504):
        return (
            "Kadan's local AI service is temporarily unavailable. "
            "Check that Kadan Server is running and try again."
        )
    cleaned = detail.strip()
    if cleaned:
        return f"Inkling could not complete this step: {cleaned}"
    return f"Kadan's local AI service returned an error (HTTP {status_code})."


def _build_user_message(
    prompt: str, images: Sequence[str] | None = None
) -> dict[str, Any]:
    if not images:
        return {"role": "user", "content": prompt}
    parts: list[dict[str, Any]] = []
    if prompt:
        parts.append({"type": "text", "text": prompt})
    for data_url in images:
        parts.append({"type": "image_url", "image_url": {"url": data_url}})
    return {"role": "user", "content": parts}


@dataclass
class InklingSessionState:
    """Durable-in-process conversation state preserving exact prefix byte identity."""

    messages: list[dict[str, Any]] = field(
        default_factory=lambda: [{"role": "system", "content": STATIC_SYSTEM_PROMPT}]
    )
    static_prefix_digest: str = STATIC_PREFIX_DIGEST
    working_memory: WorkingMemoryTracker = field(default_factory=WorkingMemoryTracker)
    approved_for_session: bool = False
    recent_steps: list[StepRecord] = field(default_factory=list)
    hydrated: bool = False


class InklingHarnessSession(HarnessSession):
    """Live local coding session running Inkling 2 Mini directly on Hikari."""

    supports_images: bool = True

    def __init__(
        self,
        config: SessionConfig,
        services: LocalServices,
        *,
        state: InklingSessionState | None = None,
        context_limit: int | None = None,
        toolset: InklingToolset | None = None,
        max_steps: int = MAX_STEPS_PER_TURN,
    ):
        self.config = config
        self.services = services
        self._state = state if state is not None else InklingSessionState()
        self._context_limit_override = context_limit
        self._weight_variant: Literal["base", "abliterated"] = (
            _checkpoint_weight_variant(config.model)
        )
        self._toolset = toolset or InklingToolset(config.working_dir)
        self._max_steps = max_steps
        self._session_id = config.harness_session_id or f"inkling:{config.thread_id}"
        self._interrupted = asyncio.Event()
        self._steer_queue: list[str] = []
        self._pending_waiters: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._pending_responses: dict[str, dict[str, Any]] = {}

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def state(self) -> InklingSessionState:
        return self._state

    async def steer(self, prompt: str) -> bool:
        cleaned = prompt.strip()
        if not cleaned or self._interrupted.is_set():
            return False
        self._steer_queue.append(cleaned)
        return True

    async def interrupt(self) -> None:
        self._interrupted.set()
        for fut in list(self._pending_waiters.values()):
            if not fut.done():
                fut.cancel()
        await self._toolset.cancel_running()

    async def respond_pending(self, request_id: str, response: dict[str, Any]) -> None:
        payload = dict(response)
        fut = self._pending_waiters.get(request_id)
        if fut is not None and not fut.done():
            fut.set_result(payload)
        else:
            self._pending_responses[request_id] = payload

    async def close(self) -> None:
        await self._toolset.cancel_running()
        for fut in list(self._pending_waiters.values()):
            if not fut.done():
                fut.cancel()
        self._pending_waiters.clear()
        self._pending_responses.clear()

    async def _resolve_context_limit(self) -> int:
        try:
            raw = await self.services.json_request("Hikari", "GET", "/v1/models")
            catalog = HikariModels.model_validate(raw)
            ckpt = _select_inkling_checkpoint(catalog)
            if ckpt is not None:
                self._weight_variant = _checkpoint_weight_variant(ckpt.id)
                if self._context_limit_override is None:
                    return clamp_context_limit(
                        ckpt.context_limit or ckpt.architecture_context_limit
                    )
        except (ServiceError, ValidationError, httpx.HTTPError):
            logger.debug("Falling back to default Inkling context limit", exc_info=True)
        if self._context_limit_override is not None:
            return clamp_context_limit(self._context_limit_override)
        return DEFAULT_CONTEXT_LIMIT

    async def _hydrate_from_activity_if_needed(self) -> None:
        """Restore prior turns from Tsuchi activity after a Kadan Runner restart."""
        if self._state.hydrated or len(self._state.messages) > 1:
            self._state.hydrated = True
            return
        self._state.hydrated = True
        if not self.config.harness_session_id:
            return
        try:
            items = await self.services.work.list_activity(
                self.config.thread_id, limit=200
            )
        except (ServiceError, ValidationError, httpx.HTTPError):
            return
        # Reconstruct prior user/assistant exchanges (excluding the current run's user message
        # which WorkRunner appends right before calling start_turn).
        reconstructed: list[dict[str, Any]] = []
        for item in items:
            if item.kind == "user_message" and item.text.strip():
                reconstructed.append({"role": "user", "content": item.text})
            elif item.kind == "assistant_message" and item.text.strip():
                if reconstructed and reconstructed[-1]["role"] == "assistant":
                    reconstructed[-1]["content"] += item.text
                else:
                    reconstructed.append({"role": "assistant", "content": item.text})
            elif item.kind == "file_change" and item.data.get("file_path"):
                self._state.working_memory.record_file_change(
                    str(item.data["file_path"]),
                    str(item.data.get("change_type", "modified")),
                )
        # Drop the trailing user_message if WorkRunner already logged the incoming turn.
        if reconstructed and reconstructed[-1]["role"] == "user":
            reconstructed.pop()
        if reconstructed:
            self._state.messages.extend(reconstructed)
            first_user = next(
                (m["content"] for m in reconstructed if m["role"] == "user"), ""
            )
            if isinstance(first_user, str):
                self._state.working_memory.record_goal_if_empty(first_user)

    def _assert_prefix_invariant(self) -> None:
        if (
            not self._state.messages
            or self._state.messages[0].get("role") != "system"
            or compute_static_prefix_digest(
                str(self._state.messages[0].get("content", "")),
                _STATIC_TOOLS_JSON,
            )
            != self._state.static_prefix_digest
        ):
            raise HarnessError(
                "Internal error: Inkling static prefix invariant was violated."
            )

    def _reasoning_effort(self) -> Literal["low", "medium", "high"] | None:
        raw = self.config.model_options.get(
            "reasoning_effort"
        ) or self.config.model_options.get("effort")
        if raw in ("low", "medium", "high"):
            return raw
        return None

    async def _compact_history(self, context_limit: int) -> tuple[bool, dict[str, Any]]:
        """Explicitly compact older turns into structured working memory."""
        before_tokens = compute_budget_snapshot(
            self._state.messages,
            context_limit=context_limit,
            tools_tokens=_STATIC_TOOLS_TOKENS,
        ).prompt_tokens

        system_msg, to_compact, recent_tail = partition_messages_for_compaction(
            self._state.messages
        )
        if not to_compact:
            # Try compacting down to the last atomic group if even tail=2 doesn't free anything.
            system_msg, to_compact, recent_tail = partition_messages_for_compaction(
                self._state.messages, keep_tail_groups=1
            )
        if not to_compact:
            return False, {
                "before_tokens": before_tokens,
                "after_tokens": before_tokens,
            }

        summary: str | None = None
        try:
            summary = await self._run_compaction_completion(to_compact)
        except (HarnessError, ServiceError, httpx.HTTPError, ValueError) as exc:
            logger.info(
                "Inkling LLM compaction fell back to deterministic summary: %s", exc
            )
        if not summary:
            summary = self._state.working_memory.format_deterministic_summary(
                to_compact
            )

        rebuilt = rebuild_compacted_messages(system_msg, summary, recent_tail)
        after_tokens = compute_budget_snapshot(
            rebuilt,
            context_limit=context_limit,
            tools_tokens=_STATIC_TOOLS_TOKENS,
        ).prompt_tokens
        if after_tokens >= before_tokens:
            return False, {
                "before_tokens": before_tokens,
                "after_tokens": before_tokens,
            }

        self._state.working_memory.last_summary = summary
        self._state.working_memory.compaction_count += 1
        self._state.messages = rebuilt
        self._assert_prefix_invariant()

        return True, {
            "compaction_count": self._state.working_memory.compaction_count,
            "compacted_messages": len(to_compact),
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "static_prefix_preserved": True,
        }

    async def _run_compaction_completion(
        self, dropped_messages: Sequence[dict[str, Any]]
    ) -> str:
        weight_variant = self._weight_variant
        payload: dict[str, Any] = {
            "messages": self._state.working_memory.build_compaction_messages(
                dropped_messages
            ),
            "stream": False,
            "weight_variant": weight_variant,
            "reuse_prefix": False,
            "priority": self.config.priority,
            "max_tokens": 1800,
            "reasoning_tokens": 256,
            "reasoning_effort": "low",
        }
        response = await self.services.client.post(
            self.services.settings.hikari_url.rstrip("/") + "/v1/chat/completions",
            json=payload,
            timeout=INFERENCE_TIMEOUT,
        )
        if not response.is_success:
            raise HarnessError(f"Compaction failed (HTTP {response.status_code}).")
        data = response.json()
        choices = data.get("choices") or []
        if choices:
            text = str(choices[0].get("message", {}).get("content", "")).strip()
            if text:
                return text
        return ""

    async def _stream_hikari_step(
        self,
        *,
        budget: BudgetSnapshot,
    ) -> AsyncIterator[NormalizedEvent | dict[str, Any]]:
        """Stream one Inkling 2 Mini step from Hikari `/v1/chat/completions`.

        Yields `NormalizedEvent(kind="text_delta")` as text arrives, and finishes by
        yielding one internal dict `{"_step_result": ...}` with text, tool_calls, and
        cache/timing telemetry.
        """
        weight_variant = self._weight_variant
        max_tokens = min(
            DEFAULT_MAX_OUTPUT_TOKENS,
            max(256, budget.context_limit - budget.prompt_tokens),
        )
        reasoning_tokens = min(DEFAULT_REASONING_TOKENS, max(128, max_tokens // 2))
        payload: dict[str, Any] = {
            "messages": self._state.messages,
            "tools": list(INKLING_TOOL_SCHEMAS),
            "stream": True,
            "weight_variant": weight_variant,
            "conversation_id": self._session_id,
            "reuse_prefix": True,
            "priority": self.config.priority,
            "max_tokens": max_tokens,
            "reasoning_tokens": reasoning_tokens,
        }
        effort = self._reasoning_effort()
        if effort is not None:
            payload["reasoning_effort"] = effort

        started = time.monotonic()
        first_token_ms: float | None = None
        content_parts: list[str] = []
        raw_calls: list[dict[str, Any]] = []
        parsed_calls: list[dict[str, Any]] = []
        finish_reason: str | None = None
        cache_info: dict[str, Any] | None = None
        timing_info: dict[str, Any] = {}
        done = False

        try:
            async with self.services.client.stream(
                "POST",
                self.services.settings.hikari_url.rstrip("/") + "/v1/chat/completions",
                json=payload,
                timeout=INFERENCE_TIMEOUT,
            ) as response:
                if not response.is_success:
                    body_text = (await response.aread()).decode(
                        "utf-8", errors="replace"
                    )
                    detail = body_text
                    with suppress(ValueError):
                        parsed_err = json.loads(body_text)
                        if isinstance(parsed_err, dict) and isinstance(
                            parsed_err.get("detail"), str
                        ):
                            detail = parsed_err["detail"]
                    raise HarnessError(
                        _friendly_hikari_error(response.status_code, detail)
                    )

                content_type = response.headers.get("content-type", "")
                if not content_type.startswith("text/event-stream"):
                    raise HarnessError(
                        "Kadan's local AI service did not return a valid chat stream."
                    )

                async for data in stream_data(response.aiter_lines()):
                    if self._interrupted.is_set():
                        break
                    if data == "[DONE]":
                        done = True
                        break
                    try:
                        chunk = json.loads(data, object_pairs_hook=unique_fields)
                    except ValueError as exc:
                        raise HarnessError(
                            "Inkling returned malformed stream data."
                        ) from exc
                    if not isinstance(chunk, dict):
                        raise HarnessError("Inkling returned an invalid stream frame.")
                    if "error" in chunk:
                        err_obj = chunk["error"]
                        msg = (
                            err_obj.get("message", str(err_obj))
                            if isinstance(err_obj, dict)
                            else str(err_obj)
                        )
                        code = (
                            int(err_obj.get("code", 502))
                            if isinstance(err_obj, dict)
                            and isinstance(err_obj.get("code"), int)
                            else 502
                        )
                        raise HarnessError(_friendly_hikari_error(code, msg))

                    if isinstance(chunk.get("cache"), dict):
                        cache_info = dict(chunk["cache"])
                    if isinstance(chunk.get("timing"), dict):
                        timing_info.update(chunk["timing"])

                    choices = chunk.get("choices")
                    if not isinstance(choices, list) or len(choices) != 1:
                        raise HarnessError(
                            "Inkling returned an unexpected stream choice."
                        )
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    text = delta.get("content", "")
                    reason = choice.get("finish_reason")
                    if text:
                        if first_token_ms is None:
                            first_token_ms = (time.monotonic() - started) * 1000.0
                        content_parts.append(text)
                        if not raw_calls:
                            yield NormalizedEvent(kind="text_delta", text=text)
                    if delta.get("tool_calls"):
                        if first_token_ms is None:
                            first_token_ms = (time.monotonic() - started) * 1000.0
                        for item in delta["tool_calls"]:
                            raw_calls.append(item)
                            try:
                                parsed_calls.append(parse_call(item, len(parsed_calls)))
                            except ServiceError:
                                # Preserve malformed tool call so guided validation recovery handles it.
                                fn = (
                                    item.get("function")
                                    if isinstance(item, dict)
                                    else {}
                                )
                                parsed_calls.append(
                                    {
                                        "id": str(
                                            item.get("id")
                                            or f"call_{len(parsed_calls)}"
                                        ),
                                        "name": str(fn.get("name", ""))
                                        if isinstance(fn, dict)
                                        else "",
                                        "args": fn.get("arguments", "")
                                        if isinstance(fn, dict)
                                        else "",
                                        "_parse_error": True,
                                    }
                                )
                    if reason is not None:
                        finish_reason = str(reason)
        except httpx.RequestError as exc:
            raise HarnessError(
                "Lost connection to Kadan's local AI service. Check that Kadan Server is running.",
                retryable=True,
            ) from exc

        if not self._interrupted.is_set() and (not done or finish_reason is None):
            raise HarnessError(
                "Inkling stopped before finishing its response. You can retry the message.",
                retryable=True,
            )

        duration_ms = (time.monotonic() - started) * 1000.0
        telemetry: dict[str, Any] = {
            "weight_variant": weight_variant,
            "priority": self.config.priority,
            "static_prefix_sha256": self._state.static_prefix_digest[:12],
            "duration_ms": round(duration_ms, 1),
            "prompt_tokens": budget.prompt_tokens,
            "context_limit": budget.context_limit,
        }
        if first_token_ms is not None:
            telemetry["ttft_ms"] = round(first_token_ms, 1)
        if cache_info is not None:
            telemetry["cache_hit"] = bool(cache_info.get("hit", False))
            if "reason" in cache_info:
                telemetry["cache_reason"] = cache_info["reason"]
            if "reused_tokens" in cache_info:
                telemetry["reused_tokens"] = cache_info["reused_tokens"]
            if "stored_tokens" in cache_info:
                telemetry["stored_tokens"] = cache_info["stored_tokens"]
        for key in (
            "weight_swap_ms",
            "prefill_ms",
            "cache_restore_ms",
            "image_encode_ms",
            "decode_tokens_per_second",
            "prefill_tokens",
            "output_tokens",
        ):
            if key in timing_info and timing_info[key] is not None:
                telemetry[key] = timing_info[key]

        yield {
            "_step_result": {
                "text": "".join(content_parts),
                "tool_calls": parsed_calls,
                "finish_reason": finish_reason,
                "telemetry": telemetry,
            }
        }

    def _format_telemetry_status(self, step_num: int, telemetry: dict[str, Any]) -> str:
        bits = [f"Step {step_num} (base weights, {telemetry['priority']})"]
        if "cache_hit" in telemetry:
            if telemetry["cache_hit"]:
                reused = telemetry.get("reused_tokens", 0)
                reason = telemetry.get("cache_reason", "hit")
                bits.append(f"cache {reason} ({reused} tok reused)")
            else:
                bits.append(f"cache miss ({telemetry.get('cache_reason', 'cold')})")
        if telemetry.get("weight_swap_ms"):
            bits.append(f"swap {telemetry['weight_swap_ms']:.0f}ms")
        if telemetry.get("prefill_ms") is not None:
            bits.append(f"prefill {telemetry['prefill_ms']:.0f}ms")
        if telemetry.get("decode_tokens_per_second") is not None:
            bits.append(f"{telemetry['decode_tokens_per_second']:.1f} tok/s")
        bits.append(f"ctx {telemetry['prompt_tokens']}/{telemetry['context_limit']}")
        return " · ".join(bits)

    async def _wait_for_pending(self, request_id: str) -> dict[str, Any] | None:
        if request_id in self._pending_responses:
            return self._pending_responses.pop(request_id)
        if self._interrupted.is_set():
            return None
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_waiters[request_id] = fut
        try:
            return await fut
        except asyncio.CancelledError:
            return None
        finally:
            self._pending_waiters.pop(request_id, None)

    async def start_turn(
        self,
        prompt: str,
        *,
        images: Sequence[str] | None = None,
    ) -> AsyncIterator[NormalizedEvent]:
        yield NormalizedEvent(kind="session_started", session_id=self._session_id)

        await self._hydrate_from_activity_if_needed()
        context_limit = await self._resolve_context_limit()

        self._state.working_memory.record_goal_if_empty(prompt)
        self._state.messages.append(_build_user_message(prompt, images))

        verification_nudged = False
        consecutive_validation_errors = 0
        final_text = ""

        for step in range(1, self._max_steps + 1):
            if self._interrupted.is_set():
                yield NormalizedEvent(
                    kind="turn_completed",
                    text=final_text,
                    terminal_status="interrupted",
                )
                return

            # Drain any live steering updates strictly at the tail of history.
            if self._steer_queue:
                updates = list(self._steer_queue)
                self._steer_queue.clear()
                for update in updates:
                    self._state.messages.append(
                        {"role": "user", "content": f"[User update]: {update}"}
                    )

            self._assert_prefix_invariant()

            # Explicit token budgeting and structured working-memory compaction.
            budget = compute_budget_snapshot(
                self._state.messages,
                context_limit=context_limit,
                tools_tokens=_STATIC_TOOLS_TOKENS,
                reserved_output_tokens=DEFAULT_OUTPUT_RESERVE_TOKENS,
            )
            if budget.needs_compaction:
                compacted, comp_meta = await self._compact_history(context_limit)
                if compacted:
                    budget = compute_budget_snapshot(
                        self._state.messages,
                        context_limit=context_limit,
                        tools_tokens=_STATIC_TOOLS_TOKENS,
                        reserved_output_tokens=DEFAULT_OUTPUT_RESERVE_TOKENS,
                    )
                    yield NormalizedEvent(
                        kind="status",
                        text=(
                            f"Compacted working memory ({comp_meta['before_tokens']} → "
                            f"{comp_meta['after_tokens']} est. tokens; static prefix preserved)."
                        ),
                        raw={"compaction": comp_meta},
                    )
            if budget.exceeds_limit:
                raise HarnessError(
                    f"This request (~{budget.prompt_tokens:,} tokens) exceeds Inkling 2 Mini's "
                    f"active {budget.context_limit:,}-token memory limit even after compaction. "
                    "Shorten the message, remove large images, or start a new Work thread."
                )

            step_result: dict[str, Any] | None = None
            async for item in self._stream_hikari_step(budget=budget):
                if isinstance(item, NormalizedEvent):
                    yield item
                else:
                    step_result = item["_step_result"]

            if self._interrupted.is_set():
                yield NormalizedEvent(
                    kind="turn_completed",
                    text=final_text,
                    terminal_status="interrupted",
                )
                return

            assert step_result is not None
            step_text: str = step_result["text"]
            tool_calls: list[dict[str, Any]] = step_result["tool_calls"]
            telemetry: dict[str, Any] = step_result["telemetry"]

            yield NormalizedEvent(
                kind="status",
                text=self._format_telemetry_status(step, telemetry),
                raw={"telemetry": telemetry},
            )

            # No tool calls: check verification gating before closing out.
            if not tool_calls:
                if (
                    self._state.working_memory.unverified_edits
                    and not verification_nudged
                    and step < self._max_steps
                ):
                    ver = await evaluate_completion_verification(
                        self._state.working_memory.modified_files,
                        step_text,
                        self._state.working_memory.verified_checks,
                        client=self.services.client,
                        hikari_url=self.services.settings.hikari_url,
                        weight_variant=self._weight_variant,
                    )
                    if ver.needs_verification:
                        verification_nudged = True
                        self._state.messages.append(
                            {"role": "assistant", "content": step_text}
                        )
                        self._state.messages.append(
                            {"role": "user", "content": ver.nudge_message}
                        )
                        yield NormalizedEvent(
                            kind="status",
                            text="Verification check triggered before completing turn.",
                            raw={
                                "guardrail": {
                                    "kind": "verification",
                                    "score": ver.score,
                                }
                            },
                        )
                        continue

                self._state.messages.append({"role": "assistant", "content": step_text})
                final_text = step_text
                yield NormalizedEvent(
                    kind="turn_completed",
                    text=final_text,
                    terminal_status="completed",
                    raw={"telemetry": telemetry},
                )
                return

            # Append the assistant tool-call turn strictly in order.
            wire_calls = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": (
                            call["args"]
                            if isinstance(call["args"], dict)
                            else {"_raw": str(call["args"])}
                        ),
                    },
                }
                for call in tool_calls
            ]
            self._state.messages.append(
                {
                    "role": "assistant",
                    "content": step_text,
                    "tool_calls": wire_calls,
                }
            )

            # Execute each tool call in sequence, preserving strict tool_call_id order.
            for call in tool_calls:
                call_id = str(call["id"])
                tool_name = str(call["name"])
                raw_args = call["args"]

                yield NormalizedEvent(
                    kind="tool_call",
                    call_id=call_id,
                    tool_name=tool_name,
                    arguments=raw_args
                    if isinstance(raw_args, dict)
                    else {"raw": str(raw_args)},
                    text=json.dumps(raw_args, ensure_ascii=False)[:1500]
                    if isinstance(raw_args, dict)
                    else str(raw_args)[:1500],
                )

                validated, err_payload = validate_tool_arguments(tool_name, raw_args)
                if err_payload is not None or validated is None:
                    consecutive_validation_errors += 1
                    err_text = json.dumps(err_payload, ensure_ascii=False)
                    self._state.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": tool_name or "invalid_tool",
                            "content": err_text,
                        }
                    )
                    self._state.recent_steps.append(
                        StepRecord(
                            tool_name=tool_name,
                            signature=f"invalid:{tool_name}:{json.dumps(err_payload, sort_keys=True)[:120]}",
                            is_error=True,
                            summary=f"invalid arguments: {err_text[:180]}",
                        )
                    )
                    yield NormalizedEvent(
                        kind="tool_result",
                        call_id=call_id,
                        tool_name=tool_name,
                        text=err_text,
                        is_error=True,
                    )
                    if (
                        consecutive_validation_errors
                        >= MAX_CONSECUTIVE_VALIDATION_ERRORS
                    ):
                        raise HarnessError(
                            "Inkling repeatedly produced invalid tool arguments and stopped to avoid looping."
                        )
                    continue

                consecutive_validation_errors = 0

                # Collaborative ask_user tool pauses the thread with a pending_question card.
                if tool_name == "ask_user":
                    assert isinstance(validated, AskUserArgs)
                    req_id = f"inkling-q-{step}-{call_id}"
                    options_payload = [
                        {"label": opt, "value": opt} for opt in validated.options
                    ]
                    yield NormalizedEvent(
                        kind="pending_question",
                        request_id=req_id,
                        title=validated.title,
                        text=validated.question,
                        tool_name="ask_user",
                        options=options_payload,
                    )
                    answer = await self._wait_for_pending(req_id)
                    if answer is None or self._interrupted.is_set():
                        yield NormalizedEvent(
                            kind="turn_completed",
                            text=final_text,
                            terminal_status="interrupted",
                        )
                        return
                    answer_text = str(
                        answer.get("answer")
                        or answer.get("text")
                        or json.dumps(answer, ensure_ascii=False)
                    )
                    self._state.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": "ask_user",
                            "content": f"User answered: {answer_text}",
                        }
                    )
                    self._state.recent_steps.append(
                        StepRecord(
                            tool_name="ask_user",
                            signature=f"ask_user:{validated.question[:60]}",
                            is_error=False,
                            summary=f"user answered: {answer_text[:120]}",
                        )
                    )
                    yield NormalizedEvent(
                        kind="tool_result",
                        call_id=call_id,
                        tool_name="ask_user",
                        text=f"User answered: {answer_text}",
                        is_error=False,
                    )
                    continue

                # Command safety / approval gating via policy + Decisions API.
                if tool_name == "run_command":
                    assert isinstance(validated, RunCommandArgs)
                    approval = await classify_command_approval(
                        validated.command,
                        cwd=validated.cwd or ".",
                        permission_mode=self.config.permission_mode,
                        session_approved=self._state.approved_for_session,
                        client=self.services.client,
                        hikari_url=self.services.settings.hikari_url,
                        weight_variant=self._weight_variant,
                    )
                    if approval.requires_approval:
                        req_id = f"inkling-cmd-{step}-{call_id}"
                        yield NormalizedEvent(
                            kind="pending_approval",
                            request_id=req_id,
                            title="Allow command?",
                            text=(
                                f"Run command `{validated.command}`?\n{approval.reason}"
                            ),
                            tool_name="run_command",
                            arguments={
                                "command": validated.command,
                                "cwd": validated.cwd or ".",
                            },
                            raw={
                                "guardrail": {
                                    "kind": "command_approval",
                                    "source": approval.source,
                                    "probability": approval.probability,
                                }
                            },
                        )
                        decision_resp = await self._wait_for_pending(req_id)
                        if decision_resp is None or self._interrupted.is_set():
                            yield NormalizedEvent(
                                kind="turn_completed",
                                text=final_text,
                                terminal_status="interrupted",
                            )
                            return
                        decision = str(decision_resp.get("decision", "deny")).lower()
                        if decision == "approve_session":
                            self._state.approved_for_session = True
                        if decision not in ("approve", "approve_session"):
                            declined_msg = (
                                "Command was declined by the user. Do not run it again; "
                                "choose a safer alternative or ask the user how to proceed."
                            )
                            self._state.messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call_id,
                                    "name": "run_command",
                                    "content": declined_msg,
                                }
                            )
                            self._state.recent_steps.append(
                                StepRecord(
                                    tool_name="run_command",
                                    signature=f"declined:{validated.command[:80]}",
                                    is_error=True,
                                    summary=f"user declined `{validated.command[:80]}`",
                                )
                            )
                            yield NormalizedEvent(
                                kind="tool_result",
                                call_id=call_id,
                                tool_name="run_command",
                                text=declined_msg,
                                is_error=True,
                            )
                            continue

                result = await self._toolset.execute(tool_name, validated)
                self._state.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": result.output,
                    }
                )
                if result.file_change is not None:
                    rel_path, change_type = result.file_change
                    self._state.working_memory.record_file_change(rel_path, change_type)
                    yield NormalizedEvent(
                        kind="file_change",
                        file_path=rel_path,
                        change_type=change_type,
                    )
                if tool_name == "run_command":
                    assert isinstance(validated, RunCommandArgs)
                    self._state.working_memory.record_check(
                        validated.command, result.exit_code
                    )
                if result.is_error:
                    self._state.working_memory.record_blocker(
                        f"{tool_name}: {result.output[:180]}"
                    )
                else:
                    self._state.working_memory.clear_blockers()

                sig_args = json.dumps(
                    validated.model_dump(mode="json"),
                    sort_keys=True,
                    ensure_ascii=False,
                )[:120]
                self._state.recent_steps.append(
                    StepRecord(
                        tool_name=tool_name,
                        signature=f"{tool_name}:{sig_args}:{result.is_error}",
                        is_error=result.is_error,
                        summary=f"{sig_args} -> {result.output[:120]}",
                    )
                )
                self._state.recent_steps = self._state.recent_steps[-16:]

                yield NormalizedEvent(
                    kind="tool_result",
                    call_id=call_id,
                    tool_name=tool_name,
                    text=result.output,
                    is_error=result.is_error,
                )

            # Loop / stuck detection guardrail after tool step.
            loop_decision = await evaluate_loop_stuck(
                self._state.recent_steps,
                client=self.services.client,
                hikari_url=self.services.settings.hikari_url,
                weight_variant=self._weight_variant,
            )
            if loop_decision.action == "nudge":
                self._state.messages.append(
                    {"role": "user", "content": loop_decision.message}
                )
                yield NormalizedEvent(
                    kind="status",
                    text="Loop recovery check triggered.",
                    raw={
                        "guardrail": {
                            "kind": "loop_nudge",
                            "is_stuck": loop_decision.is_stuck_score,
                            "should_ask_user": loop_decision.should_ask_score,
                        }
                    },
                )
            elif loop_decision.action == "ask_user":
                req_id = f"inkling-stuck-{step}"
                yield NormalizedEvent(
                    kind="pending_question",
                    request_id=req_id,
                    title="Need your guidance",
                    text=loop_decision.message,
                    options=[
                        {
                            "label": "Try a different approach",
                            "value": "Try a different approach",
                        },
                        {
                            "label": "Stop and summarize findings",
                            "value": "Stop and summarize findings",
                        },
                    ],
                    raw={
                        "guardrail": {
                            "kind": "loop_escalate",
                            "is_stuck": loop_decision.is_stuck_score,
                            "should_ask_user": loop_decision.should_ask_score,
                        }
                    },
                )
                answer = await self._wait_for_pending(req_id)
                if answer is None or self._interrupted.is_set():
                    yield NormalizedEvent(
                        kind="turn_completed",
                        text=final_text,
                        terminal_status="interrupted",
                    )
                    return
                answer_text = str(
                    answer.get("answer")
                    or answer.get("text")
                    or json.dumps(answer, ensure_ascii=False)
                )
                self._state.recent_steps.clear()
                self._state.messages.append(
                    {"role": "user", "content": f"User guidance: {answer_text}"}
                )

        raise HarnessError(
            f"Inkling reached the maximum of {self._max_steps} steps for one turn. "
            "Send a follow-up message to continue from where it left off."
        )


class InklingHarnessAdapter(HarnessAdapter):
    """First-party Inkling 2 Mini coding harness backed by Kadan Server's Hikari engine."""

    harness_id = INKLING_HARNESS_ID
    display_name = "Inkling"
    description = "Kadan Server model."

    def __init__(self, *, max_steps: int = MAX_STEPS_PER_TURN):
        self._max_steps = max_steps
        self._thread_states: dict[UUID, InklingSessionState] = {}

    def get_session_state(self, thread_id: UUID) -> InklingSessionState:
        state = self._thread_states.get(thread_id)
        if state is None:
            state = InklingSessionState()
            self._thread_states[thread_id] = state
        return state

    def drop_session_state(self, thread_id: UUID) -> None:
        self._thread_states.pop(thread_id, None)

    async def check_availability(self, services: LocalServices) -> HarnessAvailability:
        default_models = [
            HarnessModelInfo(
                id=INKLING_DEFAULT_MODEL_ID,
                label="Inkling 2 Mini",
                description="Kadan Server coding model.",
                options={"reasoning_effort": ["low", "medium", "high"]},
            )
        ]
        try:
            raw = await services.json_request("Hikari", "GET", "/v1/models")
            catalog = HikariModels.model_validate(raw)
        except (ServiceError, ValidationError):
            return HarnessAvailability(
                harness_id=self.harness_id,
                display_name=self.display_name,
                description=self.description,
                installed=False,
                binary_path=None,
                auth=HarnessAuthStatus(
                    status="unavailable",
                    login_method="local",
                    message="Kadan Server is offline.",
                ),
                models=default_models,
                default_model=INKLING_DEFAULT_MODEL_ID,
                supports_steer=True,
                supports_resume=True,
                supports_approvals=True,
                supports_questions=True,
                supports_images=True,
            )

        ckpt = _select_inkling_checkpoint(catalog)
        model_id = (
            ckpt.id
            if ckpt is not None
            else (catalog.selected_model_id or INKLING_DEFAULT_MODEL_ID)
        )
        models = [
            HarnessModelInfo(
                id=model_id,
                label=_model_label(model_id),
                description="Kadan Server coding model.",
                options={"reasoning_effort": ["low", "medium", "high"]},
            )
        ]

        if ckpt is None or not ckpt.inference_available:
            auth = HarnessAuthStatus(
                status="unavailable",
                login_method="local",
                message="Download Inkling 2 Mini in Kadan Server → Hikari → Settings.",
            )
        elif ckpt.status == "complete":
            auth = HarnessAuthStatus(
                status="ready",
                login_method="local",
                account_label="Kadan Server",
                message="Ready on Kadan Server.",
            )
        elif ckpt.status == "downloading":
            auth = HarnessAuthStatus(
                status="waiting",
                login_method="local",
                message="Downloading on Kadan Server…",
            )
        else:
            auth = HarnessAuthStatus(
                status="unavailable",
                login_method="local",
                message="Download Inkling 2 Mini in Kadan Server → Hikari → Settings.",
            )

        return HarnessAvailability(
            harness_id=self.harness_id,
            display_name=self.display_name,
            description=self.description,
            installed=True,
            binary_path="hikari-worker-llm",
            auth=auth,
            models=models,
            default_model=model_id,
            supports_steer=True,
            supports_resume=True,
            supports_approvals=True,
            supports_questions=True,
            supports_images=True,
        )

    async def start_sign_in(
        self, services: LocalServices, payload: dict[str, Any] | None = None
    ) -> HarnessAuthStatus:
        return (await self.check_availability(services)).auth

    async def sign_out(self, services: LocalServices) -> HarnessAuthStatus:
        return (await self.check_availability(services)).auth

    async def open_session(
        self, config: SessionConfig, services: LocalServices
    ) -> InklingHarnessSession:
        state = self.get_session_state(config.thread_id)
        return InklingHarnessSession(
            config,
            services,
            state=state,
            max_steps=self._max_steps,
        )
