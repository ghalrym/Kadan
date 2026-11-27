"""Pure unit tests for the Inkling 2 Mini coding harness (#530).

No live or synthetic model inference runs are performed; all Hikari HTTP responses
(`/v1/models`, `/v1/chat/completions`, `/v1/decisions`) are mocked in-process.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest

TSUCHI_DIR = Path(__file__).resolve().parents[4] / "tsuchi"
if str(TSUCHI_DIR) not in sys.path:
    sys.path.insert(0, str(TSUCHI_DIR))

from tsuchi.config import Settings as TsuchiSettings
from tsuchi.main import create_app as create_tsuchi_app

from kadan.api.config import Settings as KadanSettings
from kadan.api.pydantic_models.work import (
    WorkProjectCreate,
    WorkRunCreate,
    WorkThreadCreate,
)
from kadan.api.services import LocalServices
from kadan.work.client import WorkClient
from kadan.work.contract import HarnessError, SessionConfig
from kadan.work.inkling_adapter import (
    INKLING_HARNESS_ID,
    STATIC_PREFIX_DIGEST,
    STATIC_SYSTEM_PROMPT,
    InklingHarnessAdapter,
    InklingHarnessSession,
    InklingSessionState,
    compute_static_prefix_digest,
)
from kadan.work.inkling_budget import (
    WorkingMemoryTracker,
    compute_budget_snapshot,
    partition_messages_for_compaction,
    rebuild_compacted_messages,
)
from kadan.work.inkling_decisions import (
    StepRecord,
    classify_command_approval,
    evaluate_completion_verification,
    evaluate_loop_stuck,
)
from kadan.work.inkling_tools import (
    INKLING_TOOL_NAMES,
    INKLING_TOOL_SCHEMAS,
    EditFileArgs,
    FindFilesArgs,
    GitInspectArgs,
    InklingToolset,
    ReadFileArgs,
    RunCommandArgs,
    SearchTextArgs,
    WriteFileArgs,
    canonical_tool_schemas_json,
    cap_tool_output,
    validate_tool_arguments,
)
from kadan.work.registry import HarnessRegistry, default_registry
from kadan.work.runner import WorkRunner

SAMPLE_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+X2dIAAAAASUVORK5CYII="
)


def _sse_stream_bytes(
    *,
    content_chunks: list[str] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    cache: dict[str, Any] | None = None,
    timing: dict[str, Any] | None = None,
) -> bytes:
    frames: list[str] = []

    def _frame(payload: dict[str, Any]) -> None:
        frames.append("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n")

    _frame(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "model": "inkling-2-mini-iq4-nl",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
    )
    for text in content_chunks or []:
        _frame(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "model": "inkling-2-mini-iq4-nl",
                "choices": [
                    {"index": 0, "delta": {"content": text}, "finish_reason": None}
                ],
            }
        )
    if tool_calls:
        indexed = [
            {
                "index": idx,
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": (
                        call["arguments"]
                        if isinstance(call["arguments"], str)
                        else json.dumps(call["arguments"], ensure_ascii=False)
                    ),
                },
            }
            for idx, call in enumerate(tool_calls)
        ]
        _frame(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "model": "inkling-2-mini-iq4-nl",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": indexed},
                        "finish_reason": None,
                    }
                ],
            }
        )
    terminal: dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": "inkling-2-mini-iq4-nl",
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    if cache is not None:
        terminal["cache"] = cache
    if timing is not None:
        terminal["timing"] = timing
    _frame(terminal)
    frames.append("data: [DONE]\n\n")
    return "".join(frames).encode("utf-8")


# ── 1. Compact coding toolset, bounded output & edit mismatch recovery ───────


@pytest.mark.anyio
async def test_toolset_read_find_search_write_edit_and_workspace_containment(tmp_path):
    toolset = InklingToolset(tmp_path, max_output_chars=1200)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "secret.txt").write_text("hidden")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.js").write_text("ignored")

    # write_file creates nested directories and reports created -> modified.
    w1 = await toolset.execute(
        "write_file",
        WriteFileArgs(
            path="src/app.py",
            content="def greet(name):\n    return f'hello {name}'\n\n\ndef add(a, b):\n    return a + b\n",
        ),
    )
    assert w1.ok is True
    assert w1.file_change == ("src/app.py", "created")

    # find_files skips .git and node_modules.
    found = await toolset.execute("find_files", FindFilesArgs(pattern="**/*.py"))
    assert found.ok is True
    assert "src/app.py" in found.output
    assert ".git" not in found.output
    assert "node_modules" not in found.output

    # read_file returns 1-based line numbers and respects explicit range.
    read_res = await toolset.execute(
        "read_file", ReadFileArgs(path="src/app.py", start_line=1, end_line=2)
    )
    assert read_res.ok is True
    assert "1: def greet(name):" in read_res.output
    assert "2:     return f'hello {name}'" in read_res.output
    assert read_res.truncated is True  # more lines remain

    # search_text finds matching lines.
    search_res = await toolset.execute(
        "search_text", SearchTextArgs(query="return a + b", path="src")
    )
    assert search_res.ok is True
    assert "src/app.py:6:" in search_res.output

    # edit_file mismatch returns structured nearby_lines diagnostics for self-correction.
    miss = await toolset.execute(
        "edit_file",
        EditFileArgs(
            path="src/app.py",
            old_text="def greet(name):\n    return f'hi {name}'",
            new_text="def greet(name):\n    return f'welcome {name}'",
        ),
    )
    assert miss.ok is False
    assert miss.is_error is True
    diag = json.loads(miss.output)
    assert diag["error_type"] == "edit_mismatch"
    assert len(diag["nearby_lines"]) >= 1
    assert "1: def greet(name):" in diag["nearby_lines"][0]["snippet"]

    # edit_file succeeds with exact match.
    hit = await toolset.execute(
        "edit_file",
        EditFileArgs(
            path="src/app.py",
            old_text="def greet(name):\n    return f'hello {name}'",
            new_text="def greet(name):\n    return f'welcome {name}'",
        ),
    )
    assert hit.ok is True
    assert hit.file_change == ("src/app.py", "modified")
    assert "welcome" in (tmp_path / "src/app.py").read_text()

    # edit_file reports ambiguous match line numbers unless replace_all=True.
    (tmp_path / "src/dup.txt").write_text("alpha\nbeta\nalpha\n")
    ambig = await toolset.execute(
        "edit_file",
        EditFileArgs(path="src/dup.txt", old_text="alpha", new_text="gamma"),
    )
    assert ambig.ok is False
    ambig_diag = json.loads(ambig.output)
    assert ambig_diag["error_type"] == "ambiguous_edit"
    assert ambig_diag["match_lines"] == [1, 3]

    rep_all = await toolset.execute(
        "edit_file",
        EditFileArgs(
            path="src/dup.txt", old_text="alpha", new_text="gamma", replace_all=True
        ),
    )
    assert rep_all.ok is True
    assert (tmp_path / "src/dup.txt").read_text() == "gamma\nbeta\ngamma\n"

    # Workspace containment rejects path escape.
    escape = await toolset.execute("read_file", ReadFileArgs(path="../../etc/passwd"))
    assert escape.ok is False
    assert "outside the project workspace" in escape.output


@pytest.mark.anyio
async def test_run_command_bounded_head_tail_timeout_and_git_inspect(tmp_path):
    toolset = InklingToolset(tmp_path, max_output_chars=400)

    # Large stdout is capped with head + tail preserved and a truncation notice.
    cmd_res = await toolset.execute(
        "run_command",
        RunCommandArgs(
            command=(
                f"{sys.executable} -c \"print('HEAD_MARKER'); "
                "print('x' * 2000); print('TAIL_MARKER')\""
            )
        ),
    )
    assert cmd_res.ok is True
    assert cmd_res.truncated is True
    assert "HEAD_MARKER" in cmd_res.output
    assert "TAIL_MARKER" in cmd_res.output
    assert "output truncated" in cmd_res.output

    # Timeout cleans up cleanly and reports timeout.
    timeout_res = await toolset.execute(
        "run_command",
        RunCommandArgs(
            command=f'{sys.executable} -c "import time; time.sleep(5)"',
            timeout_seconds=1,
        ),
    )
    assert timeout_res.ok is False
    assert "timed out" in timeout_res.output

    # git_inspect works in a real local git repository.
    def _init_git_repo() -> None:
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=tmp_path,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True
        )
        (tmp_path / "README.md").write_text("hello\n")
        subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
        (tmp_path / "README.md").write_text("hello world\n")

    await asyncio.to_thread(_init_git_repo)

    st = await toolset.execute("git_inspect", GitInspectArgs(mode="status"))
    assert st.ok is True
    assert "README.md" in st.output

    diff = await toolset.execute("git_inspect", GitInspectArgs(mode="diff"))
    assert diff.ok is True
    assert "+hello world" in diff.output

    log = await toolset.execute(
        "git_inspect", GitInspectArgs(mode="log", max_commits=5)
    )
    assert log.ok is True
    assert "init" in log.output


def test_guided_argument_validation_and_cap_tool_output():
    model, err = validate_tool_arguments("read_file", {"path": "a.py", "extra_bad": 1})
    assert model is None
    assert err is not None
    assert err["error_type"] == "invalid_arguments"
    assert any(p["field"] == "extra_bad" for p in err["problems"])

    model2, err2 = validate_tool_arguments("nonexistent_tool", {})
    assert model2 is None
    assert err2 is not None
    assert err2["error_type"] == "unknown_tool"
    assert set(err2["valid_tools"]) == set(INKLING_TOOL_NAMES)

    model3, err3 = validate_tool_arguments("read_file", "{bad json")
    assert model3 is None
    assert err3 is not None
    assert err3["error_type"] == "invalid_json"

    capped, was_capped = cap_tool_output("A" * 500 + "Z" * 500, max_chars=250)
    assert was_capped is True
    assert capped.startswith("A")
    assert capped.endswith("Z")
    assert "output truncated" in capped


# ── 2. Prefix immutability, token budgeting & structured compaction ──────────


def test_static_prefix_is_canonical_and_atomic_compaction_preserves_prefix():
    digest_a = compute_static_prefix_digest(
        STATIC_SYSTEM_PROMPT, canonical_tool_schemas_json()
    )
    digest_b = compute_static_prefix_digest()
    assert digest_a == digest_b == STATIC_PREFIX_DIGEST
    assert len(INKLING_TOOL_SCHEMAS) == 8

    # Construct conversation with tool_call + tool_result pairs.
    messages = [
        {"role": "system", "content": STATIC_SYSTEM_PROMPT},
        {"role": "user", "content": "Fix parser bug"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": {"path": "a.py"}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "read_file",
            "content": "1: old",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "edit_file", "arguments": {"path": "a.py"}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c2",
            "name": "edit_file",
            "content": "Edited a.py",
        },
        {"role": "user", "content": "Now run tests"},
    ]

    sys_msg, to_compact, tail = partition_messages_for_compaction(
        messages, keep_tail_groups=2
    )
    assert sys_msg["content"] == STATIC_SYSTEM_PROMPT
    # Tool-call c2 + its tool result must stay paired in `tail`.
    assert tail[0]["role"] == "assistant"
    assert tail[1]["role"] == "tool"
    assert tail[1]["tool_call_id"] == "c2"
    assert tail[2]["role"] == "user"
    assert len(to_compact) == 3  # initial user + c1 assistant + c1 tool

    tracker = WorkingMemoryTracker()
    tracker.record_goal_if_empty("Fix parser bug")
    tracker.record_file_change("a.py", "modified")
    tracker.record_check("pytest", 0)
    summary = tracker.format_deterministic_summary(to_compact)
    for heading in (
        "## Goal",
        "## Modified files",
        "## Verified checks",
        "## Open blockers",
        "## Next steps",
    ):
        assert heading in summary

    rebuilt = rebuild_compacted_messages(sys_msg, summary, tail)
    assert rebuilt[0] == sys_msg
    assert compute_static_prefix_digest(rebuilt[0]["content"]) == STATIC_PREFIX_DIGEST
    assert rebuilt[1]["role"] == "user"
    assert "[Working-memory compaction checkpoint]" in rebuilt[1]["content"]

    budget = compute_budget_snapshot(
        rebuilt, context_limit=4096, tools_tokens=300, reserved_output_tokens=1024
    )
    assert budget.context_limit == 4096
    assert budget.available_prompt_tokens == 3072
    assert budget.exceeds_limit is False


# ── 3. Decisions guardrails with mocked Hikari responses ─────────────────────


@pytest.mark.anyio
async def test_decisions_guardrails_tag_base_variant_and_trigger_expected_actions():
    recorded_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/decisions"
        body = json.loads(request.content.decode())
        recorded_bodies.append(body)
        assert body["weight_variant"] == "base"
        keys = [q["key"] for q in body["questions"]]
        if "requires_approval" in keys:
            return httpx.Response(
                200,
                json={
                    "answers": [
                        {"key": "requires_approval", "type": "Noul", "value": 0.91}
                    ]
                },
            )
        if "is_stuck" in keys:
            return httpx.Response(
                200,
                json={
                    "answers": [
                        {"key": "is_stuck", "type": "Noul", "value": 0.88},
                        {"key": "should_ask_user", "type": "Noul", "value": 0.20},
                    ]
                },
            )
        if "needs_verification" in keys:
            return httpx.Response(
                200,
                json={
                    "answers": [
                        {"key": "needs_verification", "type": "Noul", "value": 0.82}
                    ]
                },
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://hikari"
    ) as client:
        # Hard-stop destructive pattern never even relies on session approval.
        hard = await classify_command_approval(
            "rm -rf /tmp/foo",
            cwd=".",
            permission_mode="full_auto",
            session_approved=True,
            client=client,
            hikari_url="http://hikari",
        )
        assert hard.requires_approval is True
        assert hard.source == "policy"

        # Ambiguous command invokes Decisions with weight_variant="base".
        ambig = await classify_command_approval(
            "npm install foo",
            cwd=".",
            permission_mode="ask",
            session_approved=False,
            client=client,
            hikari_url="http://hikari",
        )
        assert ambig.requires_approval is True
        assert ambig.source == "decisions"
        assert ambig.probability == pytest.approx(0.91)

        # Loop detection nudges when 3 consecutive errors occur and Decisions flags is_stuck.
        failing_steps = [
            StepRecord("edit_file", "edit:a.py:True", True, "mismatch"),
            StepRecord("edit_file", "edit:a.py:True", True, "mismatch"),
            StepRecord("edit_file", "edit:a.py:True", True, "mismatch"),
        ]
        loop_dec = await evaluate_loop_stuck(
            failing_steps, client=client, hikari_url="http://hikari"
        )
        assert loop_dec.action == "nudge"
        assert loop_dec.is_stuck_score == pytest.approx(0.88)

        # Verification gating nudges when code files changed without tests.
        ver = await evaluate_completion_verification(
            {"src/app.py": "modified"},
            "I fixed the bug.",
            [],
            client=client,
            hikari_url="http://hikari",
        )
        assert ver.needs_verification is True
        assert ver.score == pytest.approx(0.82)

    assert len(recorded_bodies) == 3
    assert all(b["weight_variant"] == "base" for b in recorded_bodies)


# ── 4. Full InklingHarnessSession loop: base variant, prefix immutability,
#       validation recovery, approvals, verification, multimodal, compaction,
#       steering & cancellation ──────────────────────────────────────────────


@pytest.mark.anyio
async def test_inkling_session_full_loop_prefix_telemetry_multimodal_and_guardrails(
    tmp_path,
):
    (tmp_path / "hello.py").write_text("VALUE = 1\n")
    chat_requests: list[dict[str, Any]] = []
    decision_requests: list[dict[str, Any]] = []
    step_counter = {"n": 0}

    def hikari_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "selected_model_id": "inkling-2-mini-abliterated-iq4-nl",
                    "models": [
                        {
                            "id": "inkling-2-mini-iq4-nl",
                            "kind": "llm",
                            "status": "complete",
                            "inference_available": True,
                            "context_limit": 65536,
                        },
                        {
                            "id": "inkling-2-mini-abliterated-iq4-nl",
                            "kind": "llm",
                            "status": "complete",
                            "inference_available": True,
                            "context_limit": 65536,
                        },
                    ],
                },
            )
        if request.url.path == "/v1/decisions":
            body = json.loads(request.content.decode())
            decision_requests.append(body)
            keys = [q["key"] for q in body["questions"]]
            if "needs_verification" in keys:
                return httpx.Response(
                    200,
                    json={
                        "answers": [
                            {"key": "needs_verification", "type": "Noul", "value": 0.9}
                        ]
                    },
                )
            if "requires_approval" in keys:
                return httpx.Response(
                    200,
                    json={
                        "answers": [
                            {"key": "requires_approval", "type": "Noul", "value": 0.85}
                        ]
                    },
                )
            return httpx.Response(
                200,
                json={
                    "answers": [{"key": k, "type": "Noul", "value": 0.1} for k in keys]
                },
            )
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content.decode())
            chat_requests.append(body)
            step_counter["n"] += 1
            n = step_counter["n"]
            if n == 1:
                # Step 1: Intentionally emit invalid arguments to verify guided recovery.
                sse = _sse_stream_bytes(
                    tool_calls=[
                        {
                            "id": "call_bad",
                            "name": "edit_file",
                            "arguments": {"path": "hello.py", "bogus": True},
                        }
                    ],
                    finish_reason="tool_calls",
                    cache={
                        "hit": False,
                        "reused_tokens": 0,
                        "stored_tokens": 512,
                        "reason": "cold",
                    },
                    timing={
                        "weight_swap_ms": 145.2,
                        "prefill_ms": 38.0,
                        "decode_tokens_per_second": 42.5,
                    },
                )
            elif n == 2:
                # Step 2: Valid edit_file after guided recovery.
                sse = _sse_stream_bytes(
                    tool_calls=[
                        {
                            "id": "call_edit",
                            "name": "edit_file",
                            "arguments": {
                                "path": "hello.py",
                                "old_text": "VALUE = 1",
                                "new_text": "VALUE = 2",
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                    cache={
                        "hit": True,
                        "reused_tokens": 512,
                        "stored_tokens": 700,
                        "reason": "checkpoint_hit",
                    },
                    timing={
                        "weight_swap_ms": 0.0,
                        "prefill_ms": 12.0,
                        "decode_tokens_per_second": 48.0,
                    },
                )
            elif n == 3:
                # Step 3: Attempt to conclude without verifying -> verification guardrail nudges!
                sse = _sse_stream_bytes(
                    content_chunks=["Updated VALUE to 2."],
                    finish_reason="stop",
                    cache={
                        "hit": True,
                        "reused_tokens": 700,
                        "stored_tokens": 760,
                        "reason": "checkpoint_hit",
                    },
                )
            elif n == 4:
                # Step 4: Run verification command (requires approval via Decisions).
                sse = _sse_stream_bytes(
                    tool_calls=[
                        {
                            "id": "call_cmd",
                            "name": "run_command",
                            "arguments": {
                                "command": f'{sys.executable} -c "import hello; assert hello.VALUE == 2"'
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                    cache={
                        "hit": True,
                        "reused_tokens": 760,
                        "stored_tokens": 820,
                        "reason": "checkpoint_hit",
                    },
                )
            else:
                # Step 5: Final verified response.
                sse = _sse_stream_bytes(
                    content_chunks=["Updated VALUE to 2 and verified with Python."],
                    finish_reason="stop",
                    cache={
                        "hit": True,
                        "reused_tokens": 820,
                        "stored_tokens": 900,
                        "reason": "checkpoint_hit",
                    },
                )
            return httpx.Response(
                200,
                content=sse,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(hikari_handler), base_url="http://hikari"
    ) as client:
        services = LocalServices(client, KadanSettings(hikari_url="http://hikari"))
        adapter = InklingHarnessAdapter()
        avail = await adapter.check_availability(services)
        assert avail.harness_id == INKLING_HARNESS_ID
        assert avail.auth.status == "ready"
        assert avail.auth.login_method == "local"
        assert avail.default_model == "inkling-2-mini-iq4-nl"
        assert avail.supports_images is True

        config = SessionConfig(
            thread_id=uuid4(),
            working_dir=str(tmp_path),
            priority="background",
        )
        session = await adapter.open_session(config, services)
        events = []
        steered = False
        async for ev in session.start_turn(
            "Change VALUE to 2 based on screenshot",
            images=[SAMPLE_PNG_DATA_URL],
        ):
            events.append(ev)
            if ev.kind == "tool_result" and ev.call_id == "call_edit" and not steered:
                steered = await session.steer("Make sure to verify VALUE == 2")
            if ev.kind == "pending_approval":
                await session.respond_pending(ev.request_id, {"decision": "approve"})

        await session.close()

    assert steered is True
    assert (tmp_path / "hello.py").read_text() == "VALUE = 2\n"
    assert len(chat_requests) == 5

    # Every single Hikari chat and decision call requested base weights & background priority.
    for req in chat_requests:
        assert req["weight_variant"] == "base"
        assert req["priority"] == "background"
        assert req["reuse_prefix"] is True
        assert req["conversation_id"] == f"inkling:{config.thread_id}"
        # Strictly byte-identical static prefix across every step!
        assert req["messages"][0] == {"role": "system", "content": STATIC_SYSTEM_PROMPT}
        assert req["tools"] == list(INKLING_TOOL_SCHEMAS)

    assert all(d["weight_variant"] == "base" for d in decision_requests)

    # First user message carried multimodal image_url part for mmproj.
    first_user_msg = chat_requests[0]["messages"][1]
    assert first_user_msg["role"] == "user"
    assert isinstance(first_user_msg["content"], list)
    assert first_user_msg["content"][0] == {
        "type": "text",
        "text": "Change VALUE to 2 based on screenshot",
    }
    assert first_user_msg["content"][1] == {
        "type": "image_url",
        "image_url": {"url": SAMPLE_PNG_DATA_URL},
    }

    # Guided validation recovery fed structured invalid_arguments back on step 2.
    step2_tool_msg = chat_requests[1]["messages"][3]
    assert step2_tool_msg["role"] == "tool"
    assert json.loads(step2_tool_msg["content"])["error_type"] == "invalid_arguments"

    # Telemetry surfaced weight_swap_ms and cache hits in status events.
    telemetry_events = [
        e.raw["telemetry"]
        for e in events
        if e.kind == "status" and "telemetry" in e.raw
    ]
    assert telemetry_events[0]["weight_swap_ms"] == pytest.approx(145.2)
    assert telemetry_events[1]["cache_hit"] is True
    assert telemetry_events[1]["cache_reason"] == "checkpoint_hit"

    # File change, pending approval, and final completion emitted.
    assert any(e.kind == "file_change" and e.file_path == "hello.py" for e in events)
    assert any(e.kind == "pending_approval" for e in events)
    assert events[-1].kind == "turn_completed"
    assert events[-1].terminal_status == "completed"
    assert "verified" in events[-1].text.lower()


@pytest.mark.anyio
async def test_inkling_compaction_resets_onto_static_prefix_and_rejects_oversized_prompt(
    tmp_path,
):
    chat_requests: list[dict[str, Any]] = []
    compaction_called = {"n": 0}

    def hikari_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content.decode())
            chat_requests.append(body)
            if body.get("stream") is False:
                compaction_called["n"] += 1
                assert body["weight_variant"] == "base"
                assert body["reuse_prefix"] is False
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "## Goal\nBuild feature\n## Modified files\n- a.py\n## Verified checks\n- None\n## Open blockers\n- None\n## Next steps\n- Continue",
                                }
                            }
                        ]
                    },
                )
            return httpx.Response(
                200,
                content=_sse_stream_bytes(
                    content_chunks=["Continued cleanly after compaction."],
                    cache={
                        "hit": True,
                        "reused_tokens": 300,
                        "stored_tokens": 450,
                        "reason": "static_prefix_hit",
                    },
                ),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404)

    # Pre-seed a long history approaching a 4096-token context window.
    state = InklingSessionState(hydrated=True)
    state.working_memory.record_goal_if_empty("Build feature")
    state.working_memory.record_file_change("a.py", "modified")
    state.working_memory.record_check("pytest", 0)
    for idx in range(6):
        state.messages.append(
            {"role": "user", "content": f"Step {idx}: " + ("x" * 900)}
        )
        state.messages.append(
            {"role": "assistant", "content": f"Done {idx}: " + ("y" * 900)}
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(hikari_handler), base_url="http://hikari"
    ) as client:
        services = LocalServices(client, KadanSettings(hikari_url="http://hikari"))
        session = InklingHarnessSession(
            SessionConfig(thread_id=uuid4(), working_dir=str(tmp_path)),
            services,
            state=state,
            context_limit=4096,
        )
        events = [ev async for ev in session.start_turn("Wrap up the feature")]
        await session.close()

    assert compaction_called["n"] == 1
    assert any(e.kind == "status" and "compaction" in e.raw for e in events)
    stream_req = next(r for r in chat_requests if r.get("stream") is True)
    assert stream_req["messages"][0] == {
        "role": "system",
        "content": STATIC_SYSTEM_PROMPT,
    }
    assert (
        "[Working-memory compaction checkpoint]" in stream_req["messages"][1]["content"]
    )
    assert events[-1].terminal_status == "completed"

    # A single oversized prompt that cannot fit even after compaction fails cleanly (no silent truncation).
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(hikari_handler), base_url="http://hikari"
    ) as client:
        services = LocalServices(client, KadanSettings(hikari_url="http://hikari"))
        oversized_session = InklingHarnessSession(
            SessionConfig(thread_id=uuid4(), working_dir=str(tmp_path)),
            services,
            state=InklingSessionState(hydrated=True),
            context_limit=4096,
        )
        with pytest.raises(HarnessError, match="exceeds Inkling 2 Mini's active"):
            _ = [ev async for ev in oversized_session.start_turn("Z" * 30_000)]
        await oversized_session.close()


@pytest.mark.anyio
async def test_inkling_availability_states_and_interrupt_during_pending_question(
    tmp_path,
):
    # 1a. Kadan Server with abliterated Inkling 2 Mini downloaded is ready out of the box.
    def abliterated_ready_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "selected_model_id": "inkling-2-mini-abliterated-iq4-nl",
                    "models": [
                        {
                            "id": "inkling-2-mini-abliterated-iq4-nl",
                            "kind": "llm",
                            "status": "complete",
                            "inference_available": True,
                            "context_limit": 65536,
                        },
                    ],
                },
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(abliterated_ready_handler),
        base_url="http://hikari",
    ) as client:
        services = LocalServices(client, KadanSettings(hikari_url="http://hikari"))
        avail = await InklingHarnessAdapter().check_availability(services)
        assert avail.installed is True
        assert avail.display_name == "Inkling"
        assert avail.auth.status == "ready"
        assert avail.auth.login_method == "local"
        assert avail.default_model == "inkling-2-mini-abliterated-iq4-nl"

    # 1b. No Inkling 2 Mini checkpoint downloaded yet -> plain-language download guidance.
    def missing_model_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "selected_model_id": "inkling-2-mini-iq4-nl",
                    "models": [
                        {
                            "id": "inkling-2-mini-iq4-nl",
                            "kind": "llm",
                            "status": "not_downloaded",
                            "inference_available": True,
                        },
                    ],
                },
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(missing_model_handler), base_url="http://hikari"
    ) as client:
        services = LocalServices(client, KadanSettings(hikari_url="http://hikari"))
        avail = await InklingHarnessAdapter().check_availability(services)
        assert avail.installed is True
        assert avail.auth.status == "unavailable"
        assert avail.auth.login_method == "local"
        assert "Hikari → Settings" in avail.auth.message
        assert "HIKARI_LLM" not in avail.auth.message  # non-technical UI copy

    # 2. Interrupt during pending ask_user exits cleanly with terminal_status="interrupted".
    def ask_user_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200, json={"selected_model_id": "inkling-2-mini-iq4-nl", "models": []}
            )
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                content=_sse_stream_bytes(
                    tool_calls=[
                        {
                            "id": "call_q",
                            "name": "ask_user",
                            "arguments": {
                                "title": "Choose format",
                                "question": "JSON or YAML?",
                                "options": ["JSON", "YAML"],
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                ),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(ask_user_handler), base_url="http://hikari"
    ) as client:
        services = LocalServices(client, KadanSettings(hikari_url="http://hikari"))
        session = InklingHarnessSession(
            SessionConfig(thread_id=uuid4(), working_dir=str(tmp_path)),
            services,
            context_limit=65536,
        )
        events = []
        async for ev in session.start_turn("Generate config"):
            events.append(ev)
            if ev.kind == "pending_question":
                await session.interrupt()
        await session.close()

    assert any(
        e.kind == "pending_question" and e.title == "Choose format" for e in events
    )
    assert events[-1].kind == "turn_completed"
    assert events[-1].terminal_status == "interrupted"


@pytest.mark.anyio
async def test_default_registry_and_work_runner_integration_with_inkling(tmp_path):
    reg = default_registry()
    ids = [adapter.harness_id for adapter in reg.list()]
    assert "inkling" in ids
    assert ids[0] == "inkling"

    tsuchi_app = create_tsuchi_app(TsuchiSettings(data_dir=tmp_path / "tsuchi"))
    tsuchi_transport = httpx.ASGITransport(app=tsuchi_app)
    project_dir = tmp_path / "repo"
    project_dir.mkdir()

    seen_priorities: list[str] = []
    seen_variants: list[str | None] = []
    seen_images: list[Any] = []

    def hikari_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "selected_model_id": "inkling-2-mini-iq4-nl",
                    "models": [
                        {
                            "id": "inkling-2-mini-iq4-nl",
                            "kind": "llm",
                            "status": "complete",
                            "inference_available": True,
                            "context_limit": 65536,
                        }
                    ],
                },
            )
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content.decode())
            seen_priorities.append(body["priority"])
            seen_variants.append(body.get("weight_variant"))
            seen_images.append(body["messages"][-1]["content"])
            return httpx.Response(
                200,
                content=_sse_stream_bytes(
                    content_chunks=["Looks good from the screenshot."],
                    cache={
                        "hit": True,
                        "reused_tokens": 200,
                        "stored_tokens": 350,
                        "reason": "static_prefix_hit",
                    },
                ),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404)

    hikari_transport = httpx.MockTransport(hikari_handler)

    class RoutingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.host == "tsuchi":
                return await tsuchi_transport.handle_async_request(request)
            return await hikari_transport.handle_async_request(request)

    async with (
        tsuchi_app.router.lifespan_context(tsuchi_app),
        httpx.AsyncClient(transport=RoutingTransport()) as client,
    ):
        services = LocalServices(
            client,
            KadanSettings(tsuchi_url="http://tsuchi", hikari_url="http://hikari"),
        )
        services.work = WorkClient(
            services,
            registry=HarnessRegistry([InklingHarnessAdapter()]),
        )

        project = await services.work.create_project(
            WorkProjectCreate(title="Repo", root_path=str(project_dir))
        )
        thread, _ = await services.work.create_thread(
            WorkThreadCreate(
                project_id=project.id,
                title="Inspect UI screenshot",
                harness="inkling",
                prompt="Check layout alignment",
                images=[SAMPLE_PNG_DATA_URL],
            )
        )

        runner = WorkRunner(services)
        await runner.reconcile_once()
        active = runner._active.get(thread.id)
        if active is not None and active.task is not None:
            await asyncio.wait_for(active.task, timeout=15.0)
        detail = await services.work.get_thread_detail(UUID(str(thread.id)))

        await runner.close()

    assert detail.thread.run_status == "completed"
    assert detail.thread.harness == "inkling"
    assert detail.thread.harness_session_id == f"inkling:{thread.id}"
    assert seen_variants == ["base"]
    assert seen_priorities == ["interactive"]
    assert isinstance(seen_images[0], list)
    assert any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for part in seen_images[0]
    )
    assert any(
        item.kind == "status" and "telemetry" in item.data for item in detail.activity
    )
    assert any(
        item.kind == "user_message" and item.data.get("image_count") == 1
        for item in detail.activity
    )


@pytest.mark.anyio
async def test_work_runner_queues_image_followup_instead_of_text_only_steer(tmp_path):
    """An image-bearing follow-up while an Inkling run is active must never be consumed by text-only steer."""
    tsuchi_app = create_tsuchi_app(TsuchiSettings(data_dir=tmp_path / "tsuchi"))
    tsuchi_transport = httpx.ASGITransport(app=tsuchi_app)
    project_dir = tmp_path / "repo"
    project_dir.mkdir()

    turn1_started = asyncio.Event()
    release_turn1 = asyncio.Event()
    received_user_contents: list[Any] = []
    call_count = {"n": 0}

    async def hikari_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "selected_model_id": "inkling-2-mini-iq4-nl",
                    "models": [
                        {
                            "id": "inkling-2-mini-iq4-nl",
                            "kind": "llm",
                            "status": "complete",
                            "inference_available": True,
                            "context_limit": 65536,
                        }
                    ],
                },
            )
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content.decode())
            call_count["n"] += 1
            received_user_contents.append(body["messages"][-1]["content"])
            if call_count["n"] == 1:
                turn1_started.set()
                await release_turn1.wait()
                sse = _sse_stream_bytes(content_chunks=["Finished first turn."])
            else:
                sse = _sse_stream_bytes(
                    content_chunks=["Reviewed screenshot on second turn."]
                )
            return httpx.Response(
                200,
                content=sse,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404)

    hikari_transport = httpx.MockTransport(hikari_handler)

    class RoutingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.host == "tsuchi":
                return await tsuchi_transport.handle_async_request(request)
            return await hikari_transport.handle_async_request(request)

    async with (
        tsuchi_app.router.lifespan_context(tsuchi_app),
        httpx.AsyncClient(transport=RoutingTransport()) as client,
    ):
        services = LocalServices(
            client,
            KadanSettings(tsuchi_url="http://tsuchi", hikari_url="http://hikari"),
        )
        services.work = WorkClient(
            services,
            registry=HarnessRegistry([InklingHarnessAdapter()]),
        )

        project = await services.work.create_project(
            WorkProjectCreate(title="Repo", root_path=str(project_dir))
        )
        thread, _ = await services.work.create_thread(
            WorkThreadCreate(
                project_id=project.id,
                title="Active run with image followup",
                harness="inkling",
                prompt="Start first turn",
            )
        )

        runner = WorkRunner(services)
        await runner.reconcile_once()
        await asyncio.wait_for(turn1_started.wait(), timeout=5.0)

        # Enqueue a follow-up with mode="steer" AND an attached image while turn 1 is running.
        followup = await services.work.send_message(
            thread.id,
            WorkRunCreate(
                prompt="Check this screenshot too",
                mode="steer",
                images=[SAMPLE_PNG_DATA_URL],
            ),
        )

        # Reconcile while turn 1 is still active: the image run must NOT be consumed by text-only steer.
        await runner.reconcile_once()
        runs_mid = await services.json_request(
            "Tsuchi", "GET", f"/api/work/threads/{thread.id}/runs"
        )
        followup_mid = next(r for r in runs_mid if r["id"] == str(followup.id))
        assert followup_mid["status"] == "queued"

        # Let turn 1 finish, then reconcile so turn 2 runs with its image intact.
        release_turn1.set()
        active1 = runner._active.get(thread.id)
        if active1 is not None and active1.task is not None:
            await asyncio.wait_for(active1.task, timeout=10.0)

        await runner.reconcile_once()
        active2 = runner._active.get(thread.id)
        assert active2 is not None and active2.task is not None
        await asyncio.wait_for(active2.task, timeout=10.0)
        await runner.close()

    assert len(received_user_contents) == 2
    assert received_user_contents[0] == "Start first turn"
    assert isinstance(received_user_contents[1], list)
    assert received_user_contents[1][0] == {
        "type": "text",
        "text": "Check this screenshot too",
    }
    assert received_user_contents[1][1] == {
        "type": "image_url",
        "image_url": {"url": SAMPLE_PNG_DATA_URL},
    }


@pytest.mark.anyio
async def test_inkling_works_out_of_the_box_with_abliterated_server_and_auto_titles(
    tmp_path,
):
    """When Kadan Server only serves the abliterated Inkling checkpoint, Work uses it cleanly and auto-names threads."""
    tsuchi_app = create_tsuchi_app(TsuchiSettings(data_dir=tmp_path / "tsuchi"))
    tsuchi_transport = httpx.ASGITransport(app=tsuchi_app)
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    seen_variants: list[str | None] = []

    def hikari_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "selected_model_id": "inkling-2-mini-abliterated-iq4-nl",
                    "models": [
                        {
                            "id": "inkling-2-mini-abliterated-iq4-nl",
                            "kind": "llm",
                            "status": "complete",
                            "inference_available": True,
                            "context_limit": 65536,
                        }
                    ],
                },
            )
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content.decode())
            seen_variants.append(body.get("weight_variant"))
            return httpx.Response(
                200,
                content=_sse_stream_bytes(content_chunks=["Done."]),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404)

    hikari_transport = httpx.MockTransport(hikari_handler)

    class RoutingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.host == "tsuchi":
                return await tsuchi_transport.handle_async_request(request)
            return await hikari_transport.handle_async_request(request)

    async with (
        tsuchi_app.router.lifespan_context(tsuchi_app),
        httpx.AsyncClient(transport=RoutingTransport()) as client,
    ):
        services = LocalServices(
            client,
            KadanSettings(tsuchi_url="http://tsuchi", hikari_url="http://hikari"),
        )
        services.work = WorkClient(
            services, registry=HarnessRegistry([InklingHarnessAdapter()])
        )
        project = await services.work.create_project(
            WorkProjectCreate(root_path=str(project_dir))
        )
        # No manual title supplied: auto-named from the first prompt.
        thread, _ = await services.work.create_thread(
            WorkThreadCreate(
                project_id=project.id,
                harness="inkling",
                prompt="Fix the sign-in redirect bug in auth.ts",
            )
        )
        assert thread.title == "Fix the sign-in redirect bug in auth.ts"

        runner = WorkRunner(services)
        await runner.reconcile_once()
        active = runner._active.get(thread.id)
        if active is not None and active.task is not None:
            await asyncio.wait_for(active.task, timeout=15.0)
        detail = await services.work.get_thread_detail(UUID(str(thread.id)))
        await runner.close()

    assert detail.thread.run_status == "completed"
    assert seen_variants == ["abliterated"]
