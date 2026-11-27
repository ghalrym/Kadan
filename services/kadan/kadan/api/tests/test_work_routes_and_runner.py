"""Unit tests for Kadan Work API routes, WorkRunner lifecycle & wakeups, and replay-safe agent tools."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest

# Make Tsuchi importable so we can back Kadan's HTTP client with an in-process Tsuchi ASGI app.
TSUCHI_DIR = Path(__file__).resolve().parents[4] / "tsuchi"
if str(TSUCHI_DIR) not in sys.path:
    sys.path.insert(0, str(TSUCHI_DIR))

from tsuchi.config import Settings as TsuchiSettings  # noqa: E402
from tsuchi.main import create_app as create_tsuchi_app  # noqa: E402

from kadan.api.config import Settings as KadanSettings  # noqa: E402
from kadan.api.pydantic_models.work import WorkProjectCreate  # noqa: E402
from kadan.api.server import create_app as create_kadan_app  # noqa: E402
from kadan.api.services import LocalServices  # noqa: E402
from kadan.runner.tools import work as work_tools  # noqa: E402
from kadan.work.client import WorkClient  # noqa: E402
from kadan.work.contract import (  # noqa: E402
    HarnessAdapter,
    HarnessAuthStatus,
    HarnessAvailability,
    HarnessModelInfo,
    HarnessSession,
    NormalizedEvent,
    SessionConfig,
)
from kadan.work.registry import HarnessRegistry  # noqa: E402
from kadan.work.runner import WorkRunner  # noqa: E402


class ScriptedHarnessSession(HarnessSession):
    def __init__(self, *, ask_approval: bool = False):
        self._ask_approval = ask_approval
        self._answered = asyncio.Event()
        self.received_response: dict | None = None
        self.steered_prompts: list[str] = []
        self._sid = "scripted-sess-1"

    @property
    def session_id(self) -> str | None:
        return self._sid

    async def start_turn(self, prompt: str) -> AsyncIterator[NormalizedEvent]:
        yield NormalizedEvent(kind="session_started", session_id=self._sid)
        yield NormalizedEvent(kind="text_delta", text="Working on it. ")
        if self._ask_approval:
            yield NormalizedEvent(
                kind="pending_approval",
                request_id="req-approve-1",
                title="Run command",
                text="Allow running unit tests?",
                tool_name="bash",
                arguments={"command": "pytest"},
            )
            await self._answered.wait()
        yield NormalizedEvent(kind="text_delta", text=f"Finished: {prompt}")
        yield NormalizedEvent(
            kind="turn_completed",
            text=f"Working on it. Finished: {prompt}",
            terminal_status="completed",
        )

    async def steer(self, prompt: str) -> bool:
        self.steered_prompts.append(prompt)
        return True

    async def interrupt(self) -> None:
        self._answered.set()

    async def respond_pending(self, request_id: str, response: dict) -> None:
        self.received_response = response
        self._answered.set()

    async def close(self) -> None:
        pass


class ScriptedHarnessAdapter(HarnessAdapter):
    harness_id = "codex"
    display_name = "Codex"
    description = "Scripted test harness"

    def __init__(self, *, ask_approval: bool = False):
        self.ask_approval = ask_approval
        self.sessions: list[ScriptedHarnessSession] = []

    async def check_availability(self, services: LocalServices) -> HarnessAvailability:
        return HarnessAvailability(
            harness_id=self.harness_id,
            display_name=self.display_name,
            description=self.description,
            installed=True,
            binary_path="/usr/bin/codex",
            auth=HarnessAuthStatus(
                status="ready", login_method="device_code", message="Ready"
            ),
            models=[HarnessModelInfo(id="gpt-5.3-codex", label="GPT-5.3 Codex")],
            default_model="gpt-5.3-codex",
            supports_steer=True,
        )

    async def start_sign_in(self, services, payload=None) -> HarnessAuthStatus:
        return HarnessAuthStatus(status="ready", login_method="device_code")

    async def sign_out(self, services) -> HarnessAuthStatus:
        return HarnessAuthStatus(status="sign_in_required", login_method="device_code")

    async def open_session(
        self, config: SessionConfig, services: LocalServices
    ) -> HarnessSession:
        sess = ScriptedHarnessSession(ask_approval=self.ask_approval)
        self.sessions.append(sess)
        return sess


@pytest.mark.anyio
async def test_work_runner_pending_approval_and_event_driven_chat_wakeups(tmp_path):
    tsuchi_app = create_tsuchi_app(TsuchiSettings(data_dir=tmp_path / "tsuchi"))
    transport = httpx.ASGITransport(app=tsuchi_app)
    project_dir = tmp_path / "repo"
    project_dir.mkdir()

    adapter = ScriptedHarnessAdapter(ask_approval=True)
    registry = HarnessRegistry([adapter])

    async with (
        tsuchi_app.router.lifespan_context(tsuchi_app),
        httpx.AsyncClient(transport=transport, base_url="http://tsuchi") as client,
    ):
        services = LocalServices(client, KadanSettings(tsuchi_url="http://tsuchi"))
        services.work = WorkClient(services, registry=registry)

        # Create an agent, conversation, and origin run so wakeups target a real conversation.
        await services.json_request(
            "Tsuchi", "POST", "/api/agents", json={"key": "kadan", "name": "Kadan"}
        )
        convo = await services.json_request(
            "Tsuchi",
            "POST",
            "/api/agents/kadan/resources",
            json={"kind": "conversation", "title": "Chat"},
        )
        origin_run = await services.json_request(
            "Tsuchi",
            "POST",
            f"/api/agents/kadan/resources/{convo['id']}/runs",
            json={
                "expected_version": 1,
                "message": {
                    "id": str(uuid4()),
                    "role": "user",
                    "content": "Fix the bug in repo",
                    "created_at": datetime.now(UTC).isoformat(),
                },
            },
        )

        # Register project folder via WorkClient.
        project = await services.work.create_project(
            WorkProjectCreate(title="Repo", root_path=str(project_dir))
        )

        # Invoke native agent tool `work_thread_launch` twice with the same action_id (replay safety).
        action_id = uuid4()
        conv_ctx = SimpleNamespace(
            runs=SimpleNamespace(services=services),
            run=SimpleNamespace(
                agent_key="kadan",
                conversation_id=UUID(convo["id"]),
                id=UUID(origin_run["id"]),
            ),
            action_id=action_id,
        )

        launch_args = work_tools.WorkThreadLaunchArguments(
            project_id=project.id,
            title="Fix bug",
            prompt="Investigate and fix",
            harness="codex",
        )
        first = await work_tools.work_thread_launch.ainvoke(
            launch_args.model_dump(),
            config={"configurable": {"conversation": conv_ctx}},
        )
        replay = await work_tools.work_thread_launch.ainvoke(
            launch_args.model_dump(),
            config={"configurable": {"conversation": conv_ctx}},
        )
        assert first["thread_id"] == replay["thread_id"]
        assert first["run_id"] == replay["run_id"]
        assert f"#work/{first['thread_id']}" in first["link"]

        # Run WorkRunner reconciliation; it should claim the run, pause on approval,
        # and emit a `kind="work"` event-run into the originating conversation.
        runner = WorkRunner(services)
        await runner.reconcile_once()

        # Wait until the pending approval and its chat wakeup are both persisted.
        work_events: list[dict] = []
        for _ in range(80):
            detail = await services.work.get_thread_detail(UUID(first["thread_id"]))
            agent_runs = await services.json_request(
                "Tsuchi",
                "GET",
                f"/api/agents/kadan/resources/{convo['id']}/runs",
            )
            work_events = [
                json.loads(r["trigger_event"]["content"])
                for r in agent_runs
                if r.get("trigger_event") and r["trigger_event"]["kind"] == "work"
            ]
            if detail.thread.run_status == "waiting" and any(
                ev["event_type"] == "pending_approval" for ev in work_events
            ):
                break
            await asyncio.sleep(0.05)

        assert detail.thread.run_status == "waiting"
        assert detail.thread.bucket == "default"
        assert len(detail.thread.pending_requests) == 1
        assert any(ev["event_type"] == "pending_approval" for ev in work_events)

        # Respond via native agent tool `work_pending_request`.
        resp = await work_tools.work_pending_request.ainvoke(
            {
                "thread_id": first["thread_id"],
                "action": "respond",
                "request_id": "req-approve-1",
                "response": {"decision": "approve"},
            },
            config={"configurable": {"conversation": conv_ctx}},
        )
        assert "error" not in resp

        # Wait for the run to complete and deliver the terminal chat wakeup.
        settled: list[dict] = []
        for _ in range(80):
            detail = await services.work.get_thread_detail(UUID(first["thread_id"]))
            agent_runs = await services.json_request(
                "Tsuchi",
                "GET",
                f"/api/agents/kadan/resources/{convo['id']}/runs",
            )
            work_events = [
                json.loads(r["trigger_event"]["content"])
                for r in agent_runs
                if r.get("trigger_event") and r["trigger_event"]["kind"] == "work"
            ]
            settled = [ev for ev in work_events if ev["event_type"] == "thread_settled"]
            if detail.thread.run_status == "completed" and settled:
                break
            await asyncio.sleep(0.05)

        assert detail.thread.run_status == "completed"
        assert detail.thread.harness_session_id == "scripted-sess-1"
        assert adapter.sessions[0].received_response == {"decision": "approve"}
        assert len(settled) == 1
        assert settled[0]["status"] == "completed"
        assert f"#work/{first['thread_id']}" in settled[0]["thread_link"]

        # Snooze into the past, then confirm WorkRunner reconciles it back to `default`.
        past = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
        org = await work_tools.work_thread_organize.ainvoke(
            {
                "thread_id": first["thread_id"],
                "action": "snooze",
                "snoozed_until": past,
            },
            config={"configurable": {"conversation": conv_ctx}},
        )
        assert org["bucket"] == "snoozed"
        await runner.reconcile_once()
        woken = await services.work.get_thread(UUID(first["thread_id"]))
        assert woken.bucket == "default"
        assert woken.snoozed_until is None

        await runner.close()


@pytest.mark.anyio
async def test_kadan_work_http_routes_end_to_end_with_mock_tsuchi(tmp_path):
    tsuchi_app = create_tsuchi_app(TsuchiSettings(data_dir=tmp_path / "tsuchi"))
    kadan_app = create_kadan_app(KadanSettings(tsuchi_url="http://tsuchi"))
    project_dir = tmp_path / "demo-project"
    project_dir.mkdir()

    async with (
        tsuchi_app.router.lifespan_context(tsuchi_app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=tsuchi_app), base_url="http://tsuchi"
        ) as tsuchi_client,
    ):
        services = LocalServices(
            tsuchi_client, KadanSettings(tsuchi_url="http://tsuchi")
        )
        services.work = WorkClient(
            services,
            registry=HarnessRegistry([ScriptedHarnessAdapter()]),
        )
        kadan_app.state.services = services

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=kadan_app), base_url="http://kadan"
        ) as api:
            harnesses = (await api.get("/work/harnesses")).json()
            assert [h["id"] for h in harnesses] == ["codex"]
            assert harnesses[0]["auth"]["status"] == "ready"

            browsed = (
                await api.get("/work/filesystem/browse", params={"path": str(tmp_path)})
            ).json()
            assert any(e["name"] == "demo-project" for e in browsed["entries"])

            proj = await api.post(
                "/work/projects",
                json={"title": "Demo", "root_path": str(project_dir)},
            )
            assert proj.status_code == 201
            proj_id = proj.json()["id"]

            thr = await api.post(
                "/work/threads",
                json={
                    "project_id": proj_id,
                    "title": "Add docs",
                    "harness": "codex",
                    "prompt": "Write docs",
                },
            )
            assert thr.status_code == 201
            thr_id = thr.json()["id"]
            assert thr.json()["bucket"] == "working"

            detail = (await api.get(f"/work/threads/{thr_id}")).json()
            assert detail["project"]["id"] == proj_id
            assert len(detail["runs"]) == 1

            interrupted = (await api.post(f"/work/threads/{thr_id}/interrupt")).json()
            assert interrupted["bucket"] == "default"

            settled = (
                await api.post(
                    f"/work/threads/{thr_id}/organize",
                    json={"action": "settle"},
                )
            ).json()
            assert settled["bucket"] == "settled"

            deleted = await api.delete(f"/work/threads/{thr_id}")
            assert deleted.status_code == 204
