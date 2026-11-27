"""Unit tests for Kadan update checking, staging, applying, and agent tools."""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from kadan.api.config import ServiceError, Settings
from kadan.api.pydantic_models.agent_runs import AgentRun, RunCall
from kadan.api.pydantic_models.resources import Message, Resource
from kadan.api.pydantic_models.updates import (
    ApplyUpdateResponse,
    DesktopClientStatus,
    StageUpdateResponse,
    UpdateCheckResult,
)
from kadan.api.server import create_app
from kadan.api.update_client import UpdateClient, is_newer_version, parse_version
from kadan.runner.agent_inference import ToolCommand
from kadan.runner.agent_runs import AgentRuns, RunState


def test_version_parsing_and_comparison():
    assert parse_version("0.1.0") == (0, 1, 0, 0)
    assert parse_version("v0.1.20261126") == (0, 1, 20261126, 0)
    assert parse_version("v0.1.20261126-2") == (0, 1, 20261126, 2)
    assert is_newer_version("0.1.0", "0.1.20261126")
    assert is_newer_version("0.1.20261126", "0.1.20261126-2")
    assert is_newer_version("0.1.20261126", "0.2.0")
    assert not is_newer_version("0.1.20261126", "0.1.20261126")
    assert not is_newer_version("0.1.20261127", "0.1.20261126")


def test_update_client_delegates_to_kadan_server_control_endpoint():
    async def run():
        seen_auth: list[str | None] = []
        seen_paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_auth.append(request.headers.get("authorization"))
            seen_paths.append(f"{request.method} {request.url.path}")
            if request.url.path == "/updates":
                return httpx.Response(
                    200,
                    json={
                        "current_version": "0.1.0",
                        "latest_version": "0.1.20261126",
                        "update_available": True,
                        "release_tag": "v0.1.20261126",
                        "release_url": "https://github.com/ghalrym/Kadan/releases/tag/v0.1.20261126",
                        "release_notes": "Bug fixes",
                        "can_self_update": True,
                        "install_kind": "appimage",
                        "desktop": {
                            "connected": True,
                            "current_version": "0.1.0",
                            "update_available": True,
                            "can_self_update": True,
                        },
                    },
                )
            if request.url.path == "/updates/stage":
                return httpx.Response(
                    200,
                    json={
                        "ok": True,
                        "target_version": "0.1.20261126",
                        "server_staged": True,
                        "desktop_update_available": True,
                        "message": "Staged",
                    },
                )
            if request.url.path == "/updates/apply":
                return httpx.Response(
                    200,
                    json={
                        "ok": True,
                        "target_version": "0.1.20261126",
                        "server_restarting": True,
                        "desktop_update_requested": True,
                        "message": "Restarting",
                    },
                )
            return httpx.Response(404)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            services = SimpleNamespace(
                client=client,
                settings=Settings(
                    server_control_url="http://127.0.0.1:45123",
                    server_control_token="secret-token",
                    app_version="0.1.0",
                ),
            )
            updates = UpdateClient(services)

            checked = await updates.check()
            assert checked.update_available is True
            assert checked.latest_version == "0.1.20261126"
            assert checked.can_self_update is True
            assert checked.desktop.connected is True

            staged = await updates.stage(target_version="0.1.20261126")
            assert staged.server_staged is True

            applied = await updates.apply(target_version="0.1.20261126")
            assert applied.server_restarting is True

        assert seen_auth == [
            "Bearer secret-token",
            "Bearer secret-token",
            "Bearer secret-token",
        ]
        assert seen_paths == [
            "GET /updates",
            "POST /updates/stage",
            "POST /updates/apply",
        ]

    asyncio.run(run())


def test_update_client_github_fallback_reports_manual_server_and_supports_desktop():
    async def run():
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/repos/ghalrym/Kadan/releases/latest"
            return httpx.Response(
                200,
                json={
                    "tag_name": "v0.1.20261126",
                    "html_url": "https://github.com/ghalrym/Kadan/releases/tag/v0.1.20261126",
                    "body": "Fresh nightly build",
                    "published_at": "2026-11-26T06:00:00Z",
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            services = SimpleNamespace(
                client=client,
                settings=Settings(
                    app_version="0.1.0",
                    update_api_url="https://api.github.com/repos/ghalrym/Kadan/releases/latest",
                ),
            )
            updates = UpdateClient(services)

            # Without a desktop client, server fallback cannot self-update.
            checked = await updates.check()
            assert checked.update_available is True
            assert checked.can_self_update is False
            assert checked.install_kind == "manual"
            with pytest.raises(ServiceError) as exc:
                await updates.stage()
            assert exc.value.code == "self_update_unsupported"

            # With a connected self-updatable desktop client, desktop update can still be signaled.
            await updates.record_desktop_heartbeat("0.1.0", True)
            staged = await updates.stage()
            assert staged.server_staged is False
            assert staged.desktop_update_available is True

            applied = await updates.apply(target_version=staged.target_version)
            assert applied.desktop_update_requested is True
            assert applied.pending_desktop_update is not None
            req_id = applied.pending_desktop_update.request_id
            assert (await updates.status()).pending_desktop_update is not None
            assert await updates.ack_desktop_update(req_id) is True
            assert (await updates.status()).pending_desktop_update is None

    asyncio.run(run())


def test_updates_routes_track_desktop_headers_and_health_version():
    app = create_app(Settings(app_version="0.1.20261120"))
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["version"] == "0.1.20261120"

        status = client.get(
            "/api/updates/status",
            headers={
                "X-Kadan-Desktop-Version": "0.1.20261119",
                "X-Kadan-Desktop-Can-Update": "true",
            },
        )
        assert status.status_code == 200
        body = status.json()
        assert body["current_version"] == "0.1.20261120"
        assert body["desktop"]["connected"] is True
        assert body["desktop"]["current_version"] == "0.1.20261119"
        assert body["desktop"]["can_self_update"] is True


class _FakeUpdates:
    def __init__(self, events: list[str], *, stage_error: ServiceError | None = None):
        self.events = events
        self.stage_error = stage_error

    async def check(self) -> UpdateCheckResult:
        self.events.append("check")
        return UpdateCheckResult(
            current_version="0.1.0",
            latest_version="0.1.20261126",
            update_available=True,
            release_tag="v0.1.20261126",
            can_self_update=True,
            install_kind="appimage",
            desktop=DesktopClientStatus(
                connected=True,
                current_version="0.1.0",
                update_available=True,
                can_self_update=True,
            ),
        )

    async def stage(self, target_version: str | None = None) -> StageUpdateResponse:
        self.events.append(f"stage:{target_version}")
        if self.stage_error is not None:
            raise self.stage_error
        return StageUpdateResponse(
            ok=True,
            target_version=target_version or "0.1.20261126",
            server_staged=True,
            desktop_update_available=True,
            message="Staged",
        )

    async def apply(
        self, target_version: str | None = None, include_desktop: bool = True
    ) -> ApplyUpdateResponse:
        self.events.append(f"apply:{target_version}:{include_desktop}")
        return ApplyUpdateResponse(
            ok=True,
            target_version=target_version or "0.1.20261126",
            server_restarting=True,
            desktop_update_requested=include_desktop,
            message="Restarting",
        )


class _UpdateRunBoundary:
    def __init__(self, events: list[str], *, stage_error: ServiceError | None = None):
        now = datetime.now(UTC)
        self.events = events
        self.updates = _FakeUpdates(events, stage_error=stage_error)
        self.settings = SimpleNamespace(hikari_url="http://unused")
        self.client = None
        user = Message(role="user", content="Please update Kadan")
        self.conversation = Resource(
            id=uuid4(),
            agent_key="test",
            kind="conversation",
            title="Chat",
            messages=[user],
            version=1,
            created_at=now,
            updated_at=now,
        )
        self.controller = AgentRuns(self)
        self.current = AgentRun(
            id=uuid4(),
            agent_key="test",
            conversation_id=self.conversation.id,
            trigger_message_id=user.id,
            status="running",
            owner_id=self.controller.owner,
            lease_expires_at=now + timedelta(seconds=60),
            version=1,
            created_at=now,
            updated_at=now,
        )
        self.delivered_messages: list[str] = []

    async def json_request(self, service, method, path, *, timeout=15, **kwargs):
        assert service == "Tsuchi"
        if method == "GET" and path.endswith(str(self.current.id)):
            return self.current.model_dump(mode="json")
        if method == "POST" and path.endswith("/messages"):
            msg = kwargs["json"]["message"]["content"]
            self.delivered_messages.append(msg)
            self.events.append("deliver_message")
            return self.current.model_dump(mode="json")
        if method == "PUT" and path.endswith(str(self.current.id)):
            payload = kwargs["json"]
            self.events.append(f"save_run:{payload['status']}")
            merged = self.current.model_dump(mode="json")
            merged.update(
                {
                    "status": payload["status"],
                    "owner_id": payload["owner_id"],
                    "lease_expires_at": payload["lease_expires_at"],
                    "transcript": payload["transcript"],
                    "version": self.current.version + 1,
                }
            )
            self.current = AgentRun.model_validate(merged)
            return self.current.model_dump(mode="json")
        raise AssertionError(f"Unexpected request: {method} {path}")


def test_install_app_update_stages_delivers_completes_then_applies():
    events: list[str] = []
    boundary = _UpdateRunBoundary(events)
    cmd = ToolCommand(
        tool="install_app_update",
        arguments={
            "announcement": "Updating Kadan to v0.1.20261126 now — I'll be right back!",
            "target_version": "0.1.20261126",
        },
    )
    boundary.current = boundary.current.model_copy(
        update={"transcript": [RunCall(call_id="c1", command=cmd)]}
    )

    out = asyncio.run(
        boundary.controller.invoke_tool(
            RunState(run=boundary.current, command=cmd, replayed=False)
        )
    )
    assert out["run"].status == "completed"
    assert boundary.delivered_messages == [
        "Updating Kadan to v0.1.20261126 now — I'll be right back!"
    ]
    # Crucial invariant: apply must run strictly AFTER the completed run is durably saved.
    assert events == [
        "stage:0.1.20261126",
        "deliver_message",
        "save_run:completed",
        "apply:0.1.20261126:True",
    ]


def test_install_app_update_does_not_restart_when_staging_fails():
    events: list[str] = []
    boundary = _UpdateRunBoundary(
        events, stage_error=ServiceError("Already up to date", 422)
    )
    cmd = ToolCommand(
        tool="install_app_update",
        arguments={"announcement": "Updating now"},
    )
    boundary.current = boundary.current.model_copy(
        update={"transcript": [RunCall(call_id="c1", command=cmd)]}
    )

    out = asyncio.run(
        boundary.controller.invoke_tool(
            RunState(run=boundary.current, command=cmd, replayed=False)
        )
    )
    assert out["run"].status == "running"
    assert boundary.delivered_messages == []
    assert events == ["stage:None", "save_run:running"]
    assert out["run"].transcript[-1].data["updated"] is False


def test_install_app_update_never_replays_after_restart():
    events: list[str] = []
    boundary = _UpdateRunBoundary(events)
    cmd = ToolCommand(
        tool="install_app_update",
        arguments={"announcement": "Updating now"},
    )
    boundary.current = boundary.current.model_copy(
        update={"transcript": [RunCall(call_id="c1", command=cmd)]}
    )

    out = asyncio.run(
        boundary.controller.invoke_tool(
            RunState(run=boundary.current, command=cmd, replayed=True)
        )
    )
    assert out["run"].status == "running"
    assert events == ["save_run:running"]
    assert out["run"].transcript[-1].data["replayed"] is True


def test_check_app_update_tool_returns_release_info():
    events: list[str] = []
    boundary = _UpdateRunBoundary(events)
    cmd = ToolCommand(tool="check_app_update", arguments={})
    boundary.current = boundary.current.model_copy(
        update={"transcript": [RunCall(call_id="c1", command=cmd)]}
    )

    out = asyncio.run(
        boundary.controller.invoke_tool(
            RunState(run=boundary.current, command=cmd, replayed=False)
        )
    )
    data = out["run"].transcript[-1].data
    assert data["update_available"] is True
    assert data["latest_version"] == "0.1.20261126"
    assert data["can_self_update"] is True
    assert data["desktop"]["connected"] is True
