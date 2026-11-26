"""Unit tests for Kadan's Remote MCP client, OAuth 2.1 flow, and routes."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlsplit
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from kadan.api.config import Settings
from kadan.api.mcp_client import McpClient, transport_url
from kadan.api.pydantic_models.mcp import McpServerCreate
from kadan.api.server import create_app
from kadan.api.services import LocalServices


class FakeTsuchiAndMcp:
    """In-memory mock for both Tsuchi MCP storage and a Streamable HTTP MCP server."""

    def __init__(
        self,
        *,
        require_oauth: bool = False,
        use_sse: bool = False,
        strict_redirect_uris: bool = False,
    ):
        self.servers: dict[str, dict] = {}
        self.oauth_states: dict[str, dict] = {}
        self.require_oauth = require_oauth
        self.use_sse = use_sse
        self.strict_redirect_uris = strict_redirect_uris
        self.registrations: list[dict] = []
        self.valid_token = "tok-initial"
        self.refreshed_token = "tok-refreshed"
        self.tools = [
            {
                "name": "t3_project_list",
                "description": "List projects",
                "inputSchema": {"type": "object", "properties": {}},
                "annotations": {"readOnlyHint": True},
            },
            {
                "name": "t3_thread_launch",
                "description": "Launch a thread",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "clientRequestId": {"type": "string"},
                    },
                    "required": ["title"],
                },
            },
            {
                "name": "pull_request_list",
                "description": "List pull requests",
                "inputSchema": {"type": "object", "properties": {}},
                "annotations": {"readOnlyHint": True},
            },
        ]
        self.calls: list[tuple[str, dict]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        path = request.url.path

        # --- Tsuchi mock routes ---
        if request.url.host == "tsuchi.test":
            if path == "/api/mcp-servers" and request.method == "GET":
                return httpx.Response(200, json=list(self.servers.values()))
            if path == "/api/mcp-servers" and request.method == "POST":
                body = json.loads(request.content)
                now = datetime.now(UTC).isoformat()
                record = {
                    **body,
                    "id": str(uuid4()),
                    "version": 1,
                    "created_at": now,
                    "updated_at": now,
                }
                self.servers[record["id"]] = record
                return httpx.Response(201, json=record)
            if path.startswith("/api/mcp-servers/"):
                sid = path.rsplit("/", 1)[-1]
                if request.method == "GET":
                    if sid not in self.servers:
                        return httpx.Response(404, json={"detail": "Not found"})
                    return httpx.Response(200, json=self.servers[sid])
                if request.method == "PUT":
                    body = json.loads(request.content)
                    cur = self.servers[sid]
                    if body.pop("expected_version") != cur["version"]:
                        return httpx.Response(409, json={"detail": "Conflict"})
                    updated = {
                        **cur,
                        **body,
                        "version": cur["version"] + 1,
                        "updated_at": datetime.now(UTC).isoformat(),
                    }
                    self.servers[sid] = updated
                    return httpx.Response(200, json=updated)
                if request.method == "DELETE":
                    self.servers.pop(sid, None)
                    return httpx.Response(204)
            if path == "/api/mcp-oauth-states" and request.method == "POST":
                body = json.loads(request.content)
                body.setdefault("created_at", datetime.now(UTC).isoformat())
                self.oauth_states[body["state"]] = body
                return httpx.Response(201, json=body)
            if path.startswith("/api/mcp-oauth-states/") and path.endswith("/consume"):
                state = path.split("/")[-2]
                saved = self.oauth_states.pop(state, None)
                if not saved:
                    return httpx.Response(404, json={"detail": "Not found"})
                return httpx.Response(200, json=saved)
            return httpx.Response(404, json={"detail": f"Unhandled {url}"})

        # --- Remote MCP + OAuth server mock ---
        if path == "/.well-known/oauth-protected-resource/mcp":
            return httpx.Response(
                200,
                json={
                    "authorization_servers": ["https://t3.example"],
                    "scopes_supported": ["mcp:tools"],
                },
            )
        if path == "/.well-known/oauth-authorization-server":
            return httpx.Response(
                200,
                json={
                    "authorization_endpoint": "https://t3.example/oauth/authorize",
                    "token_endpoint": "https://t3.example/oauth/token",
                    "registration_endpoint": "https://t3.example/oauth/register",
                    "grant_types_supported": ["authorization_code"],
                },
            )
        if path == "/oauth/register":
            body = json.loads(request.content)
            self.registrations.append(body)
            uris = body.get("redirect_uris") or []
            if self.strict_redirect_uris:
                for uri in uris:
                    parsed = urlsplit(uri)
                    allowed = parsed.scheme == "https" or (
                        parsed.scheme == "http"
                        and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
                    )
                    if not allowed:
                        return httpx.Response(
                            400,
                            json={
                                "error": "invalid_redirect_uri",
                                "error_description": "Redirect URIs must be https, or http on localhost, 127.0.0.1 or [::1].",
                            },
                        )
            return httpx.Response(
                201, json={"client_id": f"dyn-client-{len(self.registrations)}"}
            )
        if path == "/oauth/token":
            body_text = request.content.decode()
            parsed = parse_qs(body_text)
            if parsed.get("grant_type") == ["refresh_token"]:
                self.valid_token = self.refreshed_token
                return httpx.Response(
                    200,
                    json={
                        "access_token": self.refreshed_token,
                        "refresh_token": "ref-2",
                        "expires_in": 3600,
                    },
                )
            return httpx.Response(
                200,
                json={
                    "access_token": self.valid_token,
                    "refresh_token": "ref-1",
                    "expires_in": 3600,
                },
            )
        if path == "/mcp":
            if self.require_oauth:
                auth = request.headers.get("authorization", "")
                if auth != f"Bearer {self.valid_token}":
                    return httpx.Response(
                        401,
                        headers={
                            "WWW-Authenticate": 'Bearer resource_metadata="https://t3.example/.well-known/oauth-protected-resource/mcp"'
                        },
                    )
            rpc = json.loads(request.content)
            method = rpc.get("method")
            if method == "initialize":
                payload = {"jsonrpc": "2.0", "id": rpc["id"], "result": {"protocolVersion": "2026-07-28"}}
            elif method == "notifications/initialized":
                if request.headers.get("mcp-protocol-version") != "2026-07-28":
                    return httpx.Response(
                        400,
                        json={"jsonrpc": "2.0", "id": None, "error": {"code": -32022, "message": "Unsupported protocol version"}},
                    )
                return httpx.Response(202)
            elif method == "tools/list":
                if request.headers.get("mcp-protocol-version") != "2026-07-28":
                    return httpx.Response(
                        400,
                        json={"jsonrpc": "2.0", "id": rpc["id"], "error": {"code": -32022, "message": "Unsupported protocol version"}},
                    )
                payload = {"jsonrpc": "2.0", "id": rpc["id"], "result": {"tools": self.tools}}
            elif method == "tools/call":
                if request.headers.get("mcp-protocol-version") != "2026-07-28":
                    return httpx.Response(
                        400,
                        json={"jsonrpc": "2.0", "id": rpc["id"], "error": {"code": -32022, "message": "Unsupported protocol version"}},
                    )
                name = rpc["params"]["name"]
                args = rpc["params"].get("arguments", {})
                self.calls.append((name, args))
                payload = {
                    "jsonrpc": "2.0",
                    "id": rpc["id"],
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps({"ok": True, "called": name, "args": args}),
                            }
                        ]
                    },
                }
            else:
                payload = {"jsonrpc": "2.0", "id": rpc["id"], "error": {"message": "Unknown"}}

            if self.use_sse:
                sse = f"event: message\ndata: {json.dumps(payload)}\n\n"
                return httpx.Response(
                    200,
                    text=sse,
                    headers={
                        "content-type": "text/event-stream",
                        "mcp-session-id": "sess-1",
                    },
                )
            return httpx.Response(
                200,
                json=payload,
                headers={"mcp-session-id": "sess-1"},
            )
        return httpx.Response(404)


def make_services(mock: FakeTsuchiAndMcp) -> LocalServices:
    client = httpx.AsyncClient(transport=httpx.MockTransport(mock.handler))
    settings = Settings(tsuchi_url="http://tsuchi.test", hikari_url="http://hikari.test")
    return LocalServices(client, settings)


def test_transport_url_rewrites_loopback_only_in_docker(monkeypatch):
    monkeypatch.setenv("KADAN_IN_DOCKER", "1")
    assert transport_url("http://127.0.0.1:3773/mcp") == "http://host.docker.internal:3773/mcp"
    assert transport_url("http://localhost:3773/mcp") == "http://host.docker.internal:3773/mcp"
    assert transport_url("https://box.tailnet.ts.net/mcp") == "https://box.tailnet.ts.net/mcp"


def test_t3_preset_oauth_connect_callback_and_tool_sync():
    mock = FakeTsuchiAndMcp(require_oauth=True, use_sse=True)

    async def scenario():
        services = make_services(mock)
        mcp = McpClient(services)
        created = await mcp.create_server(
            McpServerCreate(
                name="T3 Code",
                url="https://t3.example/mcp",
                preset="t3_code",
                auth_type="oauth",
            )
        )
        assert created.slug == "t3-code"
        assert created.app_url == "https://t3.example"

        started = await mcp.connect(
            created.id, "http://localhost:8630/api/mcp/oauth/callback"
        )
        assert started.authorize_url is not None
        query = parse_qs(urlsplit(started.authorize_url).query)
        assert query["client_id"] == ["dyn-client-1"]
        assert query["code_challenge_method"] == ["S256"]
        state = query["state"][0]

        completed = await mcp.complete_oauth(state, "auth-code-99")
        assert completed.status == "connected"
        assert len(completed.tools) == 3
        # Curated T3 preset pins core tools, not pull_request_list
        assert completed.pinned_tools == ["t3_project_list", "t3_thread_launch"]
        # Public representation redacts credentials
        pub = completed.to_public().model_dump(mode="json")
        assert pub["has_credentials"] is True
        assert "access_token" not in json.dumps(pub)

        # Tool call with automatic refresh when token rotates on server
        mock.valid_token = mock.refreshed_token
        result = await mcp.call_tool(
            completed, "t3_thread_launch", {"title": "Fix bug", "clientRequestId": "abc"}
        )
        assert result == {
            "ok": True,
            "called": "t3_thread_launch",
            "args": {"title": "Fix bug", "clientRequestId": "abc"},
        }
        await services.client.aclose()

    asyncio.run(scenario())


def test_mcp_routes_expose_redacted_servers_and_oauth_callback():
    mock = FakeTsuchiAndMcp(require_oauth=False)
    app = create_app(Settings(tsuchi_url="http://tsuchi.test", hikari_url="http://hikari.test"))

    with TestClient(app) as client:
        app.state.services = make_services(mock)
        created = client.post(
            "/api/mcp/servers",
            json={
                "name": "Local MCP",
                "url": "https://t3.example/mcp",
                "preset": "custom",
                "auth_type": "bearer",
                "bearer_token": "super-secret-value",
            },
        )
        assert created.status_code == 201
        body = created.json()
        assert body["has_credentials"] is True
        assert "super-secret-value" not in created.text

        connected = client.post(f"/api/mcp/servers/{body['id']}/connect")
        assert connected.status_code == 200
        assert connected.json()["authorize_url"] is None
        assert connected.json()["server"]["status"] == "connected"
        assert len(connected.json()["server"]["tools"]) == 3


def test_lan_http_dcr_400_falls_back_to_localhost_and_completes_from_pasted_url():
    mock = FakeTsuchiAndMcp(require_oauth=True, strict_redirect_uris=True)
    app = create_app(Settings(tsuchi_url="http://tsuchi.test", hikari_url="http://hikari.test"))

    with TestClient(app) as client:
        app.state.services = make_services(mock)
        created = client.post(
            "/api/mcp/servers",
            json={
                "name": "T3 Code",
                "url": "https://t3.example/mcp",
                "preset": "t3_code",
                "auth_type": "oauth",
            },
        )
        assert created.status_code == 201
        server_id = created.json()["id"]

        # Simulate browser accessing Kadan over LAN HTTP (e.g. http://192.168.0.19:8630).
        started = client.post(
            f"/api/mcp/servers/{server_id}/connect",
            headers={"Host": "192.168.0.19:8630", "Origin": "http://192.168.0.19:8630"},
        )
        assert started.status_code == 200
        payload = started.json()
        assert payload["manual_callback"] is True
        assert payload["authorize_url"] is not None

        # First DCR attempt used LAN IP and was rejected; fallback used localhost:8630.
        assert len(mock.registrations) == 2
        assert mock.registrations[0]["redirect_uris"] == [
            "http://192.168.0.19:8630/api/mcp/oauth/callback"
        ]
        assert mock.registrations[0]["grant_types"] == ["authorization_code"]
        assert mock.registrations[1]["redirect_uris"] == [
            "http://localhost:8630/api/mcp/oauth/callback"
        ]

        query = parse_qs(urlsplit(payload["authorize_url"]).query)
        assert query["redirect_uri"] == ["http://localhost:8630/api/mcp/oauth/callback"]
        state = query["state"][0]

        # Complete OAuth by pasting the localhost callback URL from another device.
        completed = client.post(
            "/api/mcp/oauth/complete",
            json={
                "callback_url": f"http://localhost:8630/api/mcp/oauth/callback?code=code-123&state={state}"
            },
        )
        assert completed.status_code == 200
        assert completed.json()["status"] == "connected"
        assert len(completed.json()["tools"]) == 3


def test_callback_url_prefers_https_origin_and_re_registers_when_origin_changes():
    mock = FakeTsuchiAndMcp(require_oauth=True, strict_redirect_uris=True)
    app = create_app(Settings(tsuchi_url="http://tsuchi.test", hikari_url="http://hikari.test"))

    with TestClient(app) as client:
        app.state.services = make_services(mock)
        created = client.post(
            "/api/mcp/servers",
            json={
                "name": "T3 Code",
                "url": "https://t3.example/mcp",
                "preset": "t3_code",
                "auth_type": "oauth",
            },
        )
        server_id = created.json()["id"]

        # Reverse proxy without X-Forwarded-Proto still sends browser Origin: https://...
        first = client.post(
            f"/api/mcp/servers/{server_id}/connect",
            headers={"Host": "kadan.example.com", "Origin": "https://kadan.example.com"},
        )
        assert first.status_code == 200
        assert first.json()["manual_callback"] is False
        assert mock.registrations[-1]["redirect_uris"] == [
            "https://kadan.example.com/api/mcp/oauth/callback"
        ]

        # Switching to localhost re-registers so client_id matches the new redirect_uri.
        second = client.post(
            f"/api/mcp/servers/{server_id}/connect",
            headers={"Host": "localhost:8630", "Origin": "http://localhost:8630"},
        )
        assert second.status_code == 200
        assert len(mock.registrations) == 2
        assert mock.registrations[-1]["redirect_uris"] == [
            "http://localhost:8630/api/mcp/oauth/callback"
        ]


def test_callback_url_keeps_the_mount_prefix():
    from starlette.requests import Request

    from kadan.api.routes.mcp import _callback_url

    def request(headers: dict[str, str]) -> Request:
        return Request(
            {
                "type": "http",
                "method": "POST",
                "scheme": "http",
                "path": "/kadan/api/mcp",
                "root_path": "/kadan",
                "server": ("api", 8000),
                "query_string": b"",
                "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
            }
        )

    assert _callback_url(request({"Origin": "http://localhost:8630"})) == (
        "http://localhost:8630/kadan/api/mcp/oauth/callback"
    )
    assert _callback_url(
        request({"Host": "kadan.example", "X-Forwarded-Proto": "https"})
    ) == ("https://kadan.example/kadan/api/mcp/oauth/callback")
