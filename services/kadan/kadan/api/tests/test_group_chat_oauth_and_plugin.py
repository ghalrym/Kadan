"""Unit tests for Group Chat OAuth 2.1 (ChatGPT sign-in) and ChatGPT plugin ZIP packaging."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit
from uuid import UUID, uuid4

import httpx
from fastapi.testclient import TestClient

from kadan.api.config import Settings
from kadan.api.group_chat_client import hash_group_chat_token
from kadan.api.server import create_app
from kadan.api.services import LocalServices


class FakeTsuchiOAuth:
    """In-memory Tsuchi Group Chat + OAuth simulator for Kadan API unit tests."""

    def __init__(self):
        now = datetime.now(UTC).isoformat()
        self.room_id = UUID("00000000-0000-0000-0000-000000000001")
        self.room: dict[str, Any] = {
            "id": str(self.room_id),
            "title": "Group Chat",
            "consecutive_agent_turns": 0,
            "max_consecutive_agent_turns": 6,
            "version": 1,
            "created_at": now,
            "updated_at": now,
        }
        self.agents = [{"key": "kadan", "name": "Kadan", "created_at": now}]
        self.participants: dict[str, dict[str, Any]] = {}
        self.tokens_by_hash: dict[str, str] = {}
        self.oauth_clients: dict[str, dict[str, Any]] = {}
        self.oauth_codes: dict[str, dict[str, Any]] = {}
        self.oauth_tokens: dict[str, dict[str, Any]] = {}

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        now = datetime.now(UTC).isoformat()

        if path == "/api/agents" and method == "GET":
            return httpx.Response(200, json=self.agents)
        if path == "/api/group-chat/rooms/default" and method == "GET":
            return httpx.Response(200, json=self.room)
        if path == f"/api/group-chat/rooms/{self.room_id}" and method == "GET":
            return httpx.Response(200, json=self.room)

        if (
            path == f"/api/group-chat/rooms/{self.room_id}/participants"
            and method == "GET"
        ):
            return httpx.Response(200, json=list(self.participants.values()))

        if (
            path == f"/api/group-chat/rooms/{self.room_id}/participants"
            and method == "POST"
        ):
            body = json.loads(request.content)
            if body["kind"] == "kadan_agent":
                for p in self.participants.values():
                    if p["kind"] == "kadan_agent":
                        return httpx.Response(201, json=p)
            pid = str(uuid4())
            handle = body.get("handle") or body["display_name"].lower().replace(
                " ", "-"
            )
            rec = {
                "id": pid,
                "room_id": str(self.room_id),
                "handle": handle,
                "display_name": body["display_name"],
                "kind": body["kind"],
                "agent_key": body.get("agent_key"),
                "description": body.get("description", ""),
                "enabled": True,
                "revoked_at": None,
                "last_seen_at": None,
                "cursor_seq": 0,
                "version": 1,
                "created_at": now,
                "updated_at": now,
            }
            self.participants[pid] = rec
            if body.get("token_hash"):
                self.tokens_by_hash[body["token_hash"]] = pid
            return httpx.Response(201, json=rec)

        if path == "/api/group-chat/participants/authenticate" and method == "POST":
            token_hash = json.loads(request.content)["token_hash"]
            pid = self.tokens_by_hash.get(token_hash)
            if pid is None:
                for tok in self.oauth_tokens.values():
                    if tok["access_token_hash"] == token_hash:
                        if tok["revoked_at"] is not None:
                            return httpx.Response(
                                401,
                                json={
                                    "detail": "Revoked token",
                                    "code": "token_revoked",
                                },
                            )
                        pid = tok["participant_id"]
                        break
            if pid is None or pid not in self.participants:
                return httpx.Response(
                    401, json={"detail": "Invalid token", "code": "invalid_token"}
                )
            part = self.participants[pid]
            if part["revoked_at"] is not None:
                return httpx.Response(
                    403, json={"detail": "Revoked", "code": "participant_revoked"}
                )
            if not part["enabled"]:
                return httpx.Response(
                    403, json={"detail": "Paused", "code": "participant_paused"}
                )
            part["last_seen_at"] = now
            return httpx.Response(200, json=part)

        if path == "/api/group-chat/oauth/clients" and method == "POST":
            body = json.loads(request.content)
            rec = {**body, "created_at": now}
            self.oauth_clients[body["client_id"]] = rec
            return httpx.Response(201, json=rec)

        if path.startswith("/api/group-chat/oauth/clients/") and method == "GET":
            client_id = path[len("/api/group-chat/oauth/clients/") :]
            rec = self.oauth_clients.get(client_id)
            if rec is None:
                return httpx.Response(
                    404, json={"detail": "Client not found", "code": "invalid_client"}
                )
            return httpx.Response(200, json=rec)

        if path == "/api/group-chat/oauth/codes" and method == "POST":
            body = json.loads(request.content)
            rec = {**body, "consumed_at": None, "created_at": now}
            self.oauth_codes[body["code_hash"]] = rec
            return httpx.Response(201, json=rec)

        if path == "/api/group-chat/oauth/codes/consume" and method == "POST":
            code_hash = json.loads(request.content)["code_hash"]
            rec = self.oauth_codes.get(code_hash)
            if rec is None or rec["consumed_at"] is not None:
                return httpx.Response(
                    400,
                    json={"detail": "Invalid or used code", "code": "invalid_grant"},
                )
            rec["consumed_at"] = now
            return httpx.Response(200, json=rec)

        if path == "/api/group-chat/oauth/tokens" and method == "POST":
            body = json.loads(request.content)
            tid = str(uuid4())
            rec = {
                "id": tid,
                "room_id": body["room_id"],
                "participant_id": body["participant_id"],
                "client_id": body["client_id"],
                "scope": body["scope"],
                "resource": body.get("resource"),
                "access_token_hash": body["access_token_hash"],
                "refresh_token_hash": body["refresh_token_hash"],
                "access_expires_at": body["access_expires_at"],
                "refresh_expires_at": body["refresh_expires_at"],
                "revoked_at": None,
                "created_at": now,
                "updated_at": now,
            }
            self.oauth_tokens[tid] = rec
            return httpx.Response(
                201,
                json={
                    k: v
                    for k, v in rec.items()
                    if k not in {"access_token_hash", "refresh_token_hash"}
                },
            )

        if path == "/api/group-chat/oauth/tokens/refresh" and method == "POST":
            body = json.loads(request.content)
            match = next(
                (
                    tok
                    for tok in self.oauth_tokens.values()
                    if tok["refresh_token_hash"] == body["refresh_token_hash"]
                    and tok["revoked_at"] is None
                ),
                None,
            )
            if match is None:
                return httpx.Response(
                    400,
                    json={"detail": "Invalid refresh token", "code": "invalid_grant"},
                )
            if body.get("client_id") and body["client_id"] != match["client_id"]:
                return httpx.Response(
                    400, json={"detail": "client_id mismatch", "code": "invalid_grant"}
                )
            match["access_token_hash"] = body["new_access_token_hash"]
            match["refresh_token_hash"] = body["new_refresh_token_hash"]
            match["access_expires_at"] = body["access_expires_at"]
            match["refresh_expires_at"] = body["refresh_expires_at"]
            match["updated_at"] = now
            return httpx.Response(
                200,
                json={
                    k: v
                    for k, v in match.items()
                    if k not in {"access_token_hash", "refresh_token_hash"}
                },
            )

        return httpx.Response(404, json={"detail": f"Unhandled {method} {path}"})


def _pkce_pair() -> tuple[str, str]:
    verifier = "a" * 64
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    return verifier, challenge


def _make_client(tsuchi: FakeTsuchiOAuth, *, root_path: str = "") -> TestClient:
    app = create_app(
        Settings(tsuchi_url="http://tsuchi.test", hikari_url="http://hikari.test")
    )
    client = TestClient(app, root_path=root_path)
    client.__enter__()
    mock_http = httpx.AsyncClient(transport=httpx.MockTransport(tsuchi.handle))
    app.state.services = LocalServices(
        mock_http,
        Settings(tsuchi_url="http://tsuchi.test", hikari_url="http://hikari.test"),
    )
    return client


def test_group_chat_oauth_discovery_dcr_consent_pkce_mcp_and_refresh():
    tsuchi = FakeTsuchiOAuth()
    client = _make_client(tsuchi, root_path="/kadan")
    try:
        fwd = {
            "x-forwarded-proto": "https",
            "x-forwarded-host": "kadan.example.com",
        }

        # 1. Unauthenticated MCP call returns 401 with RFC 9728 WWW-Authenticate header.
        unauth = client.post(
            "/mcp/group-chat",
            headers=fwd,
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        )
        assert unauth.status_code == 401
        www_auth = unauth.headers["www-authenticate"]
        assert (
            'resource_metadata="https://kadan.example.com/.well-known/oauth-protected-resource/kadan/mcp/group-chat"'
            in www_auth
        )
        assert 'scope="group_chat"' in www_auth

        # 2. Protected resource & authorization server discovery documents match.
        prm = client.get(
            "/.well-known/oauth-protected-resource/kadan/mcp/group-chat", headers=fwd
        )
        assert prm.status_code == 200
        prm_data = prm.json()
        assert prm_data["resource"] == "https://kadan.example.com/kadan/mcp/group-chat"
        assert prm_data["authorization_servers"] == [
            "https://kadan.example.com/kadan/api/group-chat/oauth"
        ]

        asm = client.get(
            "/.well-known/oauth-authorization-server/kadan/api/group-chat/oauth",
            headers=fwd,
        )
        assert asm.status_code == 200
        asm_data = asm.json()
        assert (
            asm_data["issuer"] == "https://kadan.example.com/kadan/api/group-chat/oauth"
        )
        assert (
            asm_data["authorization_endpoint"]
            == "https://kadan.example.com/kadan/api/group-chat/oauth/authorize"
        )
        assert (
            asm_data["token_endpoint"]
            == "https://kadan.example.com/kadan/api/group-chat/oauth/token"
        )
        assert (
            asm_data["registration_endpoint"]
            == "https://kadan.example.com/kadan/api/group-chat/oauth/register"
        )
        assert asm_data["code_challenge_methods_supported"] == ["S256"]

        # 3. Dynamic Client Registration (RFC 7591).
        redirect_uri = "https://chatgpt.com/connector_platform_oauth_redirect"
        reg = client.post(
            "/api/group-chat/oauth/register",
            json={"client_name": "ChatGPT", "redirect_uris": [redirect_uri]},
        )
        assert reg.status_code == 201
        client_id = reg.json()["client_id"]
        assert client_id.startswith("kdn_gcl_")

        # 4. Consent page renders cleanly in plain language.
        verifier, challenge = _pkce_pair()
        auth_page = client.get(
            "/api/group-chat/oauth/authorize",
            headers=fwd,
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": "xyz-state",
                "scope": "group_chat",
                "resource": "https://kadan.example.com/kadan/mcp/group-chat",
            },
        )
        assert auth_page.status_code == 200
        assert "Connect ChatGPT" in auth_page.text
        assert "Create a new external agent" in auth_page.text

        # 5. User approves and creates "Dot" right from the consent screen.
        approved = client.post(
            "/api/group-chat/oauth/authorize",
            headers=fwd,
            follow_redirects=False,
            data={
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": "xyz-state",
                "scope": "group_chat",
                "resource": "https://kadan.example.com/kadan/mcp/group-chat",
                "participant_mode": "new",
                "display_name": "Dot",
                "description": "ChatGPT teammate",
                "decision": "approve",
            },
        )
        assert approved.status_code == 303
        loc = urlsplit(approved.headers["location"])
        assert f"{loc.scheme}://{loc.netloc}{loc.path}" == redirect_uri
        qs = parse_qs(loc.query)
        assert qs["state"] == ["xyz-state"]
        auth_code = qs["code"][0]

        # 6. Wrong PKCE verifier fails (and consumes the single-use code).
        bad_code = client.post(
            "/api/group-chat/oauth/authorize",
            headers=fwd,
            follow_redirects=False,
            data={
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": "s2",
                "scope": "group_chat",
                "participant_mode": "existing:"
                + next(
                    p["id"]
                    for p in tsuchi.participants.values()
                    if p["display_name"] == "Dot"
                ),
                "decision": "approve",
            },
        )
        bad_auth_code = parse_qs(urlsplit(bad_code.headers["location"]).query)["code"][
            0
        ]
        wrong_pkce = client.post(
            "/api/group-chat/oauth/token",
            headers=fwd,
            data={
                "grant_type": "authorization_code",
                "code": bad_auth_code,
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "code_verifier": "b" * 64,
            },
        )
        assert wrong_pkce.status_code == 400
        assert wrong_pkce.json()["error"] == "invalid_grant"

        # 7. Valid PKCE exchange mints access + refresh tokens.
        token_res = client.post(
            "/api/group-chat/oauth/token",
            headers=fwd,
            data={
                "grant_type": "authorization_code",
                "code": auth_code,
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "code_verifier": verifier,
                "resource": "https://kadan.example.com/kadan/mcp/group-chat",
            },
        )
        assert token_res.status_code == 200
        tokens = token_res.json()
        assert tokens["token_type"] == "Bearer"
        assert tokens["access_token"].startswith("kdn_gco_")
        assert tokens["refresh_token"].startswith("kdn_gcr_")

        # 8. OAuth access token authenticates MCP calls as Dot.
        mcp_res = client.post(
            "/mcp/group-chat",
            headers={
                **fwd,
                "Authorization": f"Bearer {tokens['access_token']}",
            },
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "group_chat_list_participants",
                    "arguments": {},
                },
            },
        )
        assert mcp_res.status_code == 200
        parts = mcp_res.json()["result"]["structuredContent"]["participants"]
        me = next(p for p in parts if p.get("is_me"))
        assert me["display_name"] == "Dot"

        # 9. Refresh token rotates credentials cleanly.
        refreshed = client.post(
            "/api/group-chat/oauth/token",
            headers=fwd,
            data={
                "grant_type": "refresh_token",
                "refresh_token": tokens["refresh_token"],
                "client_id": client_id,
            },
        )
        assert refreshed.status_code == 200
        next_tokens = refreshed.json()
        assert next_tokens["access_token"] != tokens["access_token"]
        # Old access token now returns 401 with WWW-Authenticate.
        stale = client.post(
            "/mcp/group-chat",
            headers={**fwd, "Authorization": f"Bearer {tokens['access_token']}"},
            json={"jsonrpc": "2.0", "id": 3, "method": "ping"},
        )
        assert stale.status_code == 401
        assert 'error="invalid_token"' in stale.headers["www-authenticate"]
        # New access token succeeds.
        fresh = client.post(
            "/mcp/group-chat",
            headers={**fwd, "Authorization": f"Bearer {next_tokens['access_token']}"},
            json={"jsonrpc": "2.0", "id": 4, "method": "ping"},
        )
        assert fresh.status_code == 200

        # 10. Denying consent redirects with error=access_denied, and CIMD client_id works.
        cimd_client = "https://chatgpt.com/oauth/client-metadata.json"
        denied = client.post(
            "/api/group-chat/oauth/authorize",
            headers=fwd,
            follow_redirects=False,
            data={
                "client_id": cimd_client,
                "redirect_uri": redirect_uri,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": "deny-1",
                "decision": "deny",
            },
        )
        assert denied.status_code == 303
        deny_qs = parse_qs(urlsplit(denied.headers["location"]).query)
        assert deny_qs["error"] == ["access_denied"]
        assert deny_qs["state"] == ["deny-1"]
    finally:
        client.__exit__(None, None, None)


def test_chatgpt_plugin_zip_download_and_repo_bundle_schema():
    tsuchi = FakeTsuchiOAuth()
    client = _make_client(tsuchi, root_path="/kadan")
    try:
        res = client.get(
            "/api/group-chat/chatgpt-plugin.zip",
            headers={
                "x-forwarded-proto": "https",
                "x-forwarded-host": "my-kadan.example.com",
            },
        )
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/zip"
        assert (
            'filename="kadan-group-chat-plugin.zip"'
            in res.headers["content-disposition"]
        )

        with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
            names = set(zf.namelist())
            assert names == {
                "plugin.json",
                "mcp.json",
                "skills/kadan-group-chat/SKILL.md",
            }
            manifest = json.loads(zf.read("plugin.json").decode("utf-8"))
            assert (
                manifest["$schema"]
                == "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
            )
            assert manifest["name"] == "kadan-group-chat"
            assert manifest["skills"] == "./skills/"
            assert manifest["mcpServers"] == "./mcp.json"

            mcp_cfg = json.loads(zf.read("mcp.json").decode("utf-8"))
            assert (
                mcp_cfg["$schema"]
                == "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"
            )
            server = mcp_cfg["mcpServers"]["kadan-group-chat"]
            assert server == {
                "type": "streamable-http",
                "url": "https://my-kadan.example.com/kadan/mcp/group-chat",
            }

            skill = zf.read("skills/kadan-group-chat/SKILL.md").decode("utf-8")
            assert "name: kadan-group-chat" in skill
            assert "group_chat_send_message" in skill
            assert "group_chat.turn_requested" in skill

        # Static repo bundle matches the same schema and skill content.
        repo_root = Path(__file__).resolve().parents[5]
        plugin_dir = repo_root / "plugins" / "kadan-group-chat"
        assert (plugin_dir / "plugin.json").is_file()
        assert (plugin_dir / "mcp.json").is_file()
        assert (plugin_dir / "skills" / "kadan-group-chat" / "SKILL.md").is_file()
        static_manifest = json.loads(
            (plugin_dir / "plugin.json").read_text(encoding="utf-8")
        )
        assert static_manifest["name"] == "kadan-group-chat"
        assert hash_group_chat_token("x")  # sanity import check
    finally:
        client.__exit__(None, None, None)
