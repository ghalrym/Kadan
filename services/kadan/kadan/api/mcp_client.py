"""Streamable HTTP Remote MCP client with OAuth 2.1 (PKCE + DCR) and token refresh."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
from uuid import UUID

import httpx

from .config import ServiceError
from .mcp_presets import default_pinned_tools, derive_app_url, derive_slug
from .pydantic_models.mcp import (
    McpAuthState,
    McpConnectResponse,
    McpOAuthStateRecord,
    McpServerCreate,
    McpServerPatch,
    McpServerRecord,
    McpToolInfo,
)

if TYPE_CHECKING:
    from .services import LocalServices

logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2025-03-26"
OAUTH_STATE_TTL_MINUTES = 15
MCP_TIMEOUT_SECONDS = 45.0
TOKEN_REFRESH_MARGIN = timedelta(seconds=60)


def _in_docker() -> bool:
    return Path("/.dockerenv").exists() or os.environ.get("KADAN_IN_DOCKER") == "1"


def transport_url(url: str) -> str:
    """Rewrite loopback URLs to host.docker.internal when running inside Docker."""
    if not _in_docker():
        return url
    parsed = urlsplit(url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return url
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"host.docker.internal{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _origin(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _parse_www_authenticate(header: str) -> dict[str, str]:
    result: dict[str, str] = {}
    if not header:
        return result
    rest = header.strip()
    if " " in rest:
        _, rest = rest.split(" ", 1)
    for part in rest.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        result[k.strip().lower()] = v.strip().strip('"')
    return result


def _parse_mcp_body(response: httpx.Response) -> dict[str, Any]:
    """Parse either an application/json or text/event-stream MCP JSON-RPC response."""
    content_type = response.headers.get("content-type", "").lower()
    if "text/event-stream" in content_type:
        last_message: dict[str, Any] | None = None
        data_lines: list[str] = []
        for raw_line in response.text.splitlines():
            line = raw_line.strip("\r")
            if line == "":
                if data_lines:
                    payload = "\n".join(data_lines)
                    data_lines = []
                    try:
                        parsed = json.loads(payload)
                    except ValueError:
                        continue
                    if isinstance(parsed, dict) and (
                        "result" in parsed or "error" in parsed
                    ):
                        last_message = parsed
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
        if data_lines:
            try:
                parsed = json.loads("\n".join(data_lines))
                if isinstance(parsed, dict) and (
                    "result" in parsed or "error" in parsed
                ):
                    last_message = parsed
            except ValueError:
                pass
        if last_message is None:
            raise ServiceError(
                "The MCP server returned an empty event stream.", 502, "mcp_invalid"
            )
        return last_message
    try:
        parsed = response.json()
    except ValueError as error:
        raise ServiceError(
            "The MCP server returned invalid JSON.", 502, "mcp_invalid"
        ) from error
    if not isinstance(parsed, dict):
        raise ServiceError(
            "The MCP server returned an unexpected response.", 502, "mcp_invalid"
        )
    return parsed


def _tool_from_mcp(raw: dict[str, Any]) -> McpToolInfo | None:
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    description = raw.get("description")
    if not isinstance(description, str):
        description = ""
    schema = raw.get("inputSchema")
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}}
    annotations = raw.get("annotations")
    if not isinstance(annotations, dict):
        annotations = {}
    props = schema.get("properties")
    accepts_client_req = isinstance(props, dict) and "clientRequestId" in props
    return McpToolInfo(
        name=name.strip(),
        description=description[:4000],
        input_schema=schema,
        read_only=bool(annotations.get("readOnlyHint", False)),
        idempotent=bool(annotations.get("idempotentHint", False)),
        accepts_client_request_id=accepts_client_req,
    )


class McpClient:
    """Manages global Remote MCP servers in Tsuchi and talks Streamable HTTP MCP."""

    def __init__(self, services: LocalServices):
        self.services = services
        self._refresh_locks: dict[UUID, asyncio.Lock] = {}

    def _lock_for(self, server_id: UUID) -> asyncio.Lock:
        lock = self._refresh_locks.get(server_id)
        if lock is None:
            lock = asyncio.Lock()
            self._refresh_locks[server_id] = lock
        return lock

    # ------------------------------------------------------------------
    # Tsuchi CRUD wrappers
    # ------------------------------------------------------------------

    async def list_servers(self) -> list[McpServerRecord]:
        values = await self.services.json_request("Tsuchi", "GET", "/api/mcp-servers")
        return [McpServerRecord.model_validate(item) for item in values]

    async def get_server(self, server_id: UUID) -> McpServerRecord:
        value = await self.services.json_request(
            "Tsuchi", "GET", f"/api/mcp-servers/{server_id}"
        )
        return McpServerRecord.model_validate(value)

    async def create_server(self, payload: McpServerCreate) -> McpServerRecord:
        slug = payload.slug or derive_slug(payload.name, payload.preset)
        auth_state = McpAuthState(
            bearer_token=payload.bearer_token if payload.auth_type == "bearer" else None
        )
        body = {
            "slug": slug,
            "name": payload.name,
            "preset": payload.preset,
            "url": payload.url,
            "app_url": derive_app_url(payload.url, payload.preset),
            "auth_type": payload.auth_type,
            "auth_state": auth_state.model_dump(mode="json"),
            "enabled": payload.enabled,
            "status": "disconnected",
            "status_error": None,
            "pinned_tools": list(payload.pinned_tools or []),
            "tools": [],
            "tools_synced_at": None,
        }
        saved = await self.services.json_request(
            "Tsuchi", "POST", "/api/mcp-servers", json=body
        )
        return McpServerRecord.model_validate(saved)

    async def _save_server(self, server: McpServerRecord) -> McpServerRecord:
        payload = server.model_dump(
            mode="json", exclude={"id", "version", "created_at", "updated_at"}
        )
        payload["expected_version"] = server.version
        saved = await self.services.json_request(
            "Tsuchi", "PUT", f"/api/mcp-servers/{server.id}", json=payload
        )
        return McpServerRecord.model_validate(saved)

    async def patch_server(
        self, server_id: UUID, patch: McpServerPatch
    ) -> McpServerRecord:
        current = await self.get_server(server_id)
        if current.version != patch.expected_version:
            raise ServiceError(
                "This integration changed. Refresh before saving.",
                409,
                "version_conflict",
            )
        updates: dict[str, Any] = {}
        url_changed = False
        if patch.name is not None:
            updates["name"] = patch.name
        if patch.url is not None and patch.url != current.url:
            updates["url"] = patch.url
            updates["app_url"] = derive_app_url(patch.url, current.preset)
            url_changed = True
        if patch.auth_type is not None and patch.auth_type != current.auth_type:
            updates["auth_type"] = patch.auth_type
            url_changed = True
        if patch.enabled is not None:
            updates["enabled"] = patch.enabled
        if patch.pinned_tools is not None:
            updates["pinned_tools"] = list(dict.fromkeys(patch.pinned_tools))

        auth_state = current.auth_state.model_copy()
        if url_changed:
            auth_state = McpAuthState()
            updates["status"] = "disconnected"
            updates["status_error"] = None
            updates["tools"] = []
            updates["tools_synced_at"] = None
        if patch.bearer_token is not None:
            auth_state.bearer_token = patch.bearer_token.strip() or None
        updates["auth_state"] = auth_state

        updated = current.model_copy(update=updates)
        return await self._save_server(updated)

    async def delete_server(self, server_id: UUID) -> None:
        await self.services.json_request(
            "Tsuchi", "DELETE", f"/api/mcp-servers/{server_id}"
        )

    async def disconnect(self, server_id: UUID) -> McpServerRecord:
        current = await self.get_server(server_id)
        cleared = current.model_copy(
            update={
                "auth_state": McpAuthState(),
                "status": "disconnected",
                "status_error": None,
            }
        )
        return await self._save_server(cleared)

    # ------------------------------------------------------------------
    # OAuth 2.1 Discovery, PKCE & Refresh
    # ------------------------------------------------------------------

    async def _discover_oauth(
        self, server: McpServerRecord, www_authenticate: str = ""
    ) -> dict[str, Any]:
        target = transport_url(server.url)
        auth_params = _parse_www_authenticate(www_authenticate)
        resource_meta_urls: list[str] = []
        if "resource_metadata" in auth_params:
            resource_meta_urls.append(transport_url(auth_params["resource_metadata"]))
        parsed = urlsplit(target)
        origin = _origin(target)
        path = parsed.path.rstrip("/")
        if path:
            resource_meta_urls.append(
                f"{origin}/.well-known/oauth-protected-resource{path}"
            )
        resource_meta_urls.append(f"{origin}/.well-known/oauth-protected-resource")

        auth_server_origins: list[str] = []
        scope: str | None = auth_params.get("scope")
        for meta_url in resource_meta_urls:
            try:
                resp = await self.services.client.get(
                    meta_url, timeout=httpx.Timeout(10, connect=3)
                )
            except httpx.RequestError:
                continue
            if resp.is_success:
                try:
                    data = resp.json()
                except ValueError:
                    continue
                if isinstance(data, dict):
                    servers = data.get("authorization_servers")
                    if isinstance(servers, list):
                        for entry in servers:
                            if isinstance(entry, str) and entry:
                                auth_server_origins.append(entry.rstrip("/"))
                    scopes = data.get("scopes_supported")
                    if not scope and isinstance(scopes, list) and scopes:
                        scope = " ".join(str(s) for s in scopes if isinstance(s, str))
                    break

        if not auth_server_origins:
            auth_server_origins.append(_origin(server.url))

        for issuer in auth_server_origins:
            issuer_transport = transport_url(issuer)
            candidates = [
                f"{issuer_transport}/.well-known/oauth-authorization-server",
                f"{issuer_transport}/.well-known/openid-configuration",
            ]
            for url in candidates:
                try:
                    resp = await self.services.client.get(
                        url, timeout=httpx.Timeout(10, connect=3)
                    )
                except httpx.RequestError:
                    continue
                if resp.is_success:
                    try:
                        meta = resp.json()
                    except ValueError:
                        continue
                    if (
                        isinstance(meta, dict)
                        and meta.get("authorization_endpoint")
                        and meta.get("token_endpoint")
                    ):
                        return {**meta, "_scope": scope}

        raise ServiceError(
            "Could not discover sign-in settings from this MCP server. "
            "Check that the URL is reachable and supports OAuth, or switch to a Bearer token.",
            502,
            "mcp_oauth_discovery_failed",
        )

    async def _register_oauth_client(
        self, registration_endpoint: str, redirect_uri: str
    ) -> tuple[str, str | None]:
        try:
            resp = await self.services.client.post(
                transport_url(registration_endpoint),
                json={
                    "client_name": "Kadan",
                    "redirect_uris": [redirect_uri],
                    "grant_types": ["authorization_code", "refresh_token"],
                    "response_types": ["code"],
                    "token_endpoint_auth_method": "none",
                },
                timeout=httpx.Timeout(10, connect=3),
            )
        except httpx.RequestError as error:
            raise ServiceError(
                "Could not register Kadan with the MCP server.", 502, "mcp_oauth_error"
            ) from error
        if not resp.is_success:
            raise ServiceError(
                f"The MCP server rejected client registration (HTTP {resp.status_code}).",
                502,
                "mcp_oauth_error",
            )
        try:
            data = resp.json()
        except ValueError as error:
            raise ServiceError(
                "Invalid client registration response from MCP server.",
                502,
                "mcp_oauth_error",
            ) from error
        client_id = data.get("client_id")
        if not isinstance(client_id, str) or not client_id:
            raise ServiceError(
                "MCP server did not return a client_id.", 502, "mcp_oauth_error"
            )
        secret = data.get("client_secret")
        return client_id, secret if isinstance(secret, str) and secret else None

    async def connect(
        self, server_id: UUID, redirect_uri: str
    ) -> McpConnectResponse:
        """Connect to an MCP server: sync tools directly or begin browser OAuth 2.1."""
        server = await self.get_server(server_id)
        if server.auth_type != "oauth":
            synced = await self.refresh_tools(server_id)
            return McpConnectResponse(server=synced.to_public())

        # Probe the MCP endpoint first: if it already works with no/existing token, sync right away.
        probe_www_auth = ""
        try:
            synced = await self.refresh_tools(server_id)
            return McpConnectResponse(server=synced.to_public())
        except ServiceError as error:
            if error.code != "mcp_unauthorized":
                if error.status != 401:
                    raise
            probe_www_auth = getattr(error, "www_authenticate", "")
            # refresh_tools persisted the disconnected status, advancing server.version.
            server = await self.get_server(server_id)

        meta = await self._discover_oauth(server, probe_www_auth)
        auth_endpoint = str(meta["authorization_endpoint"])
        token_endpoint = str(meta["token_endpoint"])
        reg_endpoint = (
            str(meta["registration_endpoint"])
            if meta.get("registration_endpoint")
            else None
        )
        scope = meta.get("_scope")

        client_id = server.auth_state.client_id
        client_secret = server.auth_state.client_secret
        if not client_id:
            if not reg_endpoint:
                raise ServiceError(
                    "This MCP server requires OAuth but does not support automatic client registration. "
                    "Use a Bearer token instead.",
                    422,
                    "mcp_oauth_no_dcr",
                )
            client_id, client_secret = await self._register_oauth_client(
                reg_endpoint, redirect_uri
            )

        updated_auth = server.auth_state.model_copy(
            update={
                "client_id": client_id,
                "client_secret": client_secret,
                "authorization_endpoint": auth_endpoint,
                "token_endpoint": token_endpoint,
                "registration_endpoint": reg_endpoint,
                "scope": scope,
            }
        )
        server = await self._save_server(server.model_copy(update={"auth_state": updated_auth}))

        verifier, challenge = _pkce_pair()
        state = secrets.token_urlsafe(32)
        expires_at = datetime.now(UTC) + timedelta(minutes=OAUTH_STATE_TTL_MINUTES)
        await self.services.json_request(
            "Tsuchi",
            "POST",
            "/api/mcp-oauth-states",
            json={
                "state": state,
                "server_id": str(server.id),
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
                "token_endpoint": token_endpoint,
                "client_id": client_id,
                "client_secret": client_secret,
                "expires_at": expires_at.isoformat(),
            },
        )

        params: dict[str, str] = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "resource": server.url,
        }
        if scope:
            params["scope"] = scope
        sep = "&" if "?" in auth_endpoint else "?"
        authorize_url = f"{auth_endpoint}{sep}{urlencode(params)}"
        return McpConnectResponse(
            server=server.to_public(), authorize_url=authorize_url
        )

    async def complete_oauth(self, state: str, code: str) -> McpServerRecord:
        saved = McpOAuthStateRecord.model_validate(
            await self.services.json_request(
                "Tsuchi", "POST", f"/api/mcp-oauth-states/{state}/consume"
            )
        )
        form: dict[str, str] = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": saved.redirect_uri,
            "client_id": saved.client_id,
            "code_verifier": saved.code_verifier,
        }
        if saved.client_secret:
            form["client_secret"] = saved.client_secret
        try:
            resp = await self.services.client.post(
                transport_url(saved.token_endpoint),
                data=form,
                headers={"Accept": "application/json"},
                timeout=httpx.Timeout(15, connect=3),
            )
        except httpx.RequestError as error:
            raise ServiceError(
                "Could not reach the MCP server to finish sign-in.",
                502,
                "mcp_oauth_error",
            ) from error
        if not resp.is_success:
            raise ServiceError(
                f"Sign-in was rejected by the MCP server (HTTP {resp.status_code}).",
                502,
                "mcp_oauth_error",
            )
        token_data = resp.json()
        access_token = token_data.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise ServiceError(
                "The MCP server did not return an access token.", 502, "mcp_oauth_error"
            )
        expires_in = token_data.get("expires_in")
        expires_at = (
            datetime.now(UTC) + timedelta(seconds=int(expires_in))
            if isinstance(expires_in, (int, float)) and expires_in > 0
            else None
        )
        server = await self.get_server(saved.server_id)
        updated_auth = server.auth_state.model_copy(
            update={
                "access_token": access_token,
                "refresh_token": token_data.get("refresh_token")
                or server.auth_state.refresh_token,
                "token_type": token_data.get("token_type") or "Bearer",
                "expires_at": expires_at,
            }
        )
        server = await self._save_server(
            server.model_copy(
                update={
                    "auth_state": updated_auth,
                    "status": "connected",
                    "status_error": None,
                }
            )
        )
        return await self.refresh_tools(server.id)

    async def _maybe_refresh_token(
        self, server: McpServerRecord, *, force: bool = False
    ) -> McpServerRecord:
        if server.auth_type != "oauth":
            return server
        auth = server.auth_state
        needs_refresh = force or (
            auth.expires_at is not None
            and auth.expires_at <= datetime.now(UTC) + TOKEN_REFRESH_MARGIN
        )
        if not needs_refresh or not auth.refresh_token or not auth.token_endpoint:
            return server
        async with self._lock_for(server.id):
            latest = await self.get_server(server.id)
            auth = latest.auth_state
            if (
                not force
                and auth.expires_at is not None
                and auth.expires_at > datetime.now(UTC) + TOKEN_REFRESH_MARGIN
            ):
                return latest
            if not auth.refresh_token or not auth.token_endpoint:
                return latest
            form: dict[str, str] = {
                "grant_type": "refresh_token",
                "refresh_token": auth.refresh_token,
            }
            if auth.client_id:
                form["client_id"] = auth.client_id
            if auth.client_secret:
                form["client_secret"] = auth.client_secret
            try:
                resp = await self.services.client.post(
                    transport_url(auth.token_endpoint),
                    data=form,
                    headers={"Accept": "application/json"},
                    timeout=httpx.Timeout(15, connect=3),
                )
            except httpx.RequestError:
                return latest
            if not resp.is_success:
                return latest
            try:
                data = resp.json()
            except ValueError:
                return latest
            access = data.get("access_token")
            if not isinstance(access, str) or not access:
                return latest
            expires_in = data.get("expires_in")
            expires_at = (
                datetime.now(UTC) + timedelta(seconds=int(expires_in))
                if isinstance(expires_in, (int, float)) and expires_in > 0
                else None
            )
            updated = latest.model_copy(
                update={
                    "auth_state": auth.model_copy(
                        update={
                            "access_token": access,
                            "refresh_token": data.get("refresh_token")
                            or auth.refresh_token,
                            "expires_at": expires_at,
                        }
                    )
                }
            )
            return await self._save_server(updated)

    # ------------------------------------------------------------------
    # Streamable HTTP JSON-RPC transport
    # ------------------------------------------------------------------

    def _auth_headers(self, server: McpServerRecord) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
        }
        token: str | None = None
        if server.auth_type == "bearer":
            token = server.auth_state.bearer_token
        elif server.auth_type == "oauth":
            token = server.auth_state.access_token
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def _rpc_session(
        self,
        server: McpServerRecord,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = MCP_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        server = await self._maybe_refresh_token(server)
        try:
            return await self._rpc_once(server, method, params, timeout=timeout)
        except ServiceError as error:
            if (
                error.code == "mcp_unauthorized"
                and server.auth_type == "oauth"
                and server.auth_state.refresh_token
            ):
                refreshed = await self._maybe_refresh_token(server, force=True)
                if refreshed.auth_state.access_token != server.auth_state.access_token:
                    return await self._rpc_once(
                        refreshed, method, params, timeout=timeout
                    )
            raise

    async def _rpc_once(
        self,
        server: McpServerRecord,
        method: str,
        params: dict[str, Any] | None,
        *,
        timeout: float,
    ) -> dict[str, Any]:
        url = transport_url(server.url)
        headers = self._auth_headers(server)

        async def post(
            payload: dict[str, Any], session_id: str | None = None
        ) -> httpx.Response:
            req_headers = dict(headers)
            if session_id:
                req_headers["Mcp-Session-Id"] = session_id
            try:
                return await self.services.client.post(
                    url,
                    json=payload,
                    headers=req_headers,
                    timeout=httpx.Timeout(timeout, connect=5),
                )
            except httpx.RequestError as error:
                raise ServiceError(
                    f"Could not reach MCP server '{server.name}'. Check that it is running.",
                    503,
                    "mcp_unavailable",
                ) from error

        # 1. Initialize handshake
        init_resp = await post(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "Kadan", "version": "0.1.0"},
                },
            }
        )
        if init_resp.status_code == 401:
            err = ServiceError(
                f"'{server.name}' requires sign-in. Open Integrations to connect it.",
                401,
                "mcp_unauthorized",
            )
            err.www_authenticate = init_resp.headers.get("www-authenticate", "")  # type: ignore[attr-defined]
            raise err
        if not init_resp.is_success:
            raise ServiceError(
                f"MCP server '{server.name}' returned HTTP {init_resp.status_code}.",
                502,
                "mcp_error",
            )
        init_msg = _parse_mcp_body(init_resp)
        if "error" in init_msg:
            msg = init_msg["error"].get("message", "Initialize failed")
            raise ServiceError(f"MCP error: {msg}", 502, "mcp_error")
        session_id = init_resp.headers.get("mcp-session-id")

        # 2. Best-effort notifications/initialized
        try:
            await post(
                {"jsonrpc": "2.0", "method": "notifications/initialized"}, session_id
            )
        except ServiceError:
            pass

        # 3. Actual JSON-RPC call
        resp = await post(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": method,
                "params": params or {},
            },
            session_id,
        )
        if resp.status_code == 401:
            err = ServiceError(
                f"'{server.name}' sign-in expired. Reconnect it in Integrations.",
                401,
                "mcp_unauthorized",
            )
            err.www_authenticate = resp.headers.get("www-authenticate", "")  # type: ignore[attr-defined]
            raise err
        if not resp.is_success:
            raise ServiceError(
                f"MCP server '{server.name}' returned HTTP {resp.status_code}.",
                502,
                "mcp_error",
            )
        msg = _parse_mcp_body(resp)
        if "error" in msg:
            detail = (
                msg["error"].get("message")
                if isinstance(msg["error"], dict)
                else str(msg["error"])
            )
            raise ServiceError(f"MCP tool error: {detail}", 502, "mcp_error")
        result = msg.get("result")
        return result if isinstance(result, dict) else {"value": result}

    # ------------------------------------------------------------------
    # High-level MCP operations
    # ------------------------------------------------------------------

    async def refresh_tools(self, server_id: UUID) -> McpServerRecord:
        server = await self.get_server(server_id)
        try:
            result = await self._rpc_session(server, "tools/list", {})
        except ServiceError as error:
            failed = server.model_copy(
                update={
                    "status": "disconnected"
                    if error.code == "mcp_unauthorized"
                    else "error",
                    "status_error": str(error),
                }
            )
            await self._save_server(failed)
            raise

        raw_tools = result.get("tools", [])
        tools: list[McpToolInfo] = []
        if isinstance(raw_tools, list):
            for entry in raw_tools:
                if isinstance(entry, dict):
                    parsed = _tool_from_mcp(entry)
                    if parsed is not None:
                        tools.append(parsed)
        names = [tool.name for tool in tools]
        # Preserve existing pinned choices (filtering out removed tools); on first sync
        # apply the preset defaults.
        if server.tools_synced_at is None and not server.pinned_tools:
            pinned = default_pinned_tools(server.preset, names)
        else:
            available = set(names)
            pinned = [name for name in server.pinned_tools if name in available]
        updated = server.model_copy(
            update={
                "status": "connected",
                "status_error": None,
                "tools": tools,
                "pinned_tools": pinned,
                "tools_synced_at": datetime.now(UTC),
            }
        )
        return await self._save_server(updated)

    async def call_tool(
        self,
        server: McpServerRecord,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        timeout: float = MCP_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        result = await self._rpc_session(
            server,
            "tools/call",
            {"name": tool_name, "arguments": arguments},
            timeout=timeout,
        )
        return _normalize_tool_result(result)


def _normalize_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    """Convert MCP tools/call content blocks into a clean dict for the agent."""
    is_error = bool(result.get("isError", False))
    structured = result.get("structuredContent")
    if isinstance(structured, dict) and not is_error:
        return structured

    content = result.get("content")
    texts: list[str] = []
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                texts.append(item["text"])
            elif item.get("type") == "resource" and isinstance(
                item.get("resource"), dict
            ):
                res = item["resource"]
                if isinstance(res.get("text"), str):
                    texts.append(res["text"])
    combined = "\n".join(texts).strip()
    if is_error:
        return {"error": combined or "MCP tool returned an error."}
    if combined:
        try:
            parsed = json.loads(combined)
            if isinstance(parsed, dict):
                return parsed
            return {"result": parsed}
        except ValueError:
            return {"text": combined}
    return result
