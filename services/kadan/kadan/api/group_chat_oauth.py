"""OAuth 2.1 + PKCE + Dynamic Client Registration helpers for Kadan Group Chat MCP."""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import secrets
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urlsplit, urlunsplit
from uuid import UUID

from fastapi import Request

from .config import ServiceError
from .group_chat_client import hash_group_chat_token
from .pydantic_models.group_chat import GroupChatParticipant

if TYPE_CHECKING:
    from .services import LocalServices

OAUTH_SCOPE = "group_chat"
AUTH_CODE_TTL_SECONDS = 300
ACCESS_TOKEN_TTL_SECONDS = 86400
REFRESH_TOKEN_TTL_SECONDS = 86400 * 30
OAUTH_CODE_PREFIX = "kdn_gcc_"
OAUTH_ACCESS_PREFIX = "kdn_gco_"
OAUTH_REFRESH_PREFIX = "kdn_gcr_"
OAUTH_CLIENT_PREFIX = "kdn_gcl_"
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _first_header_value(value: str | None) -> str | None:
    if not value:
        return None
    first = value.split(",", 1)[0].strip()
    return first or None


def request_origin(request: Request) -> str:
    """Resolve the browser/external-facing scheme + host, honoring reverse-proxy headers."""
    fwd_proto = _first_header_value(request.headers.get("x-forwarded-proto"))
    fwd_host = _first_header_value(request.headers.get("x-forwarded-host"))
    if not fwd_proto:
        origin = _first_header_value(request.headers.get("origin"))
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                return f"{parsed.scheme}://{parsed.netloc}"
    proto = fwd_proto or request.url.scheme
    host = fwd_host or _first_header_value(request.headers.get("host"))
    if host:
        return f"{proto}://{host}"
    return ""


def request_mount_prefix(request: Request) -> str:
    root_path = str(request.scope.get("root_path", "") or "").rstrip("/")
    return root_path


def group_chat_mcp_url(request: Request) -> str:
    origin = request_origin(request)
    prefix = request_mount_prefix(request)
    return f"{origin}{prefix}/mcp/group-chat"


def group_chat_oauth_issuer(request: Request) -> str:
    origin = request_origin(request)
    prefix = request_mount_prefix(request)
    return f"{origin}{prefix}/api/group-chat/oauth"


def protected_resource_metadata_url(request: Request) -> str:
    origin = request_origin(request)
    prefix = request_mount_prefix(request)
    return f"{origin}/.well-known/oauth-protected-resource{prefix}/mcp/group-chat"


def www_authenticate_header(
    request: Request,
    *,
    error: str | None = None,
    error_description: str | None = None,
) -> str:
    parts = [
        f'resource_metadata="{protected_resource_metadata_url(request)}"',
        f'scope="{OAUTH_SCOPE}"',
    ]
    if error:
        parts.append(f'error="{error}"')
    if error_description:
        cleaned = error_description.replace('"', "'").replace("\n", " ").strip()
        parts.append(f'error_description="{cleaned}"')
    return "Bearer " + ", ".join(parts)


def protected_resource_metadata(request: Request) -> dict[str, Any]:
    return {
        "resource": group_chat_mcp_url(request),
        "authorization_servers": [group_chat_oauth_issuer(request)],
        "scopes_supported": [OAUTH_SCOPE],
        "bearer_methods_supported": ["header"],
    }


def authorization_server_metadata(request: Request) -> dict[str, Any]:
    issuer = group_chat_oauth_issuer(request)
    return {
        "issuer": issuer,
        "authorization_endpoint": f"{issuer}/authorize",
        "token_endpoint": f"{issuer}/token",
        "registration_endpoint": f"{issuer}/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
        "scopes_supported": [OAUTH_SCOPE],
        "client_id_metadata_document_supported": True,
    }


def validate_redirect_uri(uri: str) -> str:
    cleaned = uri.strip()
    parsed = urlsplit(cleaned)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.netloc
        or not parsed.hostname
    ):
        raise ServiceError(
            "Redirect URI must be a valid HTTPS URL (or loopback HTTP for local testing).",
            400,
            "invalid_redirect_uri",
        )
    if parsed.fragment:
        raise ServiceError(
            "Redirect URI must not contain a fragment.",
            400,
            "invalid_redirect_uri",
        )
    if parsed.username or parsed.password:
        raise ServiceError(
            "Redirect URI must not embed credentials.",
            400,
            "invalid_redirect_uri",
        )
    if parsed.scheme == "http" and parsed.hostname.lower() not in LOOPBACK_HOSTS:
        raise ServiceError(
            "Redirect URI must use HTTPS unless targeting localhost.",
            400,
            "invalid_redirect_uri",
        )
    return cleaned


def is_cimd_client_id(client_id: str) -> bool:
    """True when client_id is an HTTPS URL (MCP Client ID Metadata Document)."""
    parsed = urlsplit(client_id.strip())
    return (
        parsed.scheme == "https"
        and bool(parsed.netloc)
        and bool(parsed.hostname)
        and bool(parsed.path and parsed.path != "/")
        and not parsed.fragment
        and not parsed.username
        and not parsed.password
    )


def verify_pkce_s256(code_verifier: str, code_challenge: str) -> bool:
    if not isinstance(code_verifier, str) or not (43 <= len(code_verifier) <= 128):
        return False
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return hmac.compare_digest(computed, code_challenge)


def append_query_params(url: str, params: dict[str, str]) -> str:
    parts = urlsplit(url)
    existing = parts.query
    added = urlencode(params)
    query = f"{existing}&{added}" if existing else added
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def _normalize_resource_url(url: str | None) -> str | None:
    if not url:
        return None
    cleaned = url.strip()
    return cleaned.rstrip("/") if cleaned else None


class GroupChatOAuthService:
    """Kadan OAuth 2.1 authorization server backed by Tsuchi durable storage."""

    def __init__(self, services: LocalServices):
        self.services = services

    async def register_client(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_redirects = payload.get("redirect_uris")
        if (
            not isinstance(raw_redirects, list)
            or not raw_redirects
            or len(raw_redirects) > 20
        ):
            raise ServiceError(
                "redirect_uris must be a non-empty list of valid redirect URLs.",
                400,
                "invalid_client_metadata",
            )
        redirect_uris: list[str] = []
        for item in raw_redirects:
            if not isinstance(item, str):
                raise ServiceError(
                    "Each redirect_uri must be a string.",
                    400,
                    "invalid_redirect_uri",
                )
            redirect_uris.append(validate_redirect_uri(item))

        client_name_raw = payload.get("client_name")
        client_name = (
            client_name_raw.strip()[:256]
            if isinstance(client_name_raw, str) and client_name_raw.strip()
            else "ChatGPT"
        )
        client_id = OAUTH_CLIENT_PREFIX + secrets.token_urlsafe(24)
        saved = await self.services.json_request(
            "Tsuchi",
            "POST",
            "/api/group-chat/oauth/clients",
            json={
                "client_id": client_id,
                "client_name": client_name,
                "redirect_uris": redirect_uris,
                "token_endpoint_auth_method": "none",
            },
        )
        issued_at = int(datetime.now(UTC).timestamp())
        return {
            "client_id": saved["client_id"],
            "client_name": saved["client_name"],
            "redirect_uris": saved["redirect_uris"],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
            "client_id_issued_at": issued_at,
        }

    async def resolve_client(
        self, client_id: str, redirect_uri: str
    ) -> tuple[str, str]:
        """Validate client_id + redirect_uri; returns (client_name, validated_redirect_uri)."""
        cleaned_id = client_id.strip()
        if not cleaned_id:
            raise ServiceError("Missing client_id.", 400, "invalid_client")
        validated_redirect = validate_redirect_uri(redirect_uri)

        try:
            record = await self.services.json_request(
                "Tsuchi",
                "GET",
                f"/api/group-chat/oauth/clients/{cleaned_id}",
            )
            allowed = record.get("redirect_uris") or []
            if validated_redirect not in allowed:
                raise ServiceError(
                    "The redirect address does not match this client's registration.",
                    400,
                    "invalid_redirect_uri",
                )
            return str(record.get("client_name") or "ChatGPT"), validated_redirect
        except ServiceError as error:
            if error.status != 404:
                raise

        if is_cimd_client_id(cleaned_id):
            host = urlsplit(cleaned_id).hostname or "ChatGPT"
            label = "ChatGPT" if "openai" in host or "chatgpt" in host else host
            return label, validated_redirect

        raise ServiceError(
            "This app is not registered with Kadan Group Chat.",
            400,
            "invalid_client",
        )

    async def issue_authorization_code(
        self,
        *,
        room_id: UUID,
        participant_id: UUID,
        client_id: str,
        redirect_uri: str,
        code_challenge: str,
        scope: str,
        resource: str | None,
    ) -> str:
        raw_code = OAUTH_CODE_PREFIX + secrets.token_urlsafe(32)
        expires_at = datetime.now(UTC) + timedelta(seconds=AUTH_CODE_TTL_SECONDS)
        await self.services.json_request(
            "Tsuchi",
            "POST",
            "/api/group-chat/oauth/codes",
            json={
                "code_hash": hash_group_chat_token(raw_code),
                "room_id": str(room_id),
                "participant_id": str(participant_id),
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "code_challenge": code_challenge,
                "code_challenge_method": "S256",
                "scope": scope or OAUTH_SCOPE,
                "resource": resource,
                "expires_at": expires_at.isoformat(),
            },
        )
        return raw_code

    async def exchange_authorization_code(
        self,
        *,
        code: str,
        client_id: str,
        redirect_uri: str,
        code_verifier: str,
        resource: str | None,
        expected_resource: str,
    ) -> dict[str, Any]:
        if not code.strip():
            raise ServiceError("Missing authorization code.", 400, "invalid_grant")
        consumed = await self.services.json_request(
            "Tsuchi",
            "POST",
            "/api/group-chat/oauth/codes/consume",
            json={"code_hash": hash_group_chat_token(code.strip())},
        )
        if consumed["client_id"] != client_id.strip():
            raise ServiceError(
                "client_id does not match the authorization code.",
                400,
                "invalid_grant",
            )
        if consumed["redirect_uri"] != redirect_uri.strip():
            raise ServiceError(
                "redirect_uri does not match the authorization code.",
                400,
                "invalid_grant",
            )
        if not verify_pkce_s256(code_verifier, consumed["code_challenge"]):
            raise ServiceError(
                "PKCE code_verifier verification failed.",
                400,
                "invalid_grant",
            )
        if resource and _normalize_resource_url(resource) != _normalize_resource_url(
            expected_resource
        ):
            raise ServiceError(
                "Requested resource does not match Kadan Group Chat MCP.",
                400,
                "invalid_target",
            )

        now = datetime.now(UTC)
        raw_access = OAUTH_ACCESS_PREFIX + secrets.token_urlsafe(32)
        raw_refresh = OAUTH_REFRESH_PREFIX + secrets.token_urlsafe(32)
        access_expires_at = now + timedelta(seconds=ACCESS_TOKEN_TTL_SECONDS)
        refresh_expires_at = now + timedelta(seconds=REFRESH_TOKEN_TTL_SECONDS)
        scope = str(consumed.get("scope") or OAUTH_SCOPE)
        await self.services.json_request(
            "Tsuchi",
            "POST",
            "/api/group-chat/oauth/tokens",
            json={
                "room_id": consumed["room_id"],
                "participant_id": consumed["participant_id"],
                "client_id": consumed["client_id"],
                "scope": scope,
                "resource": consumed.get("resource") or expected_resource,
                "access_token_hash": hash_group_chat_token(raw_access),
                "refresh_token_hash": hash_group_chat_token(raw_refresh),
                "access_expires_at": access_expires_at.isoformat(),
                "refresh_expires_at": refresh_expires_at.isoformat(),
            },
        )
        return {
            "access_token": raw_access,
            "token_type": "Bearer",
            "expires_in": ACCESS_TOKEN_TTL_SECONDS,
            "refresh_token": raw_refresh,
            "scope": scope,
        }

    async def refresh_access_token(
        self,
        *,
        refresh_token: str,
        client_id: str | None,
    ) -> dict[str, Any]:
        if not refresh_token.strip():
            raise ServiceError("Missing refresh_token.", 400, "invalid_grant")
        now = datetime.now(UTC)
        raw_access = OAUTH_ACCESS_PREFIX + secrets.token_urlsafe(32)
        raw_refresh = OAUTH_REFRESH_PREFIX + secrets.token_urlsafe(32)
        access_expires_at = now + timedelta(seconds=ACCESS_TOKEN_TTL_SECONDS)
        refresh_expires_at = now + timedelta(seconds=REFRESH_TOKEN_TTL_SECONDS)
        refreshed = await self.services.json_request(
            "Tsuchi",
            "POST",
            "/api/group-chat/oauth/tokens/refresh",
            json={
                "refresh_token_hash": hash_group_chat_token(refresh_token.strip()),
                "client_id": client_id.strip() if client_id else None,
                "new_access_token_hash": hash_group_chat_token(raw_access),
                "new_refresh_token_hash": hash_group_chat_token(raw_refresh),
                "access_expires_at": access_expires_at.isoformat(),
                "refresh_expires_at": refresh_expires_at.isoformat(),
            },
        )
        return {
            "access_token": raw_access,
            "token_type": "Bearer",
            "expires_in": ACCESS_TOKEN_TTL_SECONDS,
            "refresh_token": raw_refresh,
            "scope": str(refreshed.get("scope") or OAUTH_SCOPE),
        }


def render_oauth_error_html(title: str, detail: str) -> str:
    safe_title = html.escape(title)
    safe_detail = html.escape(detail)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title} — Kadan Group Chat</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #111214; color: #f3f4f6; margin: 0; padding: 32px 16px; display: grid; place-items: center; min-height: 100vh; box-sizing: border-box; }}
    .card {{ width: 100%; max-width: 460px; background: #1a1c20; border: 1px solid #2b2f38; border-radius: 14px; padding: 24px; }}
    h1 {{ margin: 0 0 10px; font-size: 20px; }}
    p {{ margin: 0; color: #b7becb; line-height: 1.5; }}
  </style>
</head>
<body>
  <main class="card">
    <h1>{safe_title}</h1>
    <p>{safe_detail}</p>
  </main>
</body>
</html>"""


def render_oauth_consent_html(
    *,
    action_url: str,
    client_name: str,
    room_title: str,
    external_participants: list[GroupChatParticipant],
    hidden_fields: dict[str, str],
    error_message: str | None = None,
) -> str:
    safe_action = html.escape(action_url, quote=True)
    safe_client = html.escape(client_name)
    safe_room = html.escape(room_title)
    active_externals = [
        p for p in external_participants if p.enabled and p.revoked_at is None
    ]
    hidden_inputs = "\n".join(
        f'<input type="hidden" name="{html.escape(k, quote=True)}" value="{html.escape(v, quote=True)}" />'
        for k, v in hidden_fields.items()
    )
    existing_options = ""
    for item in active_externals:
        pid = html.escape(str(item.id), quote=True)
        name = html.escape(item.display_name)
        handle = html.escape(item.handle)
        desc = html.escape(item.description)
        desc_html = f'<small class="desc">{desc}</small>' if desc else ""
        existing_options += (
            f'<label class="choice">'
            f'<input type="radio" name="participant_mode" value="existing:{pid}" />'
            f'<span><strong>{name}</strong> <span class="handle">@{handle}</span>{desc_html}</span>'
            f"</label>"
        )

    error_banner = (
        f'<div class="error" role="alert">{html.escape(error_message)}</div>'
        if error_message
        else ""
    )
    default_name = html.escape(
        "Dot" if client_name.lower() == "chatgpt" else client_name, quote=True
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Connect {safe_client} to Kadan Group Chat</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #111214; color: #f3f4f6; margin: 0; padding: 32px 16px; display: grid; place-items: center; min-height: 100vh; box-sizing: border-box; }}
    .card {{ width: 100%; max-width: 480px; background: #1a1c20; border: 1px solid #2b2f38; border-radius: 14px; padding: 24px; box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35); }}
    h1 {{ margin: 0 0 8px; font-size: 20px; }}
    p.lead {{ margin: 0 0 18px; color: #b7becb; line-height: 1.5; font-size: 14px; }}
    .error {{ background: rgba(220, 38, 38, 0.16); border: 1px solid rgba(248, 113, 113, 0.4); color: #fecaca; border-radius: 8px; padding: 10px 12px; margin-bottom: 14px; font-size: 13px; }}
    fieldset {{ border: 0; padding: 0; margin: 0 0 16px; display: grid; gap: 8px; }}
    legend {{ font-size: 13px; font-weight: 600; color: #d1d5db; margin-bottom: 8px; }}
    .choice {{ display: flex; align-items: flex-start; gap: 10px; padding: 10px 12px; border: 1px solid #2e333e; border-radius: 10px; background: #14161a; cursor: pointer; }}
    .choice:has(input:checked) {{ border-color: #60a5fa; background: rgba(96, 165, 250, 0.08); }}
    .choice input {{ margin-top: 3px; }}
    .choice span {{ display: grid; gap: 2px; font-size: 14px; }}
    .handle {{ color: #9ca3af; font-weight: 400; font-size: 13px; }}
    .desc {{ color: #9ca3af; font-size: 12px; }}
    .new-fields {{ display: grid; gap: 10px; margin: 12px 0 18px; padding: 12px; border-radius: 10px; background: #14161a; border: 1px solid #262a33; }}
    .new-fields label {{ display: grid; gap: 5px; font-size: 13px; color: #d1d5db; }}
    .new-fields input[type="text"] {{ background: #0f1115; color: #f3f4f6; border: 1px solid #353b48; border-radius: 8px; padding: 8px 10px; font-size: 14px; }}
    .actions {{ display: flex; gap: 10px; justify-content: flex-end; margin-top: 18px; }}
    button {{ border-radius: 9px; padding: 10px 16px; font-size: 14px; font-weight: 600; cursor: pointer; border: 1px solid #353b48; background: #232730; color: #f3f4f6; }}
    button.primary {{ background: #2563eb; border-color: #2563eb; color: #fff; }}
    button.primary:hover {{ background: #1d4ed8; }}
    button:hover {{ filter: brightness(1.08); }}
    .note {{ margin-top: 14px; font-size: 12px; color: #9ca3af; line-height: 1.45; }}
  </style>
</head>
<body>
  <main class="card">
    <h1>Connect {safe_client} to {safe_room}</h1>
    <p class="lead">
      Choose which agent identity <strong>{safe_client}</strong> will use in Kadan Group Chat.
      You can pause or revoke access at any time from Participants &amp; Connections.
    </p>
    {error_banner}
    <form method="post" action="{safe_action}">
      {hidden_inputs}
      <fieldset>
        <legend>Agent identity</legend>
        <label class="choice">
          <input type="radio" name="participant_mode" value="new" checked />
          <span><strong>Create a new external agent</strong><small class="desc">Adds a new participant to Group Chat for this connection.</small></span>
        </label>
        {existing_options}
      </fieldset>
      <div class="new-fields" id="new-agent-fields">
        <label>
          Display name
          <input type="text" name="display_name" value="{default_name}" maxlength="200" placeholder="e.g. Dot" />
        </label>
        <label>
          Short role description (optional)
          <input type="text" name="description" value="" maxlength="500" placeholder="e.g. ChatGPT research teammate" />
        </label>
      </div>
      <div class="actions">
        <button type="submit" name="decision" value="deny">Cancel</button>
        <button type="submit" name="decision" value="approve" class="primary">Connect agent</button>
      </div>
      <p class="note">
        Connected agents can read room messages, post replies as their chosen identity, and receive wake-ups when tagged.
      </p>
    </form>
  </main>
  <script>
    (function () {{
      var box = document.getElementById("new-agent-fields");
      function sync() {{
        var selected = document.querySelector('input[name="participant_mode"]:checked');
        if (box) box.style.display = selected && selected.value === "new" ? "grid" : "none";
      }}
      document.querySelectorAll('input[name="participant_mode"]').forEach(function (el) {{
        el.addEventListener("change", sync);
      }});
      sync();
    }})();
  </script>
</body>
</html>"""
