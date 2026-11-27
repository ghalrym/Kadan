"""OAuth 2.1 discovery, Dynamic Client Registration, consent UI, and token routes for Group Chat MCP."""

from __future__ import annotations

import base64
import binascii
from urllib.parse import parse_qs
from uuid import UUID

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ..config import ServiceError
from ..group_chat_oauth import (
    OAUTH_SCOPE,
    GroupChatOAuthService,
    append_query_params,
    authorization_server_metadata,
    group_chat_mcp_url,
    group_chat_oauth_issuer,
    protected_resource_metadata,
    render_oauth_consent_html,
    render_oauth_error_html,
)

router = APIRouter()


def _oauth_json_error(
    error: str, description: str, status_code: int = 400
) -> JSONResponse:
    return JSONResponse(
        {"error": error, "error_description": description},
        status_code=status_code,
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


async def _parse_form_or_json(request: Request) -> dict[str, str]:
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        try:
            body = await request.json()
        except ValueError:
            return {}
        if not isinstance(body, dict):
            return {}
        return {str(k): str(v) for k, v in body.items() if v is not None}
    raw = (await request.body()).decode("utf-8", errors="replace")
    parsed = parse_qs(raw, keep_blank_values=True)
    return {k: values[-1] for k, values in parsed.items() if values}


def _extract_basic_client_id(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    scheme, _, encoded = auth.partition(" ")
    if scheme.lower() != "basic" or not encoded.strip():
        return None
    try:
        decoded = base64.b64decode(encoded.strip()).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    client_id, _, _ = decoded.partition(":")
    return client_id or None


# ── RFC 9728 Protected Resource Metadata ─────────────────────────────────────


@router.get("/.well-known/oauth-protected-resource")
@router.get("/.well-known/oauth-protected-resource/{resource_path:path}")
async def group_chat_protected_resource_metadata(
    request: Request, resource_path: str = ""
) -> JSONResponse:
    _ = resource_path
    return JSONResponse(protected_resource_metadata(request))


# ── RFC 8414 Authorization Server Metadata (+ OIDC discovery fallback) ───────


@router.get("/.well-known/oauth-authorization-server")
@router.get("/.well-known/oauth-authorization-server/{issuer_path:path}")
@router.get("/.well-known/openid-configuration")
@router.get("/.well-known/openid-configuration/{issuer_path:path}")
@router.get("/api/group-chat/oauth/.well-known/oauth-authorization-server")
@router.get("/api/group-chat/oauth/.well-known/openid-configuration")
async def group_chat_authorization_server_metadata(
    request: Request, issuer_path: str = ""
) -> JSONResponse:
    _ = issuer_path
    return JSONResponse(authorization_server_metadata(request))


# ── RFC 7591 Dynamic Client Registration ─────────────────────────────────────


@router.post("/api/group-chat/oauth/register", status_code=201)
async def register_group_chat_oauth_client(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except ValueError:
        return _oauth_json_error(
            "invalid_client_metadata", "Registration body must be valid JSON."
        )
    if not isinstance(payload, dict):
        return _oauth_json_error(
            "invalid_client_metadata", "Registration body must be a JSON object."
        )
    oauth = GroupChatOAuthService(request.app.state.services)
    try:
        registered = await oauth.register_client(payload)
    except ServiceError as error:
        return _oauth_json_error(
            error.code or "invalid_client_metadata", str(error), error.status
        )
    return JSONResponse(
        registered,
        status_code=201,
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


# ── OAuth 2.1 Authorization Endpoint (Browser Consent UI) ────────────────────


@router.get("/api/group-chat/oauth/authorize", response_class=HTMLResponse)
async def get_group_chat_oauth_authorize(request: Request) -> HTMLResponse:
    params = request.query_params
    response_type = params.get("response_type", "")
    client_id = params.get("client_id", "")
    redirect_uri = params.get("redirect_uri", "")
    code_challenge = params.get("code_challenge", "")
    code_challenge_method = params.get("code_challenge_method", "")
    state = params.get("state", "")
    scope = params.get("scope", "") or OAUTH_SCOPE
    resource = params.get("resource", "")

    oauth = GroupChatOAuthService(request.app.state.services)
    try:
        client_name, validated_redirect = await oauth.resolve_client(
            client_id, redirect_uri
        )
    except ServiceError as error:
        return HTMLResponse(
            render_oauth_error_html("Cannot connect app", str(error)),
            status_code=400,
        )

    if response_type != "code":
        return HTMLResponse(
            render_oauth_error_html(
                "Unsupported sign-in request",
                "Only authorization code sign-in is supported.",
            ),
            status_code=400,
        )
    if code_challenge_method != "S256" or len(code_challenge.strip()) < 16:
        return HTMLResponse(
            render_oauth_error_html(
                "Missing security check",
                "This app must provide a PKCE S256 challenge to connect.",
            ),
            status_code=400,
        )

    services = request.app.state.services
    room = await services.group_chat.default_room()
    await services.group_chat.ensure_kadan_participant(room.id)
    participants = await services.group_chat.list_participants(room.id)
    external = [p for p in participants if p.kind == "external_agent"]
    action_url = f"{group_chat_oauth_issuer(request)}/authorize"

    return HTMLResponse(
        render_oauth_consent_html(
            action_url=action_url,
            client_name=client_name,
            room_title=room.title,
            external_participants=external,
            hidden_fields={
                "client_id": client_id,
                "redirect_uri": validated_redirect,
                "code_challenge": code_challenge.strip(),
                "code_challenge_method": "S256",
                "state": state,
                "scope": scope,
                "resource": resource,
            },
        )
    )


@router.post("/api/group-chat/oauth/authorize")
async def post_group_chat_oauth_authorize(request: Request):
    form = await _parse_form_or_json(request)
    client_id = form.get("client_id", "")
    redirect_uri = form.get("redirect_uri", "")
    code_challenge = form.get("code_challenge", "")
    code_challenge_method = form.get("code_challenge_method", "")
    state = form.get("state", "")
    scope = form.get("scope", "") or OAUTH_SCOPE
    resource = form.get("resource", "") or None
    decision = form.get("decision", "deny")

    oauth = GroupChatOAuthService(request.app.state.services)
    try:
        client_name, validated_redirect = await oauth.resolve_client(
            client_id, redirect_uri
        )
    except ServiceError as error:
        return HTMLResponse(
            render_oauth_error_html("Cannot connect app", str(error)),
            status_code=400,
        )

    if decision != "approve":
        err_params = {
            "error": "access_denied",
            "error_description": "The user cancelled connecting this agent.",
        }
        if state:
            err_params["state"] = state
        return RedirectResponse(
            append_query_params(validated_redirect, err_params), status_code=303
        )

    if code_challenge_method != "S256" or len(code_challenge.strip()) < 16:
        return HTMLResponse(
            render_oauth_error_html(
                "Missing security check",
                "This app must provide a PKCE S256 challenge to connect.",
            ),
            status_code=400,
        )

    services = request.app.state.services
    room = await services.group_chat.default_room()
    await services.group_chat.ensure_kadan_participant(room.id)
    participants = await services.group_chat.list_participants(room.id)
    external = [p for p in participants if p.kind == "external_agent"]
    action_url = f"{group_chat_oauth_issuer(request)}/authorize"

    hidden_fields = {
        "client_id": client_id,
        "redirect_uri": validated_redirect,
        "code_challenge": code_challenge.strip(),
        "code_challenge_method": "S256",
        "state": state,
        "scope": scope,
        "resource": resource or "",
    }

    mode = form.get("participant_mode", "new")
    participant_id: UUID | None = None
    try:
        if mode.startswith("existing:"):
            raw_pid = mode.split(":", 1)[1].strip()
            candidate_id = UUID(raw_pid)
            match = next(
                (
                    p
                    for p in external
                    if p.id == candidate_id and p.enabled and p.revoked_at is None
                ),
                None,
            )
            if match is None:
                raise ServiceError(
                    "That external agent is no longer active. Choose another agent or create a new one.",
                    400,
                    "participant_inactive",
                )
            participant_id = match.id
        else:
            display_name = form.get("display_name", "").strip()
            description = form.get("description", "").strip()
            if not display_name:
                raise ServiceError(
                    "Enter a display name for the new agent.", 400, "invalid_name"
                )
            created, _ = await services.group_chat.create_external_participant(
                room.id,
                display_name=display_name,
                description=description,
            )
            participant_id = created.id
    except (ValueError, ServiceError) as error:
        return HTMLResponse(
            render_oauth_consent_html(
                action_url=action_url,
                client_name=client_name,
                room_title=room.title,
                external_participants=external,
                hidden_fields=hidden_fields,
                error_message=str(error),
            ),
            status_code=400,
        )

    raw_code = await oauth.issue_authorization_code(
        room_id=room.id,
        participant_id=participant_id,
        client_id=client_id.strip(),
        redirect_uri=validated_redirect,
        code_challenge=code_challenge.strip(),
        scope=scope,
        resource=resource,
    )
    ok_params = {"code": raw_code}
    if state:
        ok_params["state"] = state
    return RedirectResponse(
        append_query_params(validated_redirect, ok_params), status_code=303
    )


# ── OAuth 2.1 Token Endpoint ─────────────────────────────────────────────────


@router.post("/api/group-chat/oauth/token")
async def post_group_chat_oauth_token(request: Request) -> JSONResponse:
    form = await _parse_form_or_json(request)
    grant_type = form.get("grant_type", "").strip()
    client_id = form.get("client_id", "").strip() or (
        _extract_basic_client_id(request) or ""
    )
    oauth = GroupChatOAuthService(request.app.state.services)

    try:
        if grant_type == "authorization_code":
            if not client_id:
                return _oauth_json_error("invalid_client", "Missing client_id.", 401)
            result = await oauth.exchange_authorization_code(
                code=form.get("code", ""),
                client_id=client_id,
                redirect_uri=form.get("redirect_uri", ""),
                code_verifier=form.get("code_verifier", ""),
                resource=form.get("resource") or None,
                expected_resource=group_chat_mcp_url(request),
            )
            return JSONResponse(
                result,
                status_code=200,
                headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
            )

        if grant_type == "refresh_token":
            result = await oauth.refresh_access_token(
                refresh_token=form.get("refresh_token", ""),
                client_id=client_id or None,
            )
            return JSONResponse(
                result,
                status_code=200,
                headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
            )

        return _oauth_json_error(
            "unsupported_grant_type",
            "Only authorization_code and refresh_token grants are supported.",
            400,
        )
    except ServiceError as error:
        oauth_err = (
            error.code
            if error.code
            in {"invalid_grant", "invalid_client", "invalid_target", "invalid_request"}
            else "invalid_grant"
        )
        return _oauth_json_error(oauth_err, str(error), 400)
