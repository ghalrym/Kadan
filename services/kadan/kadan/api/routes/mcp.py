"""Global Remote MCP server setup, OAuth 2.1 callback, and tool sync routes."""

from html import escape
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from ..config import ServiceError
from ..mcp_presets import T3_CODE_TOOL_GROUPS
from ..pydantic_models.mcp import (
    McpConnectResponse,
    McpOAuthCompleteRequest,
    McpServerCreate,
    McpServerPatch,
    McpServerPublic,
)

router = APIRouter(prefix="/api/mcp")


def _first_header_value(value: str | None) -> str | None:
    if not value:
        return None
    first = value.split(",", 1)[0].strip()
    return first or None


def _callback_url(request: Request) -> str:
    # Prefer forwarded browser origin (through reverse proxy or frontend Vite proxy)
    # so OAuth returns to the user's browser URL.
    fwd_proto = _first_header_value(request.headers.get("x-forwarded-proto"))
    fwd_host = _first_header_value(request.headers.get("x-forwarded-host"))
    if not fwd_proto:
        origin = _first_header_value(request.headers.get("origin"))
        if origin:
            parsed_origin = urlsplit(origin)
            if parsed_origin.scheme in {"http", "https"} and parsed_origin.netloc:
                return f"{parsed_origin.scheme}://{parsed_origin.netloc}/api/mcp/oauth/callback"
    proto = fwd_proto or request.url.scheme
    host = fwd_host or _first_header_value(request.headers.get("host"))
    if host:
        return f"{proto}://{host}/api/mcp/oauth/callback"
    return str(request.url_for("mcp_oauth_callback"))


@router.get("/presets")
async def mcp_presets() -> dict:
    return {"t3_code": {"tool_groups": T3_CODE_TOOL_GROUPS}}


@router.get("/servers", response_model=list[McpServerPublic])
async def list_mcp_servers(request: Request) -> list[McpServerPublic]:
    servers = await request.app.state.services.mcp.list_servers()
    return [server.to_public() for server in servers]


@router.post("/servers", response_model=McpServerPublic, status_code=201)
async def create_mcp_server(
    request: Request, payload: McpServerCreate
) -> McpServerPublic:
    created = await request.app.state.services.mcp.create_server(payload)
    return created.to_public()


@router.put("/servers/{server_id}", response_model=McpServerPublic)
async def update_mcp_server(
    request: Request, server_id: UUID, payload: McpServerPatch
) -> McpServerPublic:
    updated = await request.app.state.services.mcp.patch_server(server_id, payload)
    return updated.to_public()


@router.delete("/servers/{server_id}", status_code=204)
async def delete_mcp_server(request: Request, server_id: UUID) -> None:
    await request.app.state.services.mcp.delete_server(server_id)


@router.post("/servers/{server_id}/connect", response_model=McpConnectResponse)
async def connect_mcp_server(
    request: Request, server_id: UUID
) -> McpConnectResponse:
    return await request.app.state.services.mcp.connect(
        server_id, _callback_url(request)
    )


@router.post("/servers/{server_id}/refresh", response_model=McpServerPublic)
async def refresh_mcp_server(request: Request, server_id: UUID) -> McpServerPublic:
    synced = await request.app.state.services.mcp.refresh_tools(server_id)
    return synced.to_public()


@router.post("/servers/{server_id}/disconnect", response_model=McpServerPublic)
async def disconnect_mcp_server(request: Request, server_id: UUID) -> McpServerPublic:
    cleared = await request.app.state.services.mcp.disconnect(server_id)
    return cleared.to_public()


@router.post("/oauth/complete", response_model=McpServerPublic)
async def complete_mcp_oauth(
    request: Request, payload: McpOAuthCompleteRequest
) -> McpServerPublic:
    server = await request.app.state.services.mcp.complete_oauth_from_url(
        payload.callback_url
    )
    return server.to_public()


@router.get("/oauth/callback", name="mcp_oauth_callback", response_class=HTMLResponse)
async def mcp_oauth_callback(
    request: Request,
    state: str | None = Query(default=None),
    code: str | None = Query(default=None),
    error: str | None = Query(default=None),
    error_description: str | None = Query(default=None),
) -> HTMLResponse:
    if error:
        detail = error_description or error
        return HTMLResponse(
            _oauth_result_html(False, f"Sign-in was not completed: {detail}"),
            status_code=400,
        )
    if not state or not code:
        return HTMLResponse(
            _oauth_result_html(False, "Missing authorization code or state."),
            status_code=400,
        )
    try:
        server = await request.app.state.services.mcp.complete_oauth(state, code)
    except ServiceError as exc:
        return HTMLResponse(
            _oauth_result_html(False, str(exc)),
            status_code=exc.status if 400 <= exc.status < 600 else 502,
        )
    return HTMLResponse(
        _oauth_result_html(
            True,
            f"{server.name} is now connected ({len(server.tools)} tools available). "
            "You can close this tab and return to Kadan.",
        )
    )


def _oauth_result_html(ok: bool, message: str) -> str:
    title = "Connected to Kadan" if ok else "Connection failed"
    safe_msg = escape(message)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 460px; margin: 72px auto; padding: 24px; line-height: 1.5; }}
    h1 {{ font-size: 1.25rem; margin: 0 0 12px; }}
    a {{ color: inherit; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>{safe_msg}</p>
  <p><a href="/">Return to Kadan</a></p>
  <script>
    if (window.opener) {{
      try {{ window.opener.postMessage({{ type: "kadan:mcp-oauth", ok: {str(ok).lower()} }}, window.location.origin); }} catch (_) {{}}
    }}
  </script>
</body>
</html>"""
