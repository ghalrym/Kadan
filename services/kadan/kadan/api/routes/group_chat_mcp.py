"""Streamable HTTP MCP server endpoint for external Group Chat agents."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from ..config import ServiceError
from ..group_chat_mcp import GroupChatMcpHandler, GroupChatRateLimiter

router = APIRouter()


def _extract_bearer_token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise ServiceError(
            "Missing or invalid Authorization Bearer token.", 401, "missing_token"
        )
    return token.strip()


def _rate_limiter(request: Request) -> GroupChatRateLimiter:
    limiter = getattr(request.app.state, "group_chat_rate_limiter", None)
    if limiter is None:
        limiter = GroupChatRateLimiter()
        request.app.state.group_chat_rate_limiter = limiter
    return limiter


@router.get("/mcp/group-chat")
async def group_chat_mcp_get() -> Response:
    # Streamable HTTP clients that probe GET for standalone SSE receive 405 cleanly.
    return Response(status_code=405, headers={"Allow": "POST, DELETE"})


@router.delete("/mcp/group-chat", status_code=204)
async def group_chat_mcp_delete(request: Request) -> Response:
    token = _extract_bearer_token(request)
    await request.app.state.services.group_chat.authenticate_token(token)
    return Response(status_code=204)


@router.post("/mcp/group-chat")
async def group_chat_mcp_post(request: Request) -> Response:
    token = _extract_bearer_token(request)
    services = request.app.state.services
    participant = await services.group_chat.authenticate_token(token)

    limiter = _rate_limiter(request)
    allowed, retry_after = limiter.check_request(participant.id)
    if not allowed:
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32029,
                    "message": "Rate limit exceeded for this Group Chat participant.",
                    "data": {"retry_after": retry_after},
                },
            },
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )

    try:
        payload = await request.json()
    except ValueError:
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Invalid JSON payload."},
            },
            status_code=400,
        )

    handler = GroupChatMcpHandler(services, rate_limiter=limiter)
    status_code, body = await handler.handle_jsonrpc(participant, payload)
    if body is None:
        return Response(status_code=status_code)
    return JSONResponse(body, status_code=status_code)
