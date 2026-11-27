"""Browser API routes for Kadan Group Chat."""

from __future__ import annotations

from typing import Annotated
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import APIRouter, Query, Request

from ..pydantic_models.group_chat import (
    GroupChatCreateExternalParticipantRequest,
    GroupChatCreateExternalParticipantResponse,
    GroupChatRoom,
    GroupChatRoomSnapshot,
    GroupChatRoomUpdateRequest,
    GroupChatUpdateParticipantRequest,
    GroupChatUpdateParticipantResponse,
    GroupChatUserMessageRequest,
    GroupChatUserMessageResponse,
)

router = APIRouter(prefix="/api/group-chat")


def _first_header_value(value: str | None) -> str | None:
    if not value:
        return None
    first = value.split(",", 1)[0].strip()
    return first or None


def _group_chat_mcp_url(request: Request) -> str:
    """Build the copyable Group Chat MCP URL preserving mount prefix (e.g. /kadan)."""
    path = request.scope.get("root_path", "") + "/mcp/group-chat"
    fwd_proto = _first_header_value(request.headers.get("x-forwarded-proto"))
    fwd_host = _first_header_value(request.headers.get("x-forwarded-host"))
    if not fwd_proto:
        origin = _first_header_value(request.headers.get("origin"))
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                return f"{parsed.scheme}://{parsed.netloc}{path}"
    proto = fwd_proto or request.url.scheme
    host = fwd_host or _first_header_value(request.headers.get("host"))
    if host:
        return f"{proto}://{host}{path}"
    return path


@router.get("", response_model=GroupChatRoomSnapshot)
async def get_group_chat_snapshot(
    request: Request,
    after_seq: Annotated[int | None, Query(ge=0)] = None,
    before_seq: Annotated[int | None, Query(ge=1)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> GroupChatRoomSnapshot:
    services = request.app.state.services
    room = await services.group_chat.default_room()
    await services.group_chat.ensure_kadan_participant(room.id)
    participants = await services.group_chat.list_public_participants(room.id)
    page = await services.group_chat.list_messages(
        room.id, after_seq=after_seq, before_seq=before_seq, limit=limit
    )
    return GroupChatRoomSnapshot(
        room=page.room,
        participants=participants,
        messages=page.messages,
        latest_seq=page.latest_seq,
        has_more=page.has_more,
        turn_cap_reached=page.room.consecutive_agent_turns
        >= page.room.max_consecutive_agent_turns,
        mcp_url=_group_chat_mcp_url(request),
    )


@router.put("/room", response_model=GroupChatRoom)
async def update_group_chat_room(
    request: Request, payload: GroupChatRoomUpdateRequest
) -> GroupChatRoom:
    services = request.app.state.services
    room = await services.group_chat.default_room()
    return await services.group_chat.update_room(
        room.id,
        expected_version=payload.expected_version,
        title=payload.title,
        max_consecutive_agent_turns=payload.max_consecutive_agent_turns,
    )


@router.post("/messages", response_model=GroupChatUserMessageResponse, status_code=201)
async def post_group_chat_user_message(
    request: Request, payload: GroupChatUserMessageRequest
) -> GroupChatUserMessageResponse:
    services = request.app.state.services
    room = await services.group_chat.default_room()
    await services.group_chat.ensure_kadan_participant(room.id)
    result = await services.group_chat.create_message(
        room.id,
        sender_kind="user",
        sender_name="You",
        content=payload.content,
        handoff_to_participant_ids=payload.handoff_to_participant_ids,
        reply_to_message_id=payload.reply_to_message_id,
        client_request_id=payload.client_request_id,
    )
    return GroupChatUserMessageResponse(
        message=result.message,
        room=result.room,
        effective_handoff_to_participant_ids=result.effective_handoff_to_participant_ids,
        turn_cap_reached=result.turn_cap_reached,
        handoff_suppressed=result.handoff_suppressed,
        idempotent_replay=result.idempotent_replay,
    )


@router.post(
    "/participants",
    response_model=GroupChatCreateExternalParticipantResponse,
    status_code=201,
)
async def create_external_participant(
    request: Request, payload: GroupChatCreateExternalParticipantRequest
) -> GroupChatCreateExternalParticipantResponse:
    services = request.app.state.services
    room = await services.group_chat.default_room()
    await services.group_chat.ensure_kadan_participant(room.id)
    created, raw_token = await services.group_chat.create_external_participant(
        room.id,
        display_name=payload.display_name,
        description=payload.description,
        handle=payload.handle,
    )
    public_list = await services.group_chat.list_public_participants(room.id)
    public = next(item for item in public_list if item.id == created.id)
    return GroupChatCreateExternalParticipantResponse(
        participant=public,
        bearer_token=raw_token,
        mcp_url=_group_chat_mcp_url(request),
    )


@router.put(
    "/participants/{participant_id}",
    response_model=GroupChatUpdateParticipantResponse,
)
async def update_group_chat_participant(
    request: Request,
    participant_id: UUID,
    payload: GroupChatUpdateParticipantRequest,
) -> GroupChatUpdateParticipantResponse:
    services = request.app.state.services
    room = await services.group_chat.default_room()
    updated, raw_token = await services.group_chat.update_participant(
        room.id,
        participant_id,
        expected_version=payload.expected_version,
        display_name=payload.display_name,
        description=payload.description,
        enabled=payload.enabled,
        rotate_token=payload.rotate_token,
        revoke=payload.revoke,
    )
    public_list = await services.group_chat.list_public_participants(room.id)
    public = next(item for item in public_list if item.id == updated.id)
    return GroupChatUpdateParticipantResponse(
        participant=public,
        bearer_token=raw_token,
        mcp_url=_group_chat_mcp_url(request) if raw_token else None,
    )
