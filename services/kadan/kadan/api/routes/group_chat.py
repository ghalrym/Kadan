"""Browser API routes for Kadan Group Chat."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Request, Response

from ..config import ServiceError
from ..group_chat_oauth import group_chat_mcp_url as _group_chat_mcp_url
from ..group_chat_plugin import build_chatgpt_plugin_zip
from ..pydantic_models.group_chat import (
    GroupChatCreateExternalParticipantRequest,
    GroupChatCreateExternalParticipantResponse,
    GroupChatDirectChat,
    GroupChatRoom,
    GroupChatRoomSnapshot,
    GroupChatRoomUpdateRequest,
    GroupChatUpdateParticipantRequest,
    GroupChatUpdateParticipantResponse,
    GroupChatUserMessageRequest,
    GroupChatUserMessageResponse,
)
from ..pydantic_models.profile import UserProfile

router = APIRouter(prefix="/api/group-chat")


async def _user_display_name(services) -> str:
    """Name the user's messages after their profile so agents know who is talking."""
    profile = UserProfile.model_validate(
        await services.json_request("Tsuchi", "GET", "/api/profile")
    )
    return profile.display_name


@router.get("/chatgpt-plugin.zip")
async def download_chatgpt_plugin_zip(request: Request) -> Response:
    """Serve a ready-to-upload ChatGPT plugin .zip configured for this Kadan server."""
    archive = build_chatgpt_plugin_zip(_group_chat_mcp_url(request))
    return Response(
        content=archive,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="kadan-group-chat-plugin.zip"',
            "Cache-Control": "no-store",
        },
    )


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
        sender_name=await _user_display_name(services),
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
        pet=payload.pet,
        clear_pet=payload.clear_pet,
    )
    public_list = await services.group_chat.list_public_participants(room.id)
    public = next(item for item in public_list if item.id == updated.id)
    return GroupChatUpdateParticipantResponse(
        participant=public,
        bearer_token=raw_token,
        mcp_url=_group_chat_mcp_url(request) if raw_token else None,
    )


async def _direct_participant(services, room_id: UUID, participant_id: UUID):
    public = next(
        (
            item
            for item in await services.group_chat.list_public_participants(room_id)
            if item.id == participant_id
        ),
        None,
    )
    if public is None or public.kind != "external_agent":
        raise ServiceError("This agent was not found in Group Chat.", 404, "not_found")
    return public


@router.get("/direct", response_model=list[GroupChatDirectChat])
async def list_direct_chats(request: Request) -> list[GroupChatDirectChat]:
    """Every active external agent, with its private chat's newest message if one exists."""
    services = request.app.state.services
    room = await services.group_chat.default_room()
    participants = await services.group_chat.list_public_participants(room.id)
    rooms = {
        item.room.direct_participant_id: item
        for item in await services.group_chat.list_direct_rooms()
    }
    chats = []
    for participant in participants:
        if (
            participant.kind != "external_agent"
            or not participant.enabled
            or participant.revoked_at is not None
        ):
            continue
        summary = rooms.get(participant.id)
        chats.append(
            GroupChatDirectChat(
                participant=participant,
                room_id=summary.room.id if summary else None,
                latest_seq=summary.latest_seq if summary else 0,
                last_message_at=summary.last_message_at if summary else None,
            )
        )
    return chats


@router.get("/direct/{participant_id}", response_model=GroupChatRoomSnapshot)
async def get_direct_chat(
    request: Request,
    participant_id: UUID,
    after_seq: Annotated[int | None, Query(ge=0)] = None,
    before_seq: Annotated[int | None, Query(ge=1)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> GroupChatRoomSnapshot:
    services = request.app.state.services
    home = await services.group_chat.default_room()
    public = await _direct_participant(services, home.id, participant_id)
    room = await services.group_chat.direct_room(participant_id)
    page = await services.group_chat.list_messages(
        room.id, after_seq=after_seq, before_seq=before_seq, limit=limit
    )
    return GroupChatRoomSnapshot(
        room=page.room,
        participants=[public],
        messages=page.messages,
        latest_seq=page.latest_seq,
        has_more=page.has_more,
        turn_cap_reached=page.room.consecutive_agent_turns
        >= page.room.max_consecutive_agent_turns,
        mcp_url=_group_chat_mcp_url(request),
    )


@router.post(
    "/direct/{participant_id}/messages",
    response_model=GroupChatUserMessageResponse,
    status_code=201,
)
async def post_direct_chat_message(
    request: Request, participant_id: UUID, payload: GroupChatUserMessageRequest
) -> GroupChatUserMessageResponse:
    """Every message in a private chat asks its agent to reply."""
    services = request.app.state.services
    home = await services.group_chat.default_room()
    public = await _direct_participant(services, home.id, participant_id)
    if not public.enabled or public.revoked_at is not None:
        raise ServiceError(
            f"{public.display_name} is paused or removed, so it can't reply right now.",
            409,
            "participant_inactive",
        )
    room = await services.group_chat.direct_room(participant_id)
    result = await services.group_chat.create_message(
        room.id,
        sender_kind="user",
        sender_name=await _user_display_name(services),
        content=payload.content,
        handoff_to_participant_ids=[participant_id],
        reply_to_message_id=payload.reply_to_message_id,
        client_request_id=payload.client_request_id,
        trigger_kadan_wake=False,
    )
    return GroupChatUserMessageResponse(
        message=result.message,
        room=result.room,
        effective_handoff_to_participant_ids=result.effective_handoff_to_participant_ids,
        turn_cap_reached=result.turn_cap_reached,
        handoff_suppressed=result.handoff_suppressed,
        idempotent_replay=result.idempotent_replay,
    )
