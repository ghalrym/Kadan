"""Bounded views of delivered chat stored in a conversation resource."""

import re
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..api.pydantic_models.resources import Message, Resource


class HistoryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ReadHistoryArguments(HistoryArguments):
    start_message_id: UUID | None = Field(
        default=None,
        description="Inclusive message anchor; omitted starts at oldest delivered chat.",
    )
    expected_version: int | None = Field(
        default=None,
        ge=1,
        description="Use the returned source version when continuing a page; changes require restarting.",
    )
    limit: int = Field(default=10, ge=1, le=20)
    max_characters: int = Field(
        default=8000,
        ge=1,
        le=16000,
        description="Total returned content characters, excluding metadata.",
    )
    content_offset: int = Field(
        default=0,
        ge=0,
        description="Character offset in the anchored first message; use next_content_offset to continue a cut message.",
    )


class SearchHistoryArguments(HistoryArguments):
    query: str = Field(min_length=1, max_length=200, pattern=r"\S")
    start_message_id: UUID | None = Field(
        default=None,
        description="Inclusive message anchor; omitted starts at oldest delivered chat.",
    )
    expected_version: int | None = Field(default=None, ge=1)
    limit: int = Field(default=10, ge=1, le=20)
    max_scan: int = Field(
        default=200,
        ge=1,
        le=500,
        description="Maximum eligible messages examined on this page.",
    )
    excerpt_characters: int = Field(default=400, ge=1, le=1000)
    max_characters: int = Field(
        default=8000,
        ge=1,
        le=16000,
        description="Total returned excerpt characters, excluding metadata.",
    )


def history_page(
    resource: Resource, args: ReadHistoryArguments | SearchHistoryArguments
) -> dict:
    eligible = [
        message
        for message in resource.messages
        if message.role in {"user", "assistant"}
    ]
    if args.expected_version is not None and args.expected_version != resource.version:
        raise ValueError(
            "Conversation changed; restart pagination with its current version."
        )
    if args.start_message_id is not None and args.expected_version is None:
        raise ValueError(
            "A message anchor requires expected_version from the preceding result."
        )
    start = 0
    if args.start_message_id is not None:
        start = next(
            (
                index
                for index, message in enumerate(eligible)
                if message.id == args.start_message_id
            ),
            -1,
        )
        if start == -1:
            raise ValueError(
                "Message anchor is not delivered chat in this conversation."
            )
    search = isinstance(args, SearchHistoryArguments)
    if not search and args.content_offset and args.start_message_id is None:
        raise ValueError("A content offset requires a message anchor.")
    pattern = re.compile(re.escape(args.query), re.IGNORECASE) if search else None
    results = []
    scanned = 0
    characters = 0
    index = start
    next_offset = 0
    while index < len(eligible):
        if len(results) >= args.limit or characters >= args.max_characters:
            break
        if search and scanned >= args.max_scan:
            break
        message = eligible[index]
        scanned += 1
        match = pattern.search(message.content) if pattern else None
        if search and match is None:
            index += 1
            continue
        offset = (
            match.start() if search else (args.content_offset if index == start else 0)
        )
        if offset > len(message.content):
            raise ValueError("Content offset exceeds the anchored message length.")
        length = min(
            args.max_characters - characters,
            args.excerpt_characters if search else len(message.content),
        )
        content = message.content[offset : offset + length]
        end = offset + len(content)
        results.append(message_excerpt(message, content, offset, end))
        characters += len(content)
        if not search and end < len(message.content):
            next_offset = end
            break
        index += 1
    return {
        "conversation_id": str(resource.id),
        "source_version": resource.version,
        "source": "stored_delivered_chat",
        "fetch_scope": "full_conversation_resource_snapshot",
        "eligible_messages": len(eligible),
        "start_eligible_offset": start,
        "scanned_eligible_messages": scanned,
        "scanned_first_message_id": str(eligible[start].id) if scanned else None,
        "scanned_last_message_id": str(eligible[start + scanned - 1].id)
        if scanned
        else None,
        "messages": results,
        "returned_content_characters": characters,
        "content_cut": any(item["content_cut"] for item in results),
        "scan_limit_reached": search
        and scanned >= args.max_scan
        and index < len(eligible),
        "result_limit_reached": len(results) >= args.limit and index < len(eligible),
        "character_limit_reached": characters >= args.max_characters
        and index < len(eligible),
        "more_available": index < len(eligible),
        "next_message_id": str(eligible[index].id) if index < len(eligible) else None,
        "next_content_offset": next_offset,
        "whole_history_scanned": start == 0
        and index == len(eligible)
        and (search or args.content_offset == 0),
    }


def message_excerpt(message: Message, content: str, start: int, end: int) -> dict:
    return {
        "id": str(message.id),
        "role": message.role,
        "created_at": message.created_at.isoformat(),
        "content": content,
        "content_start": start,
        "content_end": end,
        "original_content_characters": len(message.content),
        "content_cut": start > 0 or end < len(message.content),
    }
