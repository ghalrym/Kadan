"""Workspace documents stored in the current agent's Tsuchi namespace."""

from typing import Protocol
from uuid import UUID

from pydantic import Field, field_validator

from .base import Arguments, conversation_tool


class CreateWorkspaceDocumentArguments(Arguments):
    title: str = Field(min_length=1, max_length=500, pattern=r"\S")
    markdown: str = Field(default="", max_length=48000)
    parent_id: UUID | None = Field(
        default=None,
        description="An existing active Space document owned by this agent. "
        "Omit to create a top-level document.",
    )

    @field_validator("title")
    @classmethod
    def strip_title(cls, value: str) -> str:
        return value.strip()


class Workspace(Protocol):
    async def create_workspace_document(
        self, args: CreateWorkspaceDocumentArguments
    ) -> dict: ...


@conversation_tool
async def create_workspace_document(
    args: CreateWorkspaceDocumentArguments, conversation: Workspace
) -> dict:
    """Create a Markdown Workspace document for this agent.

    Supply parent_id to create a subdocument inside an existing Space. Memories,
    personas, conversations and decks cannot be parents. Returns the saved ID
    for creating further subdocuments. A failed or uncertain result is not
    confirmation that the document was saved; check Workspace before retrying
    an uncertain write.
    """
    return await conversation.create_workspace_document(args)


TOOLS = [create_workspace_document]
