"""Workspace documents stored in the current agent's Tsuchi namespace."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from pydantic import Field, field_validator

from ..workspace import ProjectStatus
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

    async def list_workspace_documents(
        self, args: ListWorkspaceDocumentsArguments
    ) -> dict: ...

    async def read_workspace_document(
        self, args: ReadWorkspaceDocumentArguments
    ) -> dict: ...

    async def create_project(self, args: CreateProjectArguments) -> dict: ...

    async def update_workspace_document(
        self, args: UpdateWorkspaceDocumentArguments
    ) -> dict: ...

    async def update_project(self, args: UpdateProjectArguments) -> dict: ...


class ListWorkspaceDocumentsArguments(Arguments):
    query: str = Field(default="", max_length=500)
    projects_only: bool = False
    limit: int = Field(default=20, ge=1, le=50)
    offset: int = Field(default=0, ge=0)


class ReadWorkspaceDocumentArguments(Arguments):
    document_id: UUID


class CreateProjectArguments(Arguments):
    title: str = Field(min_length=1, max_length=500, pattern=r"\S")
    markdown: str = Field(
        min_length=1,
        max_length=48000,
        pattern=r"\S",
        description="The complete plan: goal, steps with their state, progress, blockers and any pending decision with options. Preserve user edits when updating.",
    )


class UpdateWorkspaceDocumentArguments(ReadWorkspaceDocumentArguments):
    expected_version: int = Field(ge=1)
    markdown: str = Field(max_length=48000)


class UpdateProjectArguments(UpdateWorkspaceDocumentArguments):
    status: ProjectStatus
    markdown: str = Field(min_length=1, max_length=48000, pattern=r"\S")


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


@conversation_tool
async def list_workspace_documents(
    args: ListWorkspaceDocumentsArguments, conversation: Workspace
) -> dict:
    """Discover this agent's Space documents, or project plans only.

    Returns IDs, versions and project states. Follow next_offset for more;
    pagination counts search results before project filtering. Read a document
    before acting on it. Decks remain readable by known ID.
    """
    return await conversation.list_workspace_documents(args)


@conversation_tool
async def read_workspace_document(
    args: ReadWorkspaceDocumentArguments, conversation: Workspace
) -> dict:
    """Read a current project plan or Workspace document, including its version.

    Read again before updating; preserve user edits. A pending decision is not
    approval to choose for the user. Memories are outside Workspace documents.
    """
    return await conversation.read_workspace_document(args)


@conversation_tool
async def create_project(args: CreateProjectArguments, conversation: Workspace) -> dict:
    """Save a goal and actionable plan as a new active Workspace Space.

    Discover existing projects first to avoid duplicates. Save useful documents
    under its returned id with create_workspace_document. Use update_project
    for progress, blockers and decisions, and send_message to inform the user.
    Saving a plan does not schedule another run.
    """
    return await conversation.create_project(args)


@conversation_tool
async def update_workspace_document(
    args: UpdateWorkspaceDocumentArguments, conversation: Workspace
) -> dict:
    """Replace a Space's Markdown using the version just read.

    Preserve user edits and useful content; metadata and parent links stay saved.
    Conflicts require a fresh read. Use update_project for project plans. Decks
    cannot be edited here. An uncertain write must be checked before retrying.
    """
    return await conversation.update_workspace_document(args)


@conversation_tool
async def update_project(args: UpdateProjectArguments, conversation: Workspace) -> dict:
    """Checkpoint a project's state and complete Markdown plan at its read version.

    Include goal, step states, progress and next action. For blocked or
    awaiting_decision, explain the blocker or exact question and options in the
    plan, inform the user with send_message, and yield. Resume waiting, paused,
    completed or cancelled projects only on explicit user direction. Completion
    of a run is not completion of a project. Conflicts require a fresh read;
    check uncertain writes before retrying.
    """
    return await conversation.update_project(args)


TOOLS = [
    create_workspace_document,
    list_workspace_documents,
    read_workspace_document,
    create_project,
    update_workspace_document,
    update_project,
]
