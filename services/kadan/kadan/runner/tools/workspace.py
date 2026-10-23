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

    async def save_workspace_file(self, args: SaveWorkspaceFileArguments) -> dict: ...

    async def read_workspace_file(self, args: ReadWorkspaceFileArguments) -> dict: ...

    async def open_workspace_file(self, args: OpenWorkspaceFileArguments) -> dict: ...

    async def list_workspace_files(self, args: ListWorkspaceFilesArguments) -> dict: ...


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


class SaveWorkspaceFileArguments(Arguments):
    path: str = Field(
        min_length=1,
        max_length=4096,
        pattern=r"^/",
        description="Absolute path to a completed file on the agent's computer.",
    )
    title: str | None = Field(default=None, min_length=1, max_length=500, pattern=r"\S")

    @field_validator("path")
    @classmethod
    def valid_path(cls, value: str) -> str:
        if "\0" in value:
            raise ValueError("A file path cannot contain a NUL character")
        return value

    @field_validator("title")
    @classmethod
    def strip_title(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None


class ReadWorkspaceFileArguments(Arguments):
    file_id: UUID


class ListWorkspaceFilesArguments(Arguments):
    query: str = Field(default="", max_length=500)
    limit: int = Field(default=20, ge=1, le=50)
    offset: int = Field(default=0, ge=0)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        return value.strip()


class OpenWorkspaceFileArguments(Arguments):
    file_id: UUID


@conversation_tool
async def save_workspace_file(
    args: SaveWorkspaceFileArguments, conversation: Workspace
) -> dict:
    """Save a completed computer file, up to 100 MiB, in this agent's Workspace Files.

    Returns a saved file ID and download URL. This does not send a message: use
    send_message with a Markdown link to that URL to deliver it to the user.
    The computer's source stays unchanged. Failed or interrupted saves may have
    completed; use read_workspace_file with the returned file_id before retrying.
    """
    return await conversation.save_workspace_file(args)


@conversation_tool
async def read_workspace_file(
    args: ReadWorkspaceFileArguments, conversation: Workspace
) -> dict:
    """Read saved file metadata and its download URL, or check an uncertain save.

    This does not read file contents. Missing or trashed files are reported as
    errors and never recreated from the computer automatically.
    """
    return await conversation.read_workspace_file(args)


@conversation_tool
async def list_workspace_files(
    args: ListWorkspaceFilesArguments, conversation: Workspace
) -> dict:
    """List or search this agent's active file metadata, newest update first, then ID.

    Optional query requires all whitespace-separated tokens to match
    case-insensitive substrings in title, filename, tags or source labels/URLs.
    Never searches contents or meaning; empty files means no metadata matches.
    Names can repeat; choose by file_id, then read_workspace_file or
    open_workspace_file. Continue with next_offset and the same query/limit;
    null ends paging. Concurrent edits can shift pages. No file bytes.
    """
    return await conversation.list_workspace_files(args)


@conversation_tool
async def open_workspace_file(
    args: OpenWorkspaceFileArguments, conversation: Workspace
) -> dict:
    """Copy an active Workspace file, up to 100 MiB, onto the agent's computer.

    Returns a verified local path for file, shell or browser tools. This does not
    read the contents into chat. The saved original is unchanged; edits to the
    computer copy are not synchronized. An interrupted copy is checked on restart
    and is never uploaded again automatically. Inspect an unconfirmed path before
    requesting another copy.
    """
    return await conversation.open_workspace_file(args)


TOOLS = [
    create_workspace_document,
    list_workspace_documents,
    read_workspace_document,
    create_project,
    update_workspace_document,
    update_project,
    save_workspace_file,
    read_workspace_file,
    list_workspace_files,
    open_workspace_file,
]
