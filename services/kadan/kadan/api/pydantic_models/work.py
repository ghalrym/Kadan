"""Pydantic models for Kadan's native Work projects, threads, runs, activity, and harnesses."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from .resources import AgentKey

WorkBucket = Literal["default", "working", "snoozed", "settled"]
WorkRunStatus = Literal[
    "idle",
    "preparing",
    "queued",
    "running",
    "waiting",
    "completed",
    "failed",
    "interrupted",
    "cancelled",
]
WorkWorkspaceMode = Literal["root", "worktree"]
WorkHarnessId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,47}$")]
WorkActivityKind = Literal[
    "user_message",
    "assistant_message",
    "reasoning",
    "tool_call",
    "tool_result",
    "file_change",
    "pending_request",
    "status",
    "error",
]


class WorkModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkProject(BaseModel):
    id: UUID
    title: str
    root_path: str
    default_branch: str = "main"
    setup_script: str | None = None
    version: int
    created_at: datetime
    updated_at: datetime


class WorkProjectCreate(WorkModel):
    title: str | None = Field(default=None, max_length=500)
    root_path: str | None = Field(default=None, max_length=4000)
    clone_url: str | None = Field(default=None, max_length=2000)
    default_branch: str | None = Field(default=None, max_length=256)
    setup_script: str | None = Field(default=None, max_length=16000)


class WorkProjectPatch(WorkModel):
    title: str | None = Field(default=None, min_length=1, max_length=500)
    default_branch: str | None = Field(default=None, min_length=1, max_length=256)
    setup_script: str | None = Field(default=None, max_length=16000)
    expected_version: int = Field(ge=1)


class WorkPendingRequest(BaseModel):
    id: str
    kind: Literal["question", "approval"]
    title: str = ""
    prompt: str
    options: list[dict[str, JsonValue]] = Field(default_factory=list)
    tool_name: str | None = None
    arguments: dict[str, JsonValue] | None = None
    status: Literal["pending", "answered", "cancelled"] = "pending"
    response: dict[str, JsonValue] | None = None
    created_at: datetime
    resolved_at: datetime | None = None


class WorkThread(BaseModel):
    id: UUID
    client_request_id: str | None = None
    project_id: UUID
    title: str
    harness: str
    model: str | None = None
    model_options: dict[str, JsonValue] = Field(default_factory=dict)
    workspace_mode: WorkWorkspaceMode = "root"
    worktree_path: str | None = None
    branch: str | None = None
    base_ref: str | None = None
    run_status: WorkRunStatus = "idle"
    bucket: WorkBucket = "default"
    snoozed_until: datetime | None = None
    origin_agent_key: AgentKey | None = None
    origin_conversation_id: UUID | None = None
    origin_run_id: UUID | None = None
    harness_session_id: str | None = None
    pending_requests: list[WorkPendingRequest] = Field(default_factory=list)
    error: str | None = None
    version: int
    created_at: datetime
    updated_at: datetime


class WorkRun(BaseModel):
    id: UUID
    thread_id: UUID
    ordinal: int
    client_request_id: str | None = None
    mode: Literal["send", "steer", "queue"] = "send"
    prompt: str
    status: WorkRunStatus
    owner_id: UUID | None = None
    lease_expires_at: datetime | None = None
    error: str | None = None
    final_message: str | None = None
    wakeup_delivered: bool = False
    version: int
    created_at: datetime
    updated_at: datetime


class WorkActivityItem(BaseModel):
    id: str
    thread_id: UUID
    position: int
    run_id: UUID | None = None
    kind: WorkActivityKind
    text: str = ""
    data: dict[str, JsonValue] = Field(default_factory=dict)
    created_at: datetime


class WorkThreadDetail(BaseModel):
    thread: WorkThread
    project: WorkProject | None = None
    runs: list[WorkRun] = Field(default_factory=list)
    activity: list[WorkActivityItem] = Field(default_factory=list)


class WorkThreadCreate(WorkModel):
    project_id: UUID
    title: str = Field(min_length=1, max_length=500)
    harness: WorkHarnessId = "codex"
    model: str | None = Field(default=None, max_length=256)
    model_options: dict[str, JsonValue] = Field(default_factory=dict)
    workspace_mode: WorkWorkspaceMode = "root"
    branch: str | None = Field(default=None, max_length=256)
    base_ref: str | None = Field(default=None, max_length=256)
    prompt: str | None = Field(default=None, max_length=200000)
    client_request_id: str | None = Field(default=None, max_length=256)

    @field_validator("title")
    @classmethod
    def nonblank_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Title cannot be blank")
        return value.strip()


class WorkThreadPatch(WorkModel):
    title: str | None = Field(default=None, min_length=1, max_length=500)
    harness: WorkHarnessId | None = None
    model: str | None = Field(default=None, max_length=256)
    model_options: dict[str, JsonValue] | None = None
    expected_version: int = Field(ge=1)


class WorkThreadOrganizeRequest(WorkModel):
    action: Literal["default", "snooze", "settle", "unsettle"]
    snoozed_until: datetime | None = None
    snooze_minutes: int | None = Field(default=None, ge=1, le=43200)
    expected_version: int | None = Field(default=None, ge=1)


class WorkRunCreate(WorkModel):
    prompt: str = Field(min_length=1, max_length=200000)
    mode: Literal["send", "steer", "queue"] = "send"
    client_request_id: str | None = Field(default=None, max_length=256)


class WorkPendingRespondRequest(WorkModel):
    response: dict[str, JsonValue]
    expected_version: int | None = Field(default=None, ge=1)


class WorkHarnessModelInfo(BaseModel):
    id: str
    label: str
    description: str = ""
    options: dict[str, list[str]] = Field(default_factory=dict)


class WorkHarnessAuthPublic(BaseModel):
    status: Literal[
        "ready",
        "sign_in_required",
        "starting",
        "waiting",
        "verifying",
        "unavailable",
        "error",
    ]
    login_method: Literal[
        "device_code", "oauth_code", "api_key", "browser", "local", "none"
    ] = "none"
    authorize_url: str | None = None
    user_code: str | None = None
    needs_code: bool = False
    account_label: str | None = None
    message: str = ""


class WorkHarnessPublic(BaseModel):
    id: str
    display_name: str
    description: str
    installed: bool
    auth: WorkHarnessAuthPublic
    models: list[WorkHarnessModelInfo] = Field(default_factory=list)
    default_model: str | None = None
    supports_steer: bool = False
    supports_resume: bool = True
    supports_approvals: bool = True
    supports_questions: bool = True


class WorkHarnessSignInSubmit(WorkModel):
    code: str | None = Field(default=None, max_length=4096)
    credential: str | None = Field(default=None, max_length=65536)
    account_label: str | None = Field(default=None, max_length=256)


class FilesystemEntry(BaseModel):
    name: str
    path: str
    is_git_repo: bool = False


class FilesystemBrowseResponse(BaseModel):
    current_path: str
    parent_path: str | None = None
    is_git_repo: bool = False
    default_branch: str | None = None
    entries: list[FilesystemEntry] = Field(default_factory=list)
