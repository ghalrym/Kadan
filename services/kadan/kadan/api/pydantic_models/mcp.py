"""Pydantic models for global Remote MCP connections and T3 Code thread watches."""

from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _validate_http_url(value: str) -> str:
    cleaned = value.strip()
    parsed = urlsplit(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Use a valid http:// or https:// URL.")
    if parsed.port == 0:
        raise ValueError("Port must be between 1 and 65535.")
    return cleaned.rstrip("/")


class McpToolInfo(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    read_only: bool = False
    idempotent: bool = False
    accepts_client_request_id: bool = False


class McpAuthState(BaseModel):
    bearer_token: str | None = Field(default=None, max_length=16000)
    client_id: str | None = Field(default=None, max_length=2000)
    client_secret: str | None = Field(default=None, max_length=8000)
    access_token: str | None = Field(default=None, max_length=16000)
    refresh_token: str | None = Field(default=None, max_length=16000)
    token_type: str | None = Field(default=None, max_length=100)
    expires_at: datetime | None = None
    authorization_endpoint: str | None = Field(default=None, max_length=2000)
    token_endpoint: str | None = Field(default=None, max_length=2000)
    registration_endpoint: str | None = Field(default=None, max_length=2000)
    scope: str | None = Field(default=None, max_length=1000)


class McpServerRecord(BaseModel):
    """Full stored MCP server record from Tsuchi (includes secrets; never send to browser)."""

    id: UUID
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,47}$")
    name: str = Field(min_length=1, max_length=500)
    preset: Literal["t3_code", "custom"] = "custom"
    url: str = Field(min_length=1, max_length=2000)
    app_url: str | None = Field(default=None, max_length=2000)
    auth_type: Literal["oauth", "bearer", "none"] = "oauth"
    auth_state: McpAuthState = Field(default_factory=McpAuthState)
    enabled: bool = True
    status: Literal["disconnected", "connected", "error"] = "disconnected"
    status_error: str | None = Field(default=None, max_length=4000)
    pinned_tools: list[str] = Field(default_factory=list, max_length=200)
    tools: list[McpToolInfo] = Field(default_factory=list, max_length=500)
    tools_synced_at: datetime | None = None
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime

    def to_public(self) -> "McpServerPublic":
        return McpServerPublic(
            id=self.id,
            slug=self.slug,
            name=self.name,
            preset=self.preset,
            url=self.url,
            app_url=self.app_url,
            auth_type=self.auth_type,
            has_credentials=bool(
                self.auth_state.access_token or self.auth_state.bearer_token
            ),
            enabled=self.enabled,
            status=self.status,
            status_error=self.status_error,
            pinned_tools=list(self.pinned_tools),
            tools=[
                McpToolSummary(
                    name=tool.name,
                    description=tool.description,
                    read_only=tool.read_only,
                    idempotent=tool.idempotent,
                )
                for tool in self.tools
            ],
            tools_synced_at=self.tools_synced_at,
            version=self.version,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class McpToolSummary(BaseModel):
    name: str
    description: str
    read_only: bool = False
    idempotent: bool = False


class McpServerPublic(BaseModel):
    """Browser-safe MCP server representation with all credentials redacted."""

    id: UUID
    slug: str
    name: str
    preset: Literal["t3_code", "custom"]
    url: str
    app_url: str | None
    auth_type: Literal["oauth", "bearer", "none"]
    has_credentials: bool
    enabled: bool
    status: Literal["disconnected", "connected", "error"]
    status_error: str | None
    pinned_tools: list[str]
    tools: list[McpToolSummary]
    tools_synced_at: datetime | None
    version: int
    created_at: datetime
    updated_at: datetime


class McpServerCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    url: str = Field(min_length=1, max_length=2000)
    preset: Literal["t3_code", "custom"] = "custom"
    slug: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]{0,47}$")
    auth_type: Literal["oauth", "bearer", "none"] = "oauth"
    bearer_token: str | None = Field(default=None, max_length=16000)
    enabled: bool = True
    pinned_tools: list[str] | None = Field(default=None, max_length=200)

    @field_validator("name")
    @classmethod
    def nonblank_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Enter a name for this integration.")
        return cleaned

    @field_validator("url")
    @classmethod
    def valid_url(cls, value: str) -> str:
        return _validate_http_url(value)


class McpServerPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(ge=1)
    name: str | None = Field(default=None, min_length=1, max_length=200)
    url: str | None = Field(default=None, min_length=1, max_length=2000)
    auth_type: Literal["oauth", "bearer", "none"] | None = None
    bearer_token: str | None = Field(default=None, max_length=16000)
    enabled: bool | None = None
    pinned_tools: list[str] | None = Field(default=None, max_length=200)

    @field_validator("name")
    @classmethod
    def nonblank_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Enter a name for this integration.")
        return cleaned

    @field_validator("url")
    @classmethod
    def valid_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_http_url(value)


class McpConnectResponse(BaseModel):
    server: McpServerPublic
    authorize_url: str | None = None


class McpOAuthStateRecord(BaseModel):
    state: str
    server_id: UUID
    code_verifier: str
    redirect_uri: str
    token_endpoint: str
    client_id: str
    client_secret: str | None = None
    expires_at: datetime
    created_at: datetime


class T3ThreadWatch(BaseModel):
    id: UUID
    server_id: UUID
    thread_id: str
    run_id: str | None = None
    environment_id: str | None = None
    thread_link: str | None = None
    thread_title: str | None = None
    origin_agent_key: str
    origin_conversation_id: UUID
    origin_run_id: UUID
    status: Literal["watching", "delivered", "dropped"] = "watching"
    last_pending_request_id: str | None = None
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
