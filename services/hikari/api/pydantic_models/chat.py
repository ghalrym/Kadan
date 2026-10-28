from typing import Literal
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue, model_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    text: str = Field(min_length=1)
    meta: str | None = None


class Lfm3Function(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_]+$")
    description: str = ""
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    strict: bool | None = None


class Lfm3Tool(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["function"] = "function"
    function: Lfm3Function


class Lfm3CallFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_]+$")
    arguments: dict[str, JsonValue]


class Lfm3ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=120)
    type: Literal["function"] = "function"
    function: Lfm3CallFunction


class Lfm3Message(BaseModel):
    """Structured history consumed only by the explicit experimental checkpoint."""

    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant", "tool"]
    text: str = Field(default="", validation_alias=AliasChoices("text", "content"))
    tool_calls: list[Lfm3ToolCall] | None = None
    tool_call_id: str | None = Field(default=None, min_length=1, max_length=120)

    @model_validator(mode="after")
    def valid_role_fields(self):
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError("Only assistant messages may contain tool calls")
        if (self.tool_call_id is not None) != (self.role == "tool"):
            raise ValueError("Tool results require their tool call ID")
        if not self.text and not self.tool_calls and self.role != "tool":
            raise ValueError("A message requires text or an assistant tool call")
        return self


def lfm3_messages(values):
    """Select the rich message schema only for this user-selected experiment."""
    if isinstance(values, dict) and (values.get("model_id") or values.get("model")) == "lfm3-1.7b-bf16":
        values = {**values, "messages": [Lfm3Message.model_validate(row.model_dump() if isinstance(row, BaseModel) else row) for row in values.get("messages", [])]}
    return values


def require_lfm3_tools(model, messages, tools):
    if model != "lfm3-1.7b-bf16" and (tools is not None or any(isinstance(row, Lfm3Message) for row in messages)):
        raise ValueError("Structured tools and tool history require the explicit experimental LFM3 text model")
