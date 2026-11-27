"""Read and edit the agent's single persistent persona."""

from typing import Protocol

from pydantic import Field, field_validator

from .base import Arguments, NoArguments, conversation_tool


class UpdatePersonaArguments(Arguments):
    markdown: str = Field(
        min_length=1,
        max_length=16000,
        pattern=r"\S",
        description="Full Markdown instructions for your persona.",
    )
    expected_version: int | None = Field(
        ge=1,
        strict=True,
        description="The version returned by read_persona, or null if no persona exists.",
    )
    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Optional label; omit to keep your current persona title.",
    )

    @field_validator("title")
    @classmethod
    def nonblank_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("Title cannot be blank")
        return value.strip()


class PersonaConversation(Protocol):
    async def read_persona(self) -> dict: ...

    async def update_persona(self, args: UpdatePersonaArguments) -> dict: ...


@conversation_tool
async def read_persona(args: NoArguments, conversation: PersonaConversation) -> dict:
    """Read your saved persona and its version before editing it."""
    return await conversation.read_persona()


@conversation_tool
async def update_persona(
    args: UpdatePersonaArguments, conversation: PersonaConversation
) -> dict:
    """Edit your saved persona for future responses, using the version from read_persona.

    Keep it grounded in the user's preferences. Saving does not replace this
    run's instructions. After a conflict or uncertain save, read before retrying.
    """
    return await conversation.update_persona(args)


TOOLS = [read_persona, update_persona]
