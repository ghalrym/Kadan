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
        default=None,
        ge=1,
        strict=True,
        description=(
            "Optional version from read_persona for strict conflict checking. "
            "Omit to save in a single call using the latest saved version."
        ),
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


class RenameSelfArguments(Arguments):
    name: str = Field(
        min_length=1,
        max_length=500,
        description="Your new display name.",
    )

    @field_validator("name")
    @classmethod
    def nonblank_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Name cannot be blank")
        return value.strip()


class PersonaConversation(Protocol):
    async def read_persona(self) -> dict: ...

    async def update_persona(self, args: UpdatePersonaArguments) -> dict: ...

    async def rename_self(self, args: RenameSelfArguments) -> dict: ...


@conversation_tool
async def read_persona(args: NoArguments, conversation: PersonaConversation) -> dict:
    """Read your saved persona and its current version."""
    return await conversation.read_persona()


@conversation_tool
async def update_persona(
    args: UpdatePersonaArguments, conversation: PersonaConversation
) -> dict:
    """Update your saved persona in a single call for future responses.

    Omit expected_version to save directly against the latest version, or pass
    expected_version from read_persona for strict conflict checking. Keep it
    grounded in the user's preferences. Saving does not replace this run's
    instructions. After a conflict or uncertain save, read before retrying.
    """
    return await conversation.update_persona(args)


@conversation_tool
async def rename_self(
    args: RenameSelfArguments, conversation: PersonaConversation
) -> dict:
    """Update your own display name, such as when the user gives you a name."""
    return await conversation.rename_self(args)


TOOLS = [read_persona, update_persona, rename_self]
