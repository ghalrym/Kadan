"""Tools for checking and installing Kadan application updates."""

from typing import Protocol

from pydantic import Field

from .base import Arguments, NoArguments, conversation_tool


class InstallAppUpdateArguments(Arguments):
    announcement: str = Field(
        min_length=1,
        max_length=4000,
        pattern=r"\S",
        description="Brief plain-language message delivered to the user right before "
        "Kadan restarts to apply the update.",
    )
    target_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="Optional release version returned by check_app_update.",
    )


class UpdateConversation(Protocol):
    async def check_app_update(self) -> dict: ...

    async def install_app_update(self, args: InstallAppUpdateArguments) -> dict: ...


@conversation_tool
async def check_app_update(args: NoArguments, conversation: UpdateConversation) -> dict:
    """Check whether a newer version of Kadan is available for the server or connected desktop app."""
    return await conversation.check_app_update()


@conversation_tool
async def install_app_update(
    args: InstallAppUpdateArguments, conversation: UpdateConversation
) -> dict:
    """Download, verify, and install the latest Kadan update, deliver the announcement, and restart the app.

    Only call this after the user has asked to update or confirmed your offer to update.
    """
    return await conversation.install_app_update(args)


TOOLS = [
    check_app_update,
    install_app_update,
]
