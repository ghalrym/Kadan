"""Skill tools: instruction packages stored on the computer."""

from pydantic import Field

from ...api.sandbox_client import SandboxClient, segment
from .base import Arguments, NoArguments, body, computer_tool


class RegisterArguments(Arguments):
    path: str | None = Field(
        default=None, description="Skill directory or archive on the computer."
    )
    name: str | None = Field(default=None, description="Skill name.")
    file: str | None = Field(
        default=None, description="Text content of a skill file to upload."
    )


@computer_tool("POST")
async def skills_register_skills(
    args: RegisterArguments, computer: SandboxClient
) -> dict:
    """Register skills from a path on the computer or an uploaded file."""
    return await computer.run(
        "POST",
        "/v1/skills/register",
        data=args.model_dump(exclude={"file"}, exclude_none=True) or None,
        files={"file": ("file", args.file.encode())} if args.file else None,
    )


class MetadataArguments(Arguments):
    names: str | None = Field(
        default=None, description="Comma-separated skill names; all if omitted."
    )


@computer_tool("GET")
async def skills_list_skills_metadata(
    args: MetadataArguments, computer: SandboxClient
) -> dict:
    """List registered skills' metadata."""
    return await computer.run("GET", "/v1/skills/metadatas", params=body(args))


class SkillArguments(Arguments):
    name: str = Field(description="Skill name.")


@computer_tool("GET")
async def skills_get_skill_content(
    args: SkillArguments, computer: SandboxClient
) -> dict:
    """Read a skill's content."""
    return await computer.run("GET", f"/v1/skills/{segment(args.name)}/content")


@computer_tool("DELETE")
async def skills_delete_skill(args: SkillArguments, computer: SandboxClient) -> dict:
    """Delete a skill."""
    return await computer.run("DELETE", f"/v1/skills/{segment(args.name)}")


@computer_tool("DELETE")
async def skills_clear_skills(args: NoArguments, computer: SandboxClient) -> dict:
    """Delete every skill."""
    return await computer.run("DELETE", "/v1/skills")


TOOLS = [
    # skills_register_skills,
    # skills_list_skills_metadata,
    # skills_get_skill_content,
    # skills_delete_skill,
    # skills_clear_skills,
]
