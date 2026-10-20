"""Utility tools: document conversion on the computer."""

from pydantic import Field

from ...api.sandbox_client import SandboxClient
from .base import Arguments, body, computer_tool


class ConvertArguments(Arguments):
    uri: str = Field(description="URL or file:// path of the document to convert.")


@computer_tool("POST")
async def util_convert_to_markdown(
    args: ConvertArguments, computer: SandboxClient
) -> dict:
    """Convert a document (web page, PDF, Office file and so on) to Markdown."""
    return await computer.run(
        "POST", "/v1/util/convert_to_markdown", json_body=body(args)
    )


TOOLS = [
    # util_convert_to_markdown,
]
