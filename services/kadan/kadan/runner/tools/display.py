"""Display tools: record the computer's whole desktop."""

from typing import Literal

from pydantic import Field

from ...api.sandbox_client import SandboxClient
from .base import Arguments, body, computer_tool


class RecordArguments(Arguments):
    action: Literal["start", "stop", "status"] = Field(description="Recording action.")
    save_path: str | None = Field(
        default=None,
        description="Output file; defaults to /tmp/recordings/recording_<time>.mp4.",
    )
    fps: int = Field(default=5, description="Frames per second.")
    crf: int = Field(
        default=28, description="H.264 quality: 0 is lossless, 51 is worst."
    )
    max_duration: float = Field(default=600.0, description="Maximum seconds.")
    width: int | None = Field(default=None, description="Video width in pixels.")
    height: int | None = Field(default=None, description="Video height in pixels.")


@computer_tool("POST")
async def display_record(args: RecordArguments, computer: SandboxClient) -> dict:
    """Start, stop or check a recording of the whole desktop: browser, tabs, popups
    and terminal."""
    return await computer.run("POST", "/v1/display/record", json_body=body(args))


TOOLS = [
    # display_record,
]
