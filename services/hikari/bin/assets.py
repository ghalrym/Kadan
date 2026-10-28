"""Offline asset conversion through the worker's C++ implementations."""

from pathlib import Path

import typer

from bin.native import DEFAULT_BUILD, execute

app = typer.Typer(no_args_is_help=True)


@app.command()
def whisper(checkpoint: Path, expected_sha256: str, destination: Path) -> None:
    """Convert a verified local Whisper checkpoint without executing a model."""
    execute(
        DEFAULT_BUILD,
        "kadan-prepare-assets",
        ["whisper", checkpoint, expected_sha256, destination],
    )


@app.command("whisper-assets")
def whisper_assets(vocabulary: int, destination: Path) -> None:
    """Prepare the pinned Whisper tokenizer and mel assets in C++."""
    execute(
        DEFAULT_BUILD,
        "kadan-prepare-assets",
        ["whisper-assets", vocabulary, destination],
    )
