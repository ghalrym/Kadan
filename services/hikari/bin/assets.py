"""Offline asset conversion through the worker's C++ implementations."""

from pathlib import Path

import typer

from bin.native import DEFAULT_BUILD, execute

app = typer.Typer(no_args_is_help=True)


@app.command("speech-tokenizer")
def speech_tokenizer(checkpoint: Path, destination: Path) -> None:
    """Prepare a local speech tokenizer without downloading or running a model."""
    execute(
        DEFAULT_BUILD,
        "kadan-prepare-assets",
        ["speech-tokenizer", checkpoint, destination],
    )


@app.command("tts-tokenizer")
def tts_tokenizer(checkpoint: Path, destination: Path) -> None:
    """Prepare a local TTS tokenizer without downloading or running a model."""
    execute(
        DEFAULT_BUILD,
        "kadan-prepare-assets",
        ["tts-tokenizer", checkpoint, destination],
    )
