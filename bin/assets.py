"""Manual exports share the API's checkpoint serialization implementations."""

import json
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def whisper(checkpoint: Path, expected_sha256: str, destination: Path) -> None:
    """Convert a trusted local Whisper checkpoint with its expected SHA-256."""
    # Torch is an optional export dependency; other commands must not initialize it.
    from api.inference.whisper_export import export

    typer.echo(json.dumps(export(checkpoint, expected_sha256, destination), indent=2))


@app.command("whisper-assets")
def whisper_assets(vocabulary: int, destination: Path) -> None:
    """Export the pinned Whisper tokenizer and mel assets."""
    # Whisper and NumPy are needed only for this export.
    from api.inference.whisper_assets import export

    typer.echo(json.dumps(export(vocabulary, destination), indent=2))


@app.command("tts-tokenizer")
def tts_tokenizer(checkpoint: Path, destination: Path) -> None:
    """Export a local TTS tokenizer without downloading or running a model."""
    # Load the API's exporter only when explicitly preparing its assets.
    from api.inference.tts_assets import export

    export(checkpoint, destination)
