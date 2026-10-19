#!/usr/bin/env python3
"""The entry point for Hikari developer tooling."""

import subprocess
import sys
from pathlib import Path

import typer

# Direct execution must resolve the repository packages from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bin import assets, checks, dev, generate, inference, native, publishing

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
app.add_typer(generate.app, name="generate", help="Generate API schemas and clients.")
app.add_typer(
    native.app,
    name="native",
    help="Inspect native checkpoint metadata and memory plans.",
)
app.add_typer(
    inference.app,
    name="infer",
    help="Submit inference to the live worker through its API.",
)
app.add_typer(
    assets.app, name="assets", help="Export local checkpoint and tokenizer assets."
)
app.add_typer(
    dev.app,
    name="dev",
    help="Serve the API, or run one worker program rebuilt on source edits.",
)
app.add_typer(checks.app, name="check", help="Run formatting, builds, and unit checks.")
app.add_typer(
    publishing.app, name="pr", help="Run the restricted GitHub workflow publisher."
)


def main() -> None:
    try:
        app()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode if error.returncode > 0 else 128 - error.returncode)
    except (OSError, inference.InferenceError) as error:
        typer.echo(str(error), err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
