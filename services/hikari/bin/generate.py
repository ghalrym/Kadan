"""API schema and frontend client generation."""

import json
from pathlib import Path

import typer

from bin.process import ROOT, run

app = typer.Typer(no_args_is_help=True)


@app.command()
def schema(output: Path = ROOT / "frontend/openapi.json") -> None:
    """Write the API's OpenAPI schema without starting the server."""
    # API dependencies are needed only for schema generation, not CLI help or other tools.
    from api.server import app as api

    output.write_text(
        json.dumps(api.openapi(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


@app.command()
def client() -> None:
    """Regenerate the TypeScript client from frontend/openapi.json."""
    run(["npm", "exec", "--", "openapi-ts"], cwd=ROOT / "frontend")
    run(
        [
            "npm",
            "exec",
            "--",
            "prettier",
            "--write",
            "--single-quote",
            "--no-semi",
            "src/api/generated",
        ],
        cwd=ROOT / "frontend",
    )


@app.command()
def api() -> None:
    """Generate the OpenAPI schema, then the frontend client."""
    schema()
    client()
