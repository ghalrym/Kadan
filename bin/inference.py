"""Submit inference to the live API and its existing native worker queue."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Annotated
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import typer

app = typer.Typer(no_args_is_help=True)


class InferenceError(RuntimeError):
    """A live API request failed; inference must remain with the server."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass
class LiveAPI:
    url: str
    timeout: float

    def request(self, method: str, path: str, payload: dict | None = None) -> bytes:
        body = (
            None if payload is None else json.dumps(payload, allow_nan=False).encode()
        )
        request = Request(
            self.url.rstrip("/") + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        try:
            # Never redirect or retry submissions: a lost reply may already be queued.
            with build_opener(NoRedirect()).open(
                request, timeout=self.timeout
            ) as response:
                return response.read()
        except HTTPError as error:
            with error:
                detail = error.read(65536).decode("utf-8", errors="replace")
            raise InferenceError(f"API returned HTTP {error.code}: {detail}") from None
        except (URLError, TimeoutError, OSError) as error:
            detail = (
                " Check server status before resubmitting; the request may already be queued."
                if method == "POST"
                else ""
            )
            raise InferenceError(f"Live API request failed: {error}.{detail}") from None


@app.callback()
def connection(
    ctx: typer.Context,
    url: Annotated[str, typer.Option(envvar="KADAN_API_URL")] = "http://127.0.0.1:8000",
    timeout: Annotated[float, typer.Option(min=0.1)] = 3600,
) -> None:
    """Use the running API; timeout includes queueing and generation time."""
    parsed = urlsplit(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise typer.BadParameter("Use an HTTP(S) API URL without a query or fragment.")
    ctx.obj = LiveAPI(url, timeout)


def submit(
    api: LiveAPI, endpoint: str, request_file: Path, output: Path | None
) -> None:
    try:
        payload = json.loads(request_file.read_text())
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON request object.")
        json.dumps(payload, allow_nan=False)
    except (ValueError, UnicodeError) as error:
        raise typer.BadParameter(f"Invalid request file: {error}") from error
    result = api.request("POST", endpoint, payload)
    if output is None:
        typer.echo(result.decode("utf-8"))
    else:
        output.write_bytes(result)


@app.command()
def chat(ctx: typer.Context, request_file: Path, output: Path | None = None) -> None:
    """Submit a chat completion JSON request; print/save the completed response."""
    submit(ctx.obj, "/v1/chat/completions", request_file, output)


@app.command()
def image(ctx: typer.Context, request_file: Path, output: Path | None = None) -> None:
    """Submit an image generation JSON request; results contain base64 PNGs."""
    submit(ctx.obj, "/v1/images/generations", request_file, output)


@app.command()
def video(ctx: typer.Context, request_file: Path, output: Path | None = None) -> None:
    """Queue a video JSON request and return its job ID; acceptance is not completion."""
    submit(ctx.obj, "/v1/videos/generations", request_file, output)


@app.command()
def speech(ctx: typer.Context, request_file: Path, output: Path | None = None) -> None:
    """Submit a speech generation JSON request; results contain base64 WAV audio."""
    submit(ctx.obj, "/v1/audio/speech", request_file, output)


@app.command()
def transcribe(
    ctx: typer.Context, request_file: Path, output: Path | None = None
) -> None:
    """Submit transcription JSON containing a base64 PCM WAV data URL."""
    submit(ctx.obj, "/v1/audio/transcriptions", request_file, output)


@app.command()
def decisions(
    ctx: typer.Context, request_file: Path, output: Path | None = None
) -> None:
    """Submit a decision JSON request containing state and typed questions."""
    submit(ctx.obj, "/v1/decisions", request_file, output)


@app.command("video-status")
def video_status(ctx: typer.Context, job_id: str) -> None:
    """Read a video's queue/progress/completion state from the live API."""
    typer.echo(ctx.obj.request("GET", f"/v1/videos/{quote(job_id, safe='')}").decode())


@app.command("video-cancel")
def video_cancel(ctx: typer.Context, job_id: str) -> None:
    """Ask the live worker to cancel a queued or running video job."""
    typer.echo(
        ctx.obj.request("DELETE", f"/v1/videos/{quote(job_id, safe='')}").decode()
    )


@app.command("video-download")
def video_download(ctx: typer.Context, job_id: str, output: Path) -> None:
    """Download a completed job's MP4; unfinished jobs return an API error."""
    content = ctx.obj.request("GET", f"/v1/videos/{quote(job_id, safe='')}/content")
    output.write_bytes(content)
