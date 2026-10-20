import re
from urllib.parse import urlsplit

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MAX_BODY_BYTES = 1024 * 1024
# Pet packages carry a full spritesheet; pet_import checks each file against 20 MB.
LARGE_BODY_PATHS = re.compile(r"/api/agents/[^/]+/pets/imports")
MAX_LARGE_BODY_BYTES = 64 * 1024 * 1024


def body_limit(path: str) -> int:
    return MAX_LARGE_BODY_BYTES if LARGE_BODY_PATHS.fullmatch(path) else MAX_BODY_BYTES


def same_origin(origin: str, host: str, scheme: str) -> bool:
    try:
        supplied, expected = urlsplit(origin), urlsplit(f"{scheme}://{host}")
        if (
            supplied.scheme not in {"http", "https"}
            or supplied.username
            or supplied.password
        ):
            return False
        if supplied.path or supplied.query or supplied.fragment:
            return False
        supplied_port = supplied.port or (443 if supplied.scheme == "https" else 80)
        expected_port = expected.port or (443 if expected.scheme == "https" else 80)
        return (supplied.scheme, supplied.hostname, supplied_port) == (
            expected.scheme,
            expected.hostname,
            expected_port,
        )
    except ValueError:
        return False


class LocalRequestSafety:
    """Restrict request hosts and reject cross-origin writes."""

    def __init__(self, app: ASGIApp, allowed_hosts: list[str]):
        self.app = app
        self.allowed_hosts = {host.lower().strip("[]") for host in allowed_hosts}

    def allowed_host(self, host: str) -> bool:
        try:
            parsed = urlsplit(f"http://{host}")
            return (
                parsed.hostname in self.allowed_hosts
                and parsed.username is None
                and parsed.password is None
                and parsed.port != 0
                and not (parsed.path or parsed.query or parsed.fragment)
            )
        except ValueError:
            return False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await self.websocket(scope, receive, send)
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        host = headers.get(b"host", b"").decode("latin-1")
        if not self.allowed_host(host):
            await JSONResponse(
                {"detail": "Host is not allowed for this local service"},
                status_code=400,
            )(scope, receive, send)
            return
        if scope["method"] not in {"GET", "HEAD", "OPTIONS"}:
            origin = headers.get(b"origin")
            if (
                origin is not None
                and not same_origin(
                    origin.decode("latin-1"), host, scope.get("scheme", "http")
                )
            ) or headers.get(b"sec-fetch-site") == b"cross-site":
                await JSONResponse(
                    {"detail": "Cross-origin writes are not allowed"}, status_code=403
                )(scope, receive, send)
                return
        limit = body_limit(scope.get("path", ""))
        too_large = f"Request body must be at most {limit // (1024 * 1024)} MiB"
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = -1
        if declared < 0 or declared > limit:
            await JSONResponse({"detail": too_large}, status_code=413)(
                scope, receive, send
            )
            return
        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(413, too_large)
            return message

        async def secure_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                message.setdefault("headers", []).extend(
                    [
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"referrer-policy", b"same-origin"),
                    ]
                )
            await send(message)

        try:
            await self.app(scope, limited_receive, secure_send)
        except HTTPException as error:
            if started:
                raise
            await JSONResponse({"detail": error.detail}, status_code=error.status_code)(
                scope, receive, secure_send
            )

    async def websocket(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Browsers let any page open a WebSocket to localhost; CORS never applies.
        # Require this exact origin so other sites cannot drive the computer.
        headers = dict(scope["headers"])
        host = headers.get(b"host", b"").decode("latin-1")
        origin = headers.get(b"origin", b"").decode("latin-1")
        scheme = "https" if scope.get("scheme") == "wss" else "http"
        if not self.allowed_host(host) or not same_origin(origin, host, scheme):
            await send({"type": "websocket.close", "code": 1008})
            return
        await self.app(scope, receive, send)
