from urllib.parse import urlsplit

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


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
    """Keep the unauthenticated prototype local and reject cross-origin writes."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        host = headers.get(b"host", b"").decode("latin-1")
        try:
            parsed = urlsplit(f"http://{host}")
            allowed = (
                parsed.hostname in {"127.0.0.1", "localhost", "::1"}
                and parsed.username is None
                and parsed.password is None
            )
            if parsed.port == 0:
                allowed = False
            if parsed.path or parsed.query or parsed.fragment:
                allowed = False
        except ValueError:
            allowed = False
        if not allowed:
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
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = -1
        if declared < 0 or declared > 1024 * 1024:
            await JSONResponse(
                {"detail": "Request body must be at most 1 MiB"}, status_code=413
            )(scope, receive, send)
            return
        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > 1024 * 1024:
                    raise HTTPException(413, "Request body must be at most 1 MiB")
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
