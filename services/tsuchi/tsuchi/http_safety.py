from urllib.parse import urlsplit

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


def allowed_request_origin(origin: str, host: str, scheme: str) -> bool:
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
    """Local Host/Origin checks and bounded bodies, without an account system."""

    def __init__(self, app: ASGIApp, allowed_hosts: list[str], max_body_bytes: int):
        self.app = app
        self.allowed_hosts = {host.lower().strip("[]") for host in allowed_hosts}
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        host = headers.get(b"host", b"").decode("latin-1")
        try:
            parsed_host = urlsplit(f"http://{host}")
            host_is_allowed = (
                parsed_host.hostname in self.allowed_hosts
                and not parsed_host.username
                and not parsed_host.password
            )
            # Accessing port rejects malformed numbers and out-of-range values.
            parsed_host.port
            if parsed_host.path or parsed_host.query or parsed_host.fragment:
                host_is_allowed = False
        except ValueError:
            host_is_allowed = False
        if not host_is_allowed:
            await JSONResponse(
                {"detail": "Host is not allowed for this local service"},
                status_code=400,
            )(scope, receive, send)
            return
        if scope["method"] not in {"GET", "HEAD", "OPTIONS"}:
            origin = headers.get(b"origin")
            if origin is not None and not allowed_request_origin(
                origin.decode("latin-1"), host, scope.get("scheme", "http")
            ):
                await JSONResponse(
                    {"detail": "Cross-origin writes are not allowed"}, status_code=403
                )(scope, receive, send)
                return
            if headers.get(b"sec-fetch-site") == b"cross-site":
                await JSONResponse(
                    {"detail": "Cross-site writes are not allowed"}, status_code=403
                )(scope, receive, send)
                return
        try:
            declared_length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared_length = -1
        if declared_length < 0:
            await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(
                scope, receive, send
            )
            return
        if declared_length > self.max_body_bytes:
            await JSONResponse(
                {"detail": "Request body exceeds the upload limit"}, status_code=413
            )(scope, receive, send)
            return
        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise HTTPException(413, "Request body exceeds the upload limit")
            return message

        async def secure_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
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
            if response_started:
                raise
            await JSONResponse({"detail": error.detail}, status_code=error.status_code)(
                scope, receive, secure_send
            )
