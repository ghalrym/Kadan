"""Standard Webhooks HMAC-SHA256 signing and payload helpers for MCP Events."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from typing import Any
from urllib.parse import urlsplit

WEBHOOK_SECRET_PREFIX = "whsec_"
MIN_SECRET_BYTES = 16
MAX_WEBHOOK_PAYLOAD_BYTES = 128 * 1024  # Well under the 256 KiB MCP Events limit.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def decode_whsec_secret(signing_secret: str) -> bytes:
    """Validate and base64-decode a Standard Webhooks `whsec_...` signing secret."""
    if not isinstance(signing_secret, str) or not signing_secret.startswith(
        WEBHOOK_SECRET_PREFIX
    ):
        raise ValueError("Signing secret must start with 'whsec_'.")
    encoded = signing_secret[len(WEBHOOK_SECRET_PREFIX) :].strip()
    if not encoded:
        raise ValueError("Signing secret cannot be empty.")
    # Pad if unpadded base64 was supplied.
    padded = encoded + "=" * ((4 - len(encoded) % 4) % 4)
    try:
        raw = base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(
            "Signing secret must be valid base64 after 'whsec_'."
        ) from error
    if len(raw) < MIN_SECRET_BYTES:
        raise ValueError(
            f"Signing secret must decode to at least {MIN_SECRET_BYTES} bytes."
        )
    return raw


def validate_webhook_callback_url(callback_url: str) -> str:
    """Require an HTTPS callback URL (allowing plain HTTP only for loopback hosts)."""
    cleaned = callback_url.strip()
    parsed = urlsplit(cleaned)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.netloc
        or not parsed.hostname
    ):
        raise ValueError("Callback URL must be a valid HTTPS URL.")
    if parsed.scheme == "http" and parsed.hostname.lower() not in LOOPBACK_HOSTS:
        raise ValueError("Callback URL must use HTTPS.")
    if parsed.username or parsed.password:
        raise ValueError("Callback URL must not embed credentials.")
    return cleaned


def sign_standard_webhook(
    signing_secret: str,
    webhook_id: str,
    timestamp: int,
    payload_bytes: bytes,
) -> str:
    """Compute a Standard Webhooks `v1,<base64>` HMAC-SHA256 signature."""
    key = decode_whsec_secret(signing_secret)
    signed_content = f"{webhook_id}.{timestamp}.".encode("utf-8") + payload_bytes
    digest = hmac.new(key, signed_content, hashlib.sha256).digest()
    return "v1," + base64.b64encode(digest).decode("ascii")


def build_webhook_headers(
    *,
    subscription_id: str,
    event_id: str,
    timestamp: int,
    signing_secret: str,
    payload_bytes: bytes,
) -> dict[str, str]:
    signature = sign_standard_webhook(
        signing_secret, event_id, timestamp, payload_bytes
    )
    return {
        "Content-Type": "application/json",
        "webhook-id": event_id,
        "webhook-timestamp": str(timestamp),
        "webhook-signature": signature,
        "X-MCP-Subscription-Id": subscription_id,
    }


def encode_bounded_webhook_payload(
    *,
    event_id: str,
    name: str,
    timestamp_iso: str,
    cursor: str,
    data: dict[str, Any],
    max_bytes: int = MAX_WEBHOOK_PAYLOAD_BYTES,
) -> bytes:
    """Serialize an MCP Events occurrence payload, trimming context if needed to stay bounded."""
    envelope: dict[str, Any] = {
        "eventId": event_id,
        "name": name,
        "timestamp": timestamp_iso,
        "cursor": cursor,
        "data": dict(data),
    }
    raw = json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    if len(raw) <= max_bytes:
        return raw

    # Trim recent_messages first, then message content, so envelope metadata stays intact.
    trimmed_data = dict(data)
    if isinstance(trimmed_data.get("recent_messages"), list):
        recent = list(trimmed_data["recent_messages"])
        while recent and len(raw) > max_bytes:
            recent.pop(0)
            trimmed_data["recent_messages"] = recent
            envelope["data"] = trimmed_data
            raw = json.dumps(
                envelope, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")

    if len(raw) > max_bytes and isinstance(trimmed_data.get("message"), dict):
        message = dict(trimmed_data["message"])
        content = str(message.get("content", ""))
        overflow = len(raw) - max_bytes + 64
        if len(content) > overflow:
            message["content"] = content[: max(0, len(content) - overflow)] + "…"
        else:
            message["content"] = content[:256]
        trimmed_data["message"] = message
        envelope["data"] = trimmed_data
        raw = json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    return raw
