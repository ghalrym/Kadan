"""Bounded process-local HTTP measurements; no prompts, outputs or credentials stored."""
import asyncio
from collections import deque
from datetime import datetime, timezone
import json
from statistics import median
import threading
import time
from uuid import uuid4

from api.pydantic_models.requests import RequestRecord

GENERATION_PATHS = {
    '/v1/chat/completions': 'LLM', '/v1/decisions': 'Decision',
    '/v1/images/generations': 'Image', '/v1/images/edits': 'Image',
    '/v1/videos/generations': 'Video', '/v1/audio/speech': 'TTS',
    '/v1/audio/transcriptions': 'STT',
}
MAX_BODY_SUMMARY_BYTES = 16_384


class TelemetryStore:
    def __init__(self, capacity=1000):
        if capacity < 1:
            raise ValueError('Telemetry capacity must be positive')
        self.capacity = capacity
        self._records = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.active = self.completed = 0
        self._last_evicted = float('-inf')

    def begin(self):
        with self._lock:
            self.active += 1

    def finish(self, record, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            self.active -= 1
            self.completed += 1
            if len(self._records) == self.capacity:
                self._last_evicted = self._records[0][0]
            self._records.append((now, record))

    def records(self):
        with self._lock:
            return [record for _, record in reversed(self._records)]

    def metrics(self, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            recent = [record for stamp, record in self._records if stamp > now - 60]
            return dict(
                started_at=self.started_at, retention_limit=self.capacity,
                retained_requests=len(self._records), completed_requests=self.completed,
                active_requests=self.active, requests_per_minute=len(recent),
                window_truncated=self._last_evicted > now - 60,
                p50_latency_seconds=median(r.latency_ms for r in recent) / 1000 if recent else None,
                error_rate_percent=100 * sum(r.status >= 400 for r in recent) / len(recent) if recent else None,
            )


telemetry = TelemetryStore()


def summarize(body, size, complete):
    """Summarize structure only; never retain user text, uploaded media or arbitrary keys."""
    if size > MAX_BODY_SUMMARY_BYTES:
        return f'{size} request bytes; payload exceeds summary limit', None
    if not complete:
        return f'{size} request bytes received; body incomplete or not consumed', None
    try:
        value = json.loads(body)
    except (ValueError, UnicodeError, RecursionError):
        return f'{size} request bytes; non-JSON or malformed JSON', None
    if not isinstance(value, dict):
        return f'{size} request bytes; JSON {type(value).__name__}', None
    parts = [f'{size} request bytes', 'JSON object']
    messages = value.get('messages')
    if isinstance(messages, list):
        parts.append(f'{len(messages)} messages')
        chars = sum(len(m.get('text', '')) for m in messages
                    if isinstance(m, dict) and isinstance(m.get('text'), str))
        parts.append(f'{chars} message characters')
    for key in ('prompt', 'text', 'input', 'instruction', 'state'):
        if isinstance(value.get(key), str):
            parts.append(f'{key}: {len(value[key])} characters')
    # Model IDs can also contain user text. Keep only the curated public IDs.
    model = value.get('model')
    if not isinstance(model, str) or model not in ('small', 'medium', 'large'):
        model = None
    return '; '.join(parts), model


class TelemetryMiddleware:
    def __init__(self, app, store=None):
        self.app, self.store = app, store if store is not None else telemetry

    async def __call__(self, scope, receive, send):
        path = scope.get('path', '')
        if scope['type'] != 'http' or scope.get('method') != 'POST' or path not in GENERATION_PATHS:
            return await self.app(scope, receive, send)
        start = time.monotonic()
        timestamp = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
        body = bytearray()
        received = sent = 0
        status = 500
        complete = disconnected = response_complete = False
        self.store.begin()

        async def capture_receive():
            nonlocal received, complete, disconnected
            message = await receive()
            if message['type'] == 'http.request':
                chunk = message.get('body', b'')
                received += len(chunk)
                if len(body) < MAX_BODY_SUMMARY_BYTES:
                    body.extend(chunk[:MAX_BODY_SUMMARY_BYTES - len(body)])
                complete = not message.get('more_body', False)
            elif message['type'] == 'http.disconnect':
                disconnected = True
            return message

        async def capture_send(message):
            nonlocal status, sent, response_complete
            if message['type'] == 'http.response.start':
                status = message['status']
            elif message['type'] == 'http.response.body':
                sent += len(message.get('body', b''))
            await send(message)
            if message['type'] == 'http.response.body' and not message.get('more_body', False):
                response_complete = True

        try:
            await self.app(scope, capture_receive, capture_send)
        except asyncio.CancelledError:
            status = 499
            raise
        except Exception:
            status = 500
            raise
        finally:
            if disconnected and not response_complete:
                status = 499
            summary, model = summarize(body, received, complete)
            elapsed = max(0, (time.monotonic() - start) * 1000)
            self.store.finish(RequestRecord(
                id='req_' + uuid4().hex, time=timestamp, type=GENERATION_PATHS[path],
                model=model, status=status, latency=f'{elapsed / 1000:.3f} s',
                latency_ms=round(elapsed, 3), endpoint=path, prompt=summary,
                output=f'HTTP {status}; {sent} response bytes' + ('' if response_complete else '; response incomplete'),
                request_bytes=received, response_bytes=sent,
            ))
