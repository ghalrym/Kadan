"""CPU Laya worker adapter; no Python model fallback or import-time load.

KADAN_NATIVE_DECISION_WORKER selects an executable (default
/opt/kadan/bin/kadan-decision-worker). KADAN_LAYA_MODEL selects a complete local
checkpoint or a pinned cached Hub snapshot; worker selection never downloads.
KADAN_NATIVE_DECISION_TIMEOUT_SECONDS defaults to 300. Parent admission defaults
to the child's 3 GiB model envelope plus 2 MiB transport/parent bookkeeping;
KADAN_NATIVE_DECISION_RAM_BYTES may increase it. These are conservative shared
reservations, not OS RSS limits. CPU weights remain resident until pressure or
cleanup stops and reaps the worker. The global MemoryManager queue owns order.
"""
import asyncio
import json
import math
import os
from pathlib import Path
import re
import selectors
import threading
import time

from huggingface_hub import snapshot_download

from api.inference.decisions.laya_python import DEFAULT_MODEL, DEFAULT_REVISION, parse_answer
from api.inference.cancellation import run_cancellable_thread
from api.inference.line_protocol import LineProtocolError, LineProtocolProcess, check_cancel
from api.inference.resources import ResourceBusy, ResourceExhausted
from api.services.model_downloads import model_manager
from api.inference.errors import InferenceFailure
from api.services.chat_runtime import chat_runtime

DEFAULT_HOST_BUDGET_BYTES = 3 * 1024**3 + 2 * 1024**2
MAX_FRAME_BYTES = 65536


def resolve_laya_command():
    binary = Path(os.getenv('KADAN_NATIVE_DECISION_WORKER', '/opt/kadan/bin/kadan-decision-worker')).expanduser()
    if not binary.is_absolute() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise InferenceFailure('Decision worker is unavailable; configure KADAN_NATIVE_DECISION_WORKER.')
    model = os.getenv('KADAN_LAYA_MODEL', DEFAULT_MODEL)
    root = Path(model).expanduser()
    if not root.is_dir():
        revision = os.getenv('KADAN_LAYA_REVISION', DEFAULT_REVISION if model == DEFAULT_MODEL else '')
        if not re.fullmatch('[0-9a-f]{40}', revision):
            raise InferenceFailure('KADAN_LAYA_REVISION must pin a commit SHA.')
        try:
            root = Path(snapshot_download(model, revision=revision,
                cache_dir=model_manager.root / 'hub', local_files_only=True))
        except Exception as error:
            raise InferenceFailure('Laya checkpoint is not cached; configure a complete local checkpoint.') from error
    for name in ('rl_agent_config.json', 'model.safetensors', 'tokenizer/tokenizer.json', 'encoder/config.json'):
        if not (root / name).is_file():
            raise InferenceFailure(f'Laya checkpoint is incomplete: missing {name}.')
    return [str(binary.resolve()), str(root.resolve())]


def decode_laya_response(frame):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise LineProtocolError('Duplicate worker response field')
            result[key] = value
        return result
    def constant(_):
        raise LineProtocolError('Nonfinite worker response')
    try:
        return json.loads(frame.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise LineProtocolError('Invalid decision JSON') from error


class LayaJsonlProcess(LineProtocolProcess):
    """Reuse owned process-group cleanup with bounded UTF-8 JSONL transport."""
    def exchange(self, request, timeout, cancel=None):
        check_cancel(cancel)
        if self.buffer or self.process.poll() is not None:
            raise LineProtocolError('Decision worker unavailable or sent unsolicited data')
        frame = request.encode('utf-8') + b'\n'
        if len(frame) > MAX_FRAME_BYTES + 1:
            raise InferenceFailure('Decision request exceeds its byte limit.', 422)
        deadline = time.monotonic() + timeout
        offset = 0
        os.set_blocking(self.process.stdin.fileno(), False)
        self.selector.register(self.process.stdin, selectors.EVENT_WRITE, 'in')
        while True:
            check_cancel(cancel)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Decision deadline expired')
            for key, _ in self.selector.select(min(.05, remaining)):
                try:
                    if key.data == 'in':
                        offset += os.write(key.fileobj.fileno(), frame[offset:offset + 4096])
                        if offset == len(frame):
                            self.selector.unregister(key.fileobj)
                        continue
                    data = os.read(key.fileobj.fileno(), 4096)
                except BlockingIOError:
                    continue
                if not data:
                    self.selector.unregister(key.fileobj)
                    if key.data == 'out':
                        raise LineProtocolError('Decision worker closed its response pipe')
                elif key.data == 'err':
                    self.diagnostics.extend(data)
                    del self.diagnostics[:-8192]
                else:
                    self.buffer.extend(data)
                    if len(self.buffer) > MAX_FRAME_BYTES + 1:
                        raise LineProtocolError('Decision response exceeds its byte limit')
            if b'\n' in self.buffer:
                line, _, extra = self.buffer.partition(b'\n')
                if extra or offset != len(frame):
                    raise LineProtocolError('Unexpected decision response ordering')
                self.buffer.clear()
                return decode_laya_response(line)


def parse_laya_responses(questions, response):
    if not isinstance(response, dict):
        raise LineProtocolError('Decision response must be an object')
    if set(response) == {'error'}:
        # Only request-domain rejections are client errors. Checkpoint, transport
        # and implementation failures never fall back to a Python model.
        client_errors = {'decision_literal_mask', 'decision_text_size', 'decision_empty_or_long_text',
                         'decision_state_token_budget',
                         'decision_head_token_budget', 'decision_instruction_token_budget',
                         'decision_option_token_budget', 'decision_request_size',
                         'decision_choice_options', 'decision_score_levels'}
        error = response['error']
        status = 422 if isinstance(error, str) and error in client_errors else 502
        raise InferenceFailure('Decision rejected the request: ' + str(error)[:200], status)
    if set(response) != {'answers'} or not isinstance(response['answers'], list) or len(response['answers']) != len(questions):
        raise LineProtocolError('Invalid decision answer count')
    result = []
    for question, answer in zip(questions, response['answers']):
        if (not isinstance(answer, dict) or set(answer) != {'key', 'type', 'value', 'confidence', 'probabilities'}
                or answer['key'] != question.key or answer['type'] != question.type
                or (question.type == 'Noul' and answer['probabilities'] is not None)):
            raise LineProtocolError('Invalid decision answer fields or order')
        result.append(parse_answer(question, dict(type=question.type.lower(),
            **{question.type.lower(): answer['value']}, answer_confidence=answer['confidence'],
            probabilities=answer['probabilities'])))
    return result


class LayaSubprocessEvaluator:
    def __init__(self, resources=None, resolve=resolve_laya_command, process_factory=LayaJsonlProcess):
        self.resources, self.resolve, self.process_factory = resources, resolve, process_factory
        self.agent = self.reservation = None
        self._lock = threading.Lock()
        self._generation = asyncio.Lock()
        self._owner = f'laya-subprocess:{id(self)}'
        self._quarantined = False

    def _settings(self):
        try:
            budget = int(os.getenv('KADAN_NATIVE_DECISION_RAM_BYTES', str(DEFAULT_HOST_BUDGET_BYTES)))
            timeout = float(os.getenv('KADAN_NATIVE_DECISION_TIMEOUT_SECONDS', '300'))
            if budget < DEFAULT_HOST_BUDGET_BYTES or not math.isfinite(timeout) or not 0 < timeout <= 3600:
                raise ValueError()
        except ValueError as error:
            raise InferenceFailure('Invalid decision RAM budget or timeout configuration.') from error
        return budget, timeout

    async def preflight(self):
        self._settings()
        await asyncio.to_thread(self.resolve)

    def check_execution_state(self):
        # A failed reap is unresolved execution, not ordinary memory pressure.
        # Keep this check nonblocking: only confirmed close clears quarantine.
        if self._quarantined:
            raise InferenceFailure('Decision cleanup is unconfirmed; close the runtime before executing another job.')

    def _close_locked(self):
        if self.agent is not None:
            try:
                self.agent.stop()
            except BaseException:
                self._quarantined = True
                raise
            self.agent = None
        if self.reservation is not None:
            self.reservation.release()
            self.reservation = None
        self._quarantined = False

    def _evict(self):
        if not self._lock.acquire(blocking=False):
            raise ResourceBusy('Decisions are active')
        try:
            self._close_locked()
        finally:
            self._lock.release()

    def _run(self, state, questions, cancel):
        with self._lock:
            if self._quarantined:
                raise InferenceFailure('Decision cleanup is unconfirmed; restart or close the runtime.')
            budget, timeout = self._settings()
            argv = self.resolve()
            self.resources = self.resources or chat_runtime.ensure_resources()
            try:
                if self.reservation is None:
                    self.reservation = self.resources.reserve(self._owner, 'decision', host_bytes=budget,
                        evict=self._evict, cancel_event=cancel)
                with self.reservation.lease(cancel):
                    if self.agent is None:
                        self.agent = self.process_factory()
                        self.agent.start(argv)
                    request = json.dumps(dict(state=state,
                        questions=[q.model_dump(mode='json', by_alias=True) for q in questions]),
                        ensure_ascii=False, allow_nan=False, separators=(',', ':'))
                    result = parse_laya_responses(questions, self.agent.exchange(request, timeout, cancel))
                    check_cancel(cancel)
                    return result
            except BaseException as error:
                # Leave the active lease before release. Child exit must be
                # confirmed first; a failed reap deliberately keeps admission.
                self._close_locked()
                if isinstance(error, (ResourceBusy, ResourceExhausted)):
                    raise InferenceFailure(str(error), 503) from error
                if isinstance(error, TimeoutError):
                    raise InferenceFailure('Decision worker timed out.', 504) from error
                if isinstance(error, (LineProtocolError, OSError)):
                    raise InferenceFailure('Decision worker failed: ' + str(error), 502) from error
                raise

    async def evaluate(self, state, questions):
        if self._generation.locked():
            raise InferenceFailure('Decisions are active.', 429)
        async with self._generation:
            return await run_cancellable_thread(self._run, state, questions)

    async def load(self):
        # Model loading is owned by the first admitted worker request.
        await self.preflight()

    def offload_to_ram(self, cancel=None):
        check_cancel(cancel)
        if self._generation.locked():
            raise ResourceBusy('Decisions are active')
        if self._quarantined:
            raise InferenceFailure('Decision cleanup is unconfirmed.')
        # This CPU executor never acquires device residency.

    async def close(self):
        async with self._generation:
            await run_cancellable_thread(self._close)

    def _close(self, cancel):
        with self._lock:
            self._close_locked()
