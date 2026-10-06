"""Bounded Redis FIFO consumed by one task in the API process.

The kernel lock lasts through native cleanup, including a Redis outage. A Redis
namespace is bound to that lock file's persistent identity: processes using a
separate model volume cannot silently become a second consumer. Do not unlink
or replace the lock file while the API is running.
"""
import asyncio
from contextlib import suppress
import fcntl
import json
import logging
import os
from pathlib import Path
import uuid

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from redis.asyncio import Redis
from redis.exceptions import RedisError

from api.services.runtime import RuntimeFailure, finish_cleanup

log = logging.getLogger(__name__)


class Job(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(pattern=r'^[a-f0-9]{32}$')
    feature: str = Field(pattern=r'^(llm|video|image|stt|tts|decisions)$')
    operation: str = Field(min_length=1, max_length=32)
    model: str | None = Field(default=None, max_length=256)
    payload: dict[str, JsonValue]


_ENQUEUE = """
if redis.call('SCARD', KEYS[2]) >= tonumber(ARGV[3]) then return 0 end
redis.call('HSET', KEYS[3], 'state', 'queued', 'job', ARGV[2])
redis.call('SADD', KEYS[2], ARGV[1])
redis.call('RPUSH', KEYS[1], ARGV[1])
return 1
"""
_CLAIM = """
local id = redis.call('LPOP', KEYS[1])
if not id then return false end
redis.call('HSET', ARGV[1] .. id, 'state', 'running')
return id
"""
_CANCEL = """
local state = redis.call('HGET', KEYS[3], 'state')
if state == 'queued' then
 redis.call('LREM', KEYS[1], 0, ARGV[1])
 redis.call('SREM', KEYS[2], ARGV[1])
 redis.call('HSET', KEYS[3], 'state', 'cancelled')
 redis.call('HDEL', KEYS[3], 'job')
 redis.call('EXPIRE', KEYS[3], ARGV[2])
elseif state == 'running' then
 redis.call('HSET', KEYS[3], 'cancel', '1')
end
return state
"""
_FINISH = """
local state = ARGV[2]
if redis.call('HGET', KEYS[2], 'cancel') == '1' then state = 'cancelled' end
local status = ARGV[5]
local result = ARGV[3]
if state == 'cancelled' then status = '499'; result = 'null' end
redis.call('HSET', KEYS[2], 'state', state, 'result', result, 'error', ARGV[4], 'status', status)
redis.call('HDEL', KEYS[2], 'job')
redis.call('SREM', KEYS[1], ARGV[1])
redis.call('EXPIRE', KEYS[2], ARGV[6])
return state
"""


class InferenceQueue:
    def __init__(self, execute, *, url, lock_path, prefix='kadan:inference:', limit=32,
                 retention=3600, max_payload=16 * 1024 * 1024):
        if limit < 1 or retention < 1 or max_payload < 1:
            raise ValueError('Queue bounds must be positive')
        self.execute = execute
        self.redis = Redis.from_url(url, decode_responses=True, socket_timeout=2, socket_connect_timeout=2)
        self.prefix, self.limit, self.retention, self.max_payload = prefix, limit, retention, max_payload
        self.lock_path = Path(lock_path)
        self._lock_file = None
        self._consumer = None
        self._active = None
        self._active_id = None
        self._wake = asyncio.Event()
        self._ready = False
        self.error = 'Inference queue has not started.'

    def key(self, suffix):
        return self.prefix + suffix

    def _lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open('a+')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle.seek(0)
            identity = handle.read().strip()
            if not identity:
                identity = uuid.uuid4().hex
                handle.write(identity)
                handle.flush()
                os.fsync(handle.fileno())
            self._lock_file = handle
            return identity
        except BaseException:
            handle.close()
            raise

    async def start(self):
        if self._consumer is not None:
            raise RuntimeError('Inference consumer already started')
        try:
            identity = self._lock()
            await self.redis.set(self.key('volume'), identity, nx=True)
            if await self.redis.get(self.key('volume')) != identity:
                raise RuntimeFailure('Redis inference namespace belongs to a different model volume.')
            # The old process has relinquished its kernel lock. Never replay an
            # uncertain inference; callers must deliberately submit a new job.
            for job_id in await self.redis.smembers(self.key('unfinished')):
                await self._finish(job_id, 'failed', error='API restarted before inference completed.', status=503)
            await self.redis.delete(self.key('pending'))
            self._ready, self.error = True, None
            self._consumer = asyncio.create_task(self._consume(), name='kadan-inference-consumer')
        except (OSError, RedisError, RuntimeFailure) as exc:
            self.error = f'Inference queue unavailable: {exc}'
            self._unlock()
            raise RuntimeFailure(self.error) from exc

    def _unlock(self):
        if self._lock_file is not None:
            self._lock_file.close()
            self._lock_file = None

    async def submit(self, feature, operation, payload, model=None):
        if not self._ready:
            raise RuntimeFailure(self.error or 'Inference queue is stopping.')
        job = Job(id=uuid.uuid4().hex, feature=feature, operation=operation, model=model, payload=payload)
        encoded = job.model_dump_json()
        if len(encoded.encode()) > self.max_payload:
            raise RuntimeFailure('Inference request exceeds the queue payload limit.', 413)
        try:
            accepted = await self.redis.eval(_ENQUEUE, 3, self.key('pending'), self.key('unfinished'),
                self.key('job:' + job.id), job.id, encoded, self.limit)
        except (RedisError, asyncio.CancelledError) as exc:
            # Enqueue may have committed even if its response was lost. Cancel
            # that stable ID before returning an error/disconnect to the caller.
            cleanup = asyncio.create_task(self.redis.eval(_CANCEL, 3, self.key('pending'),
                self.key('unfinished'), self.key('job:' + job.id), job.id, self.retention))
            with suppress(RedisError, asyncio.CancelledError):
                await finish_cleanup(cleanup)
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise RuntimeFailure('Redis inference queue is unavailable.') from exc
        if not accepted:
            raise RuntimeFailure('Inference queue is full. Retry after a job completes.', 429)
        self._wake.set()
        return job.id

    async def get(self, job_id):
        if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
            raise KeyError(job_id)
        try:
            fields = ('state', 'cancel', 'result', 'error', 'status')
            values = await self.redis.hmget(self.key('job:' + job_id), fields)
            record = {key: value for key, value in zip(fields, values) if value is not None}
        except RedisError as exc:
            raise RuntimeFailure('Redis inference queue is unavailable.') from exc
        if not record:
            raise KeyError(job_id)
        return record

    async def cancel(self, job_id):
        try:
            await self.get(job_id)
            await self.redis.eval(_CANCEL, 3, self.key('pending'), self.key('unfinished'),
                self.key('job:' + job_id), job_id, self.retention)
        except (RedisError, RuntimeFailure) as exc:
            # The local consumer must still stop if cancellation cannot reach Redis.
            if self._active_id == job_id and self._active is not None:
                self._active.cancel()
            raise RuntimeFailure('Redis inference cancellation is unavailable.') from exc
        self._wake.set()

    async def wait(self, job_id):
        try:
            while True:
                record = await self.get(job_id)
                state = record['state']
                if state == 'succeeded':
                    return json.loads(record['result'])
                if state in ('failed', 'cancelled'):
                    raise RuntimeFailure(record.get('error') or 'Inference cancelled.',
                                         int(record.get('status') or (499 if state == 'cancelled' else 503)))
                if not self._ready:
                    raise RuntimeFailure(self.error or 'Inference queue stopped.')
                await asyncio.sleep(.05)
        except (asyncio.CancelledError, RuntimeFailure):
            # Do not return ownership while a native thread is still running.
            with suppress(RuntimeFailure, KeyError):
                await self.cancel(job_id)
            if self._active_id == job_id and self._active is not None:
                with suppress(asyncio.CancelledError, Exception):
                    await finish_cleanup(self._active)
            raise

    async def _finish(self, job_id, state, result=None, error='', status=503):
        encoded = json.dumps(result, allow_nan=False)
        if len(encoded.encode()) > self.max_payload:
            raise ValueError('Inference result exceeds the queue payload limit')
        await self.redis.eval(_FINISH, 2, self.key('unfinished'), self.key('job:' + job_id),
            job_id, state, encoded, error[:2000], status, self.retention)

    async def _run(self, job_id):
        record = await self.get(job_id)
        if record.get('cancel') == '1':
            await self._finish(job_id, 'cancelled', status=499)
            return
        encoded = await self.redis.hget(self.key('job:' + job_id), 'job')
        job = Job.model_validate_json(encoded)
        self._active_id = job_id
        self._active = asyncio.create_task(self.execute(job))
        try:
            while not self._active.done():
                if not self._ready or (await self.get(job_id)).get('cancel') == '1':
                    self._active.cancel()
                    break
                await asyncio.sleep(.05)
            try:
                result = await self._active
            except asyncio.CancelledError:
                await self._finish(job_id, 'cancelled', status=499)
            except Exception as exc:
                log.warning('Inference %s failed: %s', job_id, exc)
                await self._finish(job_id, 'failed', error=str(exc), status=getattr(exc, 'status_code', 502))
            else:
                try:
                    await self._finish(job_id, 'succeeded', result=result)
                except (TypeError, ValueError):
                    await self._finish(job_id, 'failed', error='Inference returned an invalid or oversized JSON result.', status=502)
        finally:
            if not self._active.done():
                self._active.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await finish_cleanup(self._active)
            self._active = self._active_id = None

    async def _consume(self):
        try:
            while self._ready:
                job_id = await self.redis.eval(_CLAIM, 1, self.key('pending'), self.key('job:'))
                if job_id:
                    await self._run(job_id)
                else:
                    self._wake.clear()
                    with suppress(TimeoutError):
                        await asyncio.wait_for(self._wake.wait(), .1)
        except Exception as exc:
            self._ready = False
            self.error = f'Inference consumer stopped: {exc}'
            log.exception(self.error)
            # Keep the kernel lock until close() even when Redis is down.

    async def close(self):
        if self._lock_file is None:
            await self.redis.aclose()
            return
        self._ready = False
        self._wake.set()
        try:
            if self._consumer is not None:
                await finish_cleanup(self._consumer)
        finally:
            self._consumer = None
            try:
                for job_id in await self.redis.smembers(self.key('unfinished')):
                    await self._finish(job_id, 'failed', error='API shut down before inference completed.')
                await self.redis.delete(self.key('pending'))
            except RedisError:
                pass  # The next lock owner will mark these interrupted jobs failed.
            finally:
                try:
                    await self.redis.aclose()
                finally:
                    self._unlock()
