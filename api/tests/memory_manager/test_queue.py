"""Real Redis lifecycle tests. CI provisions Redis; no fake broker semantics."""
import asyncio
import os
import subprocess
import sys
import threading
from pathlib import Path
import tempfile
import unittest
import uuid

from redis.asyncio import Redis

from api.inference.cancellation import run_cancellable_thread
from api.memory_manager.queue import InferenceQueue
from api.inference.errors import InferenceFailure


@unittest.skipUnless(os.getenv('KADAN_TEST_REDIS_URL'), 'Set KADAN_TEST_REDIS_URL for real Redis integration')
class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.prefix = f'kadan:test:{uuid.uuid4().hex}:'
        self.order = []
        self.started, self.release = asyncio.Event(), asyncio.Event()

        async def execute(job):
            self.order.append(job.payload['number'])
            if job.payload.get('wait'):
                self.started.set()
                await self.release.wait()
            if job.payload.get('fail'):
                raise InferenceFailure('native failure', 422)
            return {'number': job.payload['number']}

        self.execute = execute
        self.queue = self.make_queue()
        await self.queue.start()

    def make_queue(self, **kwargs):
        return InferenceQueue(self.execute, url=os.environ['KADAN_TEST_REDIS_URL'],
            lock_path=Path(self.directory.name) / 'consumer.lock', prefix=self.prefix, limit=3, **kwargs)

    async def asyncTearDown(self):
        self.release.set()
        await self.queue.close()
        # Delete only this test's isolated namespace, never a whole Redis DB.
        redis = self.make_queue().redis
        keys = [key async for key in redis.scan_iter(match=self.prefix + '*')]
        if keys:
            await redis.delete(*keys)
        await redis.aclose()
        self.directory.cleanup()

    async def submit(self, number, **kwargs):
        return await self.queue.submit('llm', 'generate', {'number': number, **kwargs}, 'fixture')

    async def test_fifo_capacity_and_failure_recovery(self):
        first = await self.submit(1, wait=True)
        await asyncio.wait_for(self.started.wait(), 2)
        second = await self.submit(2, fail=True)
        third = await self.submit(3)
        with self.assertRaises(InferenceFailure) as caught:
            await self.submit(4)
        self.assertEqual(caught.exception.status_code, 429)
        self.release.set()
        self.assertEqual(await self.queue.wait(first), {'number': 1})
        with self.assertRaisesRegex(InferenceFailure, 'native failure'):
            await self.queue.wait(second)
        self.assertEqual(await self.queue.wait(third), {'number': 3})
        self.assertEqual(self.order, [1, 2, 3])
        self.assertEqual(await self.queue.redis.scard(self.queue.key('unfinished')), 0)
        self.assertNotIn('job', await self.queue.get(first))
        self.assertGreater(await self.queue.redis.ttl(self.queue.key('job:' + first)), 0)

    async def test_queued_cancel_never_executes_and_frees_capacity(self):
        first = await self.submit(1, wait=True)
        await self.started.wait()
        second = await self.submit(2)
        await self.queue.cancel(second)
        self.release.set()
        await self.queue.wait(first)
        with self.assertRaises(InferenceFailure):
            await self.queue.wait(second)
        self.assertEqual(self.order, [1])
        self.assertEqual((await self.queue.get(second))['state'], 'cancelled')

    async def test_inference_cancel_waits_for_thread_before_next_job(self):
        started, cleaned = threading.Event(), threading.Event()

        def native(cancel):
            started.set()
            cancel.wait(3)
            cleaned.set()

        async def execute(job):
            if job.payload['number'] == 1:
                await run_cancellable_thread(native)
            else:
                self.assertTrue(cleaned.is_set())
            return job.payload

        self.queue.execute = execute
        first = await self.submit(1)
        await asyncio.wait_for(asyncio.to_thread(started.wait), 2)
        second = await self.submit(2)
        waiting = asyncio.create_task(self.queue.wait(first))
        waiting.cancel()
        # Explicit DELETE has the same cooperative cancellation semantics.
        await self.queue.cancel(first)
        self.assertEqual(await self.queue.wait(second), {'number': 2})
        self.assertTrue(cleaned.is_set())
        self.assertEqual((await self.queue.get(first))['state'], 'cancelled')

    async def test_disconnect_waiter_retains_cleanup_ownership(self):
        cleaned = asyncio.Event()

        async def execute(job):
            self.started.set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(.05)
                cleaned.set()

        self.queue.execute = execute
        job_id = await self.submit(1)
        await self.started.wait()
        waiter = asyncio.create_task(self.queue.wait(job_id))
        await asyncio.sleep(0)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertTrue(cleaned.is_set())

    async def test_second_consumer_cannot_take_lock_or_clean_live_jobs(self):
        first = await self.submit(1, wait=True)
        await self.started.wait()
        other = self.make_queue()
        with self.assertRaises(InferenceFailure):
            await other.start()
        await other.close()
        self.assertEqual((await self.queue.get(first))['state'], 'running')
        self.release.set()
        await self.queue.wait(first)

    async def test_different_volume_cannot_consume_same_namespace(self):
        other = self.make_queue()
        other.lock_path = Path(self.directory.name) / 'other.lock'
        with self.assertRaisesRegex(InferenceFailure, 'different model volume'):
            await other.start()
        await other.close()

    async def test_restart_marks_uncertain_jobs_failed_without_replay(self):
        await self.queue.close()
        self.queue = self.make_queue()
        stale_ids = [uuid.uuid4().hex, uuid.uuid4().hex]
        for job_id, state in zip(stale_ids, ('queued', 'running')):
            await self.queue.redis.hset(self.queue.key('job:' + job_id), mapping={'state': state})
            await self.queue.redis.sadd(self.queue.key('unfinished'), job_id)
        await self.queue.redis.rpush(self.queue.key('pending'), stale_ids[0])
        await self.queue.start()
        for job_id in stale_ids:
            with self.assertRaisesRegex(InferenceFailure, 'restarted'):
                await self.queue.wait(job_id)
        self.assertEqual(self.order, [])
        self.assertEqual(await self.queue.wait(await self.submit(3)), {'number': 3})

    async def test_json_only_and_payload_bound(self):
        with self.assertRaises(ValueError):
            await self.queue.submit('llm', 'generate', {'tensor': object()})
        self.queue.max_payload = 100
        with self.assertRaises(InferenceFailure) as caught:
            await self.submit(1, text='x' * 200)
        self.assertEqual(caught.exception.status_code, 413)

    async def test_shutdown_cleans_active_and_fails_pending(self):
        first = await self.submit(1, wait=True)
        await self.started.wait()
        second = await self.submit(2)
        await self.queue.close()
        self.queue = self.make_queue()
        await self.queue.start()
        self.assertEqual((await self.queue.get(first))['state'], 'cancelled')
        self.assertEqual((await self.queue.get(second))['state'], 'failed')
        self.assertEqual(self.order, [1])

    async def test_redis_connection_failure_stops_inference_and_retains_lock(self):
        cleaned = asyncio.Event()

        async def execute(job):
            self.started.set()
            try:
                await asyncio.Future()
            finally:
                cleaned.set()

        self.queue.execute = execute
        first = await self.submit(1)
        await self.started.wait()
        original = self.queue.redis
        self.queue.redis = Redis.from_url('redis://127.0.0.1:1', socket_connect_timeout=.1)
        await asyncio.wait_for(cleaned.wait(), 2)
        await asyncio.wait_for(self.queue._consumer, 2)
        self.assertFalse(self.queue._ready)
        other = self.make_queue()
        with self.assertRaises(InferenceFailure):
            await other.start()
        await other.close()
        await self.queue.redis.aclose()
        self.queue.redis = original
        await self.queue.close()
        self.queue = self.make_queue()
        await self.queue.start()
        self.assertEqual((await self.queue.get(first))['state'], 'failed')

    async def test_actual_process_exit_releases_lock_and_fails_stale_jobs(self):
        await self.queue.close()
        script = '''
import asyncio, os
from api.memory_manager.queue import InferenceQueue
async def main():
    started = asyncio.Event()
    async def execute(job):
        started.set()
        await asyncio.Future()
    queue = InferenceQueue(execute, url=os.environ['KADAN_TEST_REDIS_URL'],
        lock_path=os.environ['TEST_LOCK'], prefix=os.environ['TEST_PREFIX'])
    await queue.start()
    first = await queue.submit('llm', 'generate', {'number': 1})
    await started.wait()
    second = await queue.submit('llm', 'generate', {'number': 2})
    print(first, second, flush=True)
    os._exit(23)
asyncio.run(main())
'''
        env = dict(os.environ, TEST_LOCK=str(self.queue.lock_path), TEST_PREFIX=self.prefix)
        process = await asyncio.to_thread(subprocess.run, [sys.executable, '-c', script],
            env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(process.returncode, 23, process.stderr)
        self.queue = self.make_queue()
        await self.queue.start()
        for job_id in process.stdout.strip().split():
            self.assertEqual((await self.queue.get(job_id))['state'], 'failed')
        self.assertEqual(self.order, [])

    async def test_invalid_inference_result_fails_one_job_without_stopping_consumer(self):
        async def execute(job):
            return float('nan') if job.payload['number'] == 1 else {'number': 2}
        self.queue.execute = execute
        first, second = await self.submit(1), await self.submit(2)
        with self.assertRaisesRegex(InferenceFailure, 'invalid or oversized'):
            await self.queue.wait(first)
        self.assertEqual(await self.queue.wait(second), {'number': 2})
        self.assertTrue(self.queue._ready)
