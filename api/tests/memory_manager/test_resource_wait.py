"""Real Redis FIFO waits on resource ownership without replaying published output."""
import asyncio
import os
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from api.inference.image.rank_session import RankBudget, RankSession
from api.inference.resources import ResourceManager, ResourcePending, ResourceRecoveryRequired
from api.tests.inference.image.test_rank_session import FakeRanks
from api.memory_manager.queue import InferenceQueue
from api.services.runtime import RuntimeFailure


@unittest.skipUnless(os.getenv('KADAN_TEST_REDIS_URL'), 'Dedicated Redis required')
class ResourceWaitTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.resources = ResourceManager(100, {0: 100})
        self.events = []
        self.attempts = 0
        self.prefix = 'kadan:test:' + uuid4().hex + ':'
        async def execute(job):
            self.attempts += 1
            self.resources.offload_workload_devices('llm')
            handle = self.resources.reserve(job.id, 'video', host_bytes=job.payload.get('bytes', 50))
            self.events.append(job.feature)
            handle.release()
            return job.feature
        self.queue = InferenceQueue(execute, url=os.environ['KADAN_TEST_REDIS_URL'],
            lock_path=Path(self.temp.name) / 'consumer.lock', prefix=self.prefix,
            resources=lambda: self.resources)
        await self.queue.start()

    async def asyncTearDown(self):
        await self.queue.close()
        redis = self.queue.redis
        keys = [key async for key in redis.scan_iter(match=self.prefix + '*')]
        if keys:
            await redis.delete(*keys)
        await redis.aclose()
        self.assertFalse(self.resources._listeners)
        self.temp.cleanup()

    async def state(self, job, expected):
        async def wait():
            while (await self.queue.get(job))['state'] != expected:
                await asyncio.sleep(.01)
        await asyncio.wait_for(wait(), 3)

    async def test_active_startup_load_waits_then_parks_before_fifo_modalities(self):
        parked = []
        held = self.resources.reserve('loading-text', 'llm', device_bytes={0: 90}, evict=lambda: parked.append('text'))
        lease = held.lease()
        lease.__enter__()
        first = await self.queue.submit('stt', 'generate', {})
        await self.state(first, 'waiting_for_resources')
        second = await self.queue.submit('image', 'generate', {})
        third = await self.queue.submit('decisions', 'generate', {})
        await asyncio.sleep(.15)
        self.assertEqual(self.attempts, 1)
        self.assertEqual(self.events, [])
        self.assertEqual(self.resources.snapshot()['reservations']['loading-text']['active_leases'], 1)
        lease.__exit__(None, None, None)
        self.assertEqual(await asyncio.wait_for(self.queue.wait(first), 1), 'stt')
        await self.queue.wait(second)
        await self.queue.wait(third)
        self.assertEqual(self.events, ['stt', 'image', 'decisions'])
        self.assertEqual(parked, ['text'])
        self.assertFalse(self.resources.snapshot()['reservations'])

    async def test_capacity_wait_release_and_cancel_preserve_head_order(self):
        held = self.resources.reserve('external', 'llm', host_bytes=80)
        first = await self.queue.submit('video', 'generate', {})
        await self.state(first, 'waiting_for_resources')
        second = await self.queue.submit('stt', 'generate', {})
        await self.queue.cancel(first)
        await self.state(first, 'cancelled')
        await self.state(second, 'waiting_for_resources')
        self.assertEqual(self.resources.snapshot()['reservations']['external']['host_bytes'], 80)
        held.release()
        self.assertEqual(await asyncio.wait_for(self.queue.wait(second), 1), 'stt')
        self.assertEqual(self.events, ['stt'])

    async def test_impossible_single_reservation_is_not_an_infinite_retry(self):
        first = await self.queue.submit('video', 'generate', {'bytes': 101})
        with self.assertRaises(RuntimeFailure):
            await asyncio.wait_for(self.queue.wait(first), 1)
        self.assertEqual((await self.queue.get(first))['state'], 'failed')
        self.assertEqual(self.attempts, 1)
        second = await self.queue.submit('stt', 'generate', {})
        self.assertEqual(await self.queue.wait(second), 'stt')

    async def test_rank_cleanup_uncertainty_fails_without_blocking_fifo(self):
        self.resources = ResourceManager(1000, {0: 100, 1: 100})
        transport = FakeRanks()
        session = RankSession(self.resources, transport, RankBudget(200, (0, 1), 10, 70), enabled=True)
        session.execute('a' * 32)
        before = self.resources.snapshot()['reservations']
        transport.confirmed = False
        transport.mutate = lambda command, replies: [replies[0], dict(replies[1], resident_bytes=1)]

        async def execute(job):
            self.attempts += 1
            if job.feature == 'video':
                # Real handoff invokes RankSession.park -> failed acknowledgement
                # -> unconfirmed transport stop, retaining all three reservations.
                self.resources.offload_workload_devices('image')
            elif job.feature == 'image':
                session.execute(job.id)
            return job.feature
        self.queue.execute = execute
        try:
            for feature in ('video', 'image'):
                job = await self.queue.submit(feature, 'generate', {})
                with self.assertRaisesRegex(RuntimeFailure, 'cleanup.*unconfirmed'):
                    await asyncio.wait_for(self.queue.wait(job), 1)
                self.assertEqual((await self.queue.get(job))['state'], 'failed')
            self.assertEqual(self.attempts, 2)
            self.assertEqual(session.state, 'quarantined')
            self.assertEqual(self.resources.snapshot()['reservations'], before)
            self.assertFalse(self.resources._listeners)
            # Capacity eviction must propagate quarantine instead of swallowing
            # it as a busy candidate and converting it into ResourcePending.
            with self.assertRaises(ResourceRecoveryRequired):
                self.resources.reserve('next', 'video', device_bytes={0: 90, 1: 90})
            self.assertEqual(self.resources.snapshot()['reservations'], before)
            following = await self.queue.submit('decisions', 'generate', {})
            self.assertEqual(await asyncio.wait_for(self.queue.wait(following), 1), 'decisions')
        finally:
            transport.confirmed = True
            session.close(recover=True)
        self.assertFalse(self.resources.snapshot()['reservations'])

    async def test_recovery_failure_does_not_reuse_transient_cause(self):
        async def blocked(job):
            self.attempts += 1
            try:
                raise ResourcePending('initial admission')
            except ResourcePending as error:
                raise ResourceRecoveryRequired('cleanup unconfirmed') from error
        self.queue.execute = blocked
        first = await self.queue.submit('video', 'generate', {})
        with self.assertRaisesRegex(RuntimeFailure, 'cleanup unconfirmed'):
            await asyncio.wait_for(self.queue.wait(first), 1)
        self.assertEqual(self.attempts, 1)
        self.assertFalse(self.resources._listeners)

    async def test_wrapped_transient_failure_waits_and_close_cancels_waiter(self):
        async def blocked(job):
            try:
                raise ResourcePending('temporarily busy')
            except ResourcePending as error:
                raise RuntimeFailure('wrapped', 503) from error
        self.queue.execute = blocked
        first = await self.queue.submit('video', 'generate', {})
        await self.state(first, 'waiting_for_resources')
        await asyncio.wait_for(self.queue.close(), 1)
        self.assertFalse(self.resources._listeners)

    async def test_change_between_observation_and_wait_is_not_lost(self):
        revision = self.resources.revision
        held = self.resources.reserve('test', 'llm', host_bytes=1)
        held.release()
        await asyncio.wait_for(self.resources.wait_for_change(revision), .1)
