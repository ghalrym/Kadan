import asyncio
import os
from pathlib import Path
import tempfile
import threading
import unittest
import uuid

from api.inference.feature import native_call
from api.memory_manager.queue import InferenceQueue
from api.memory_manager.streaming import QueuedStream


@unittest.skipUnless(os.getenv('KADAN_TEST_REDIS_URL'), 'Real Redis required')
class StreamQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_queue_stream_and_cancel_retains_cleanup_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            cleaned = threading.Event()
            def work(cancel):
                channel.emit({'content': 'one '})
                cancel.wait(5)
                cleaned.set()
            async def execute(job):
                await native_call(work)
                return 'done'
            queue = InferenceQueue(execute, url=os.environ['KADAN_TEST_REDIS_URL'],
                prefix='kadan:test:'+uuid.uuid4().hex+':', lock_path=Path(directory)/'lock')
            try:
                await queue.start()
                channel = QueuedStream(queue)
                channel.job_id = await queue.submit('llm','completion',{},stream=channel)
                channel.start()
                iterator = channel.__aiter__()
                self.assertEqual(await asyncio.wait_for(anext(iterator), 2), {'content':'one '})
                self.assertFalse(channel.waiter.done())
                await channel.aclose()
                self.assertTrue(cleaned.is_set())
                self.assertNotIn(channel.job_id, queue.streams)
                await iterator.aclose()
            finally:
                await queue.close()
                keys = [key async for key in queue.redis.scan_iter(match=queue.prefix+'*')]
                if keys: await queue.redis.delete(*keys)
                await queue.redis.aclose()

    async def test_disconnect_releases_backpressured_producer(self):
        channel = QueuedStream(None)
        for i in range(16):
            channel.events.put_nowait({'content':'x'})
        worker = asyncio.create_task(asyncio.to_thread(channel.emit, {'content':'blocked'}))
        await asyncio.sleep(.03)
        self.assertFalse(worker.done())
        channel.closed.set()
        with self.assertRaises(InterruptedError):
            await asyncio.wait_for(worker, 1)
