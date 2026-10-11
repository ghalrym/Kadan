import asyncio
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

from api.pydantic_models.inference import EmptyPayload
from api.services.native_worker import NativeJob, NativeRequest, NativeWorker


class NativeTransportTests(unittest.IsolatedAsyncioTestCase):
    def job(self):
        return NativeJob(NativeRequest(feature='image', operation='generate', model='image'), False)

    async def test_prepared_send_finishes_before_repeated_cancellation_escapes(self):
        worker = NativeWorker(Path('/unused'))
        entered, release = asyncio.Event(), asyncio.Event()
        async def send(message):
            entered.set()
            await release.wait()
        job = self.job()
        with patch.object(worker, '_send', side_effect=send):
            task = asyncio.create_task(worker.prepared(job, EmptyPayload()))
            await entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertFalse(job.prepared_sent)
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(job.prepared_sent)

    async def test_cancel_preparation_acknowledges_empty_payload_once(self):
        worker, job = NativeWorker(Path('/unused')), self.job()
        async def prepared(*args):
            job.result.set_result(None)
        with patch.object(worker, 'cancel', new=AsyncMock()) as cancel, \
             patch.object(worker, 'prepared', side_effect=prepared) as prepare:
            await worker.cancel_and_join(job)
        cancel.assert_awaited_once_with(job.request.id)
        self.assertEqual(prepare.await_count, 1)

    async def test_cancel_running_job_never_sends_second_preparation(self):
        worker, job = NativeWorker(Path('/unused')), self.job()
        job.prepared_sent = True
        async def cancel(_):
            job.result.set_result(None)
        with patch.object(worker, 'cancel', side_effect=cancel), \
             patch.object(worker, 'prepared', new=AsyncMock()) as prepare:
            await worker.cancel_and_join(job)
        prepare.assert_not_awaited()
