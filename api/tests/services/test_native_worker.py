import asyncio
from pathlib import Path
from types import SimpleNamespace
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


class WorkerExitTests(unittest.IsolatedAsyncioTestCase):
    async def exited_worker(self, returncode, *, ready=True, protocol=b''):
        worker = NativeWorker(Path('/unused'))
        worker.ready = asyncio.get_running_loop().create_future()
        if ready:
            worker.ready.set_result(123)
        job = NativeJob(NativeRequest(feature='image', operation='generate', model='image'), False)
        worker.jobs[job.request.id] = job
        stdout, stderr = asyncio.StreamReader(), asyncio.StreamReader()
        stdout.feed_data(protocol)
        stdout.feed_eof()
        worker.process = SimpleNamespace(stdout=stdout, stderr=stderr, pid=123,
            returncode=returncode, wait=AsyncMock(return_value=returncode))
        worker.stderr_reader = asyncio.create_task(worker._stderr())
        reader = asyncio.create_task(worker._read())
        await asyncio.sleep(0)
        self.assertFalse(reader.done())
        stderr.feed_data(b'x' * 10000 + b'last native diagnostic')
        stderr.feed_eof()
        with self.assertLogs('api.services.native_worker', level='ERROR') as logs:
            await reader
        for future in (job.accepted, job.result):
            with self.assertRaisesRegex(Exception, 'last native diagnostic'):
                await future
        if not ready:
            with self.assertRaisesRegex(Exception, 'last native diagnostic'):
                await worker.ready
        self.assertIn('last native diagnostic', logs.output[0])
        self.assertEqual(len(worker.last_exit.stderr), 8192)
        return worker

    async def test_stdout_eof_drains_late_stderr_and_retains_exit_code(self):
        worker = await self.exited_worker(7)
        self.assertEqual(worker.last_exit.returncode, 7)
        self.assertIsNone(worker.last_exit.signal)
        self.assertIn('code 7', worker.last_exit.describe())

    async def test_startup_signal_exit_is_retained_with_stderr(self):
        worker = await self.exited_worker(-11, ready=False)
        self.assertEqual(worker.last_exit.signal, 11)
        self.assertIn('signal 11', worker.last_exit.describe())

    async def test_protocol_error_does_not_replace_native_stderr(self):
        worker = await self.exited_worker(1, protocol=b'not json\n')
        self.assertTrue(worker.last_exit.protocol_error)
        self.assertTrue(worker.last_exit.stderr.endswith('last native diagnostic'))
