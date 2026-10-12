import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from api.services.inference_worker import InferenceJob, InferenceTask, InferenceWorker


class InferenceTransportTests(unittest.IsolatedAsyncioTestCase):
    def job(self):
        return InferenceJob(
            InferenceTask(input={"feature": "image", "prompt": "test"}), False
        )

    async def test_cancellation_only_sends_cancel_and_waits_for_worker(self):
        worker, job = InferenceWorker(Path("/unused")), self.job()
        entered = asyncio.Event()

        async def cancel(_):
            entered.set()

        with patch.object(worker, "cancel", side_effect=cancel) as send:
            task = asyncio.create_task(worker.cancel_and_join(job))
            await entered.wait()
            self.assertFalse(task.done())
            job.result.set_result(None)
            await task
        send.assert_awaited_once_with(job.request.id)

    async def test_oversized_request_is_rejected_without_registering_or_cancelling(
        self,
    ):
        worker = InferenceWorker(Path("/unused"))
        request = InferenceTask(
            input={"feature": "image", "prompt": "x" * (3 * 1024 * 1024)}
        )
        with patch.object(worker, "_send", new=AsyncMock()) as send:
            with self.assertRaisesRegex(Exception, "transport limit"):
                await worker.submit(request)
        send.assert_not_awaited()
        self.assertFalse(worker.jobs)

    async def test_submission_serializes_complete_task(self):
        worker = InferenceWorker(Path("/unused"))
        request = InferenceTask(input={"feature": "image", "prompt": "mountain"})

        async def send(value):
            self.assertEqual(InferenceTask.model_validate(value), request)
            self.assertEqual(value["command"], "submit")
            worker.jobs[request.id].accepted.set_result(None)

        with patch.object(worker, "_send", side_effect=send):
            job = await worker.submit(request)
        self.assertEqual(job.request, request)

    async def test_duplicate_id_does_not_send_cancel_or_replace_original(self):
        worker, job = InferenceWorker(Path("/unused")), self.job()
        worker.jobs[job.request.id] = job
        with (
            patch.object(worker, "_send", new=AsyncMock()) as send,
            self.assertRaisesRegex(Exception, "already active"),
        ):
            await worker.submit(job.request)
        send.assert_not_awaited()
        self.assertIs(worker.jobs[job.request.id], job)

    async def test_image_progress_is_initialized_before_acceptance_events(self):
        worker = InferenceWorker(Path("/unused"))
        request = InferenceTask(
            input={"feature": "image", "prompt": "test", "count": 2, "steps": 20}
        )

        async def send(_):
            job = worker.jobs[request.id]
            self.assertEqual(job.image_progress.image_count, 2)
            self.assertEqual(job.image_progress.steps_total, 20)
            job.image_progress.phase = "denoise"
            job.image_progress.steps_completed = 1
            job.accepted.set_result(None)

        with patch.object(worker, "_send", side_effect=send):
            job = await worker.submit(request)
        self.assertEqual(job.image_progress.phase, "denoise")
        self.assertEqual(job.image_progress.steps_completed, 1)


class WorkerExitTests(unittest.IsolatedAsyncioTestCase):
    async def exited_worker(self, returncode, *, ready=True, protocol=b""):
        worker = InferenceWorker(Path("/unused"))
        worker.ready = asyncio.get_running_loop().create_future()
        if ready:
            worker.ready.set_result(123)
        job = InferenceJob(
            InferenceTask(input={"feature": "image", "prompt": "test"}), False
        )
        worker.jobs[job.request.id] = job
        stdout, stderr = asyncio.StreamReader(), asyncio.StreamReader()
        stdout.feed_data(protocol)
        stdout.feed_eof()
        worker.process = SimpleNamespace(
            stdout=stdout,
            stderr=stderr,
            pid=123,
            returncode=returncode,
            wait=AsyncMock(return_value=returncode),
        )
        worker.stderr_reader = asyncio.create_task(worker._stderr())
        reader = asyncio.create_task(worker._read())
        await asyncio.sleep(0)
        self.assertFalse(reader.done())
        stderr.feed_data(b"x" * 10000 + b"last native diagnostic")
        stderr.feed_eof()
        with (
            self.assertLogs("api.services.inference_worker", level="ERROR") as logs,
            patch("api.services.inference_worker.os.killpg"),
        ):
            await reader
        for future in (job.accepted, job.result):
            with self.assertRaisesRegex(Exception, "last native diagnostic"):
                await future
        if not ready:
            with self.assertRaisesRegex(Exception, "last native diagnostic"):
                await worker.ready
        self.assertIn("last native diagnostic", logs.output[0])
        self.assertEqual(len(worker.last_exit.stderr), 8192)
        return worker

    async def test_stdout_eof_drains_late_stderr_and_retains_exit_code(self):
        worker = await self.exited_worker(7)
        self.assertEqual(worker.last_exit.returncode, 7)
        self.assertIsNone(worker.last_exit.signal)
        self.assertIn("code 7", worker.last_exit.describe())

    async def test_startup_signal_exit_is_retained_with_stderr(self):
        worker = await self.exited_worker(-11, ready=False)
        self.assertEqual(worker.last_exit.signal, 11)
        self.assertIn("signal 11", worker.last_exit.describe())

    async def test_model_snapshot_does_not_require_an_outstanding_job(self):
        worker = await self.exited_worker(
            0,
            protocol=(
                b'{"state":"models","models":{"models":[],"selected_model_id":"small","selected_whisper":"tiny.en"}}\n'
            ),
        )
        self.assertIsNone(worker.last_exit.protocol_error)
        self.assertEqual(worker.models.selected_model_id, "small")
        self.assertEqual(worker.models.selected_whisper, "tiny.en")

    async def test_protocol_error_does_not_replace_native_stderr(self):
        worker = await self.exited_worker(1, protocol=b"not json\n")
        self.assertTrue(worker.last_exit.protocol_error)
        self.assertTrue(worker.last_exit.stderr.endswith("last native diagnostic"))
