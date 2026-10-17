import asyncio
import io
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from api.services.inference_worker import (
    InferenceJob,
    InferenceTask,
    InferenceWorker,
    encode_request,
)


class InferenceWorkerConfigurationTests(unittest.IsolatedAsyncioTestCase):
    def test_model_directory_uses_hikari_then_legacy_then_persistent_default(self):
        cases = (
            ({}, "/tmp/hikari-test-home/.local/share/kadan/models/inference"),
            (
                {"KADAN_MODEL_DIR": "~/legacy-models"},
                "/tmp/hikari-test-home/legacy-models/inference",
            ),
            (
                {
                    "HIKARI_MODEL_DIR": "~/hikari-models",
                    "KADAN_MODEL_DIR": "~/legacy-models",
                },
                "/tmp/hikari-test-home/hikari-models/inference",
            ),
        )
        for environment, expected in cases:
            with (
                self.subTest(environment=environment),
                patch.dict(
                    os.environ,
                    {"HOME": "/tmp/hikari-test-home", **environment},
                    clear=True,
                ),
            ):
                self.assertEqual(InferenceWorker().root, Path(expected))

    def test_explicit_root_overrides_model_directory_environment(self):
        with patch.dict(
            os.environ,
            {"HIKARI_MODEL_DIR": "/hikari-models", "KADAN_MODEL_DIR": "/legacy-models"},
            clear=True,
        ):
            self.assertEqual(InferenceWorker(Path("/explicit")).root, Path("/explicit"))

    async def test_executable_uses_hikari_then_legacy_then_installed_default(self):
        cases = (
            ({}, "/opt/kadan/bin/kadan-inference-worker"),
            ({"KADAN_INFERENCE_WORKER": "/legacy-worker"}, "/legacy-worker"),
            (
                {
                    "HIKARI_INFERENCE_WORKER": "/hikari-worker",
                    "KADAN_INFERENCE_WORKER": "/legacy-worker",
                },
                "/hikari-worker",
            ),
        )
        for environment, expected in cases:
            with (
                self.subTest(environment=environment),
                patch.dict(os.environ, environment, clear=True),
                patch("pathlib.Path.mkdir"),
                patch("pathlib.Path.is_file", return_value=True),
                patch(
                    "asyncio.create_subprocess_exec",
                    side_effect=OSError("Launch disabled for configuration test"),
                ) as launch,
            ):
                worker = InferenceWorker(Path("/unused"))
                with self.assertRaisesRegex(OSError, "Launch disabled"):
                    await worker.start()
                launch.assert_awaited_once()
                self.assertEqual(launch.call_args.args[0], expected)

    async def test_video_directory_is_forwarded_without_changing_parent_environment(
        self,
    ):
        cases = (
            ({}, None),
            ({"KADAN_VIDEO_DIR": "/legacy-videos"}, None),
            ({"HIKARI_VIDEO_DIR": "~/hikari-videos"}, "~/hikari-videos"),
            (
                {
                    "HIKARI_VIDEO_DIR": "/hikari-videos",
                    "KADAN_VIDEO_DIR": "/legacy-videos",
                },
                "/hikari-videos",
            ),
            ({"HIKARI_VIDEO_DIR": "", "KADAN_VIDEO_DIR": "/legacy-videos"}, ""),
        )
        for configuration, override in cases:
            environment = {"UNRELATED_SETTING": "retained", **configuration}
            with (
                self.subTest(configuration=configuration),
                patch.dict(os.environ, environment, clear=True),
                patch("pathlib.Path.mkdir"),
                patch("pathlib.Path.is_file", return_value=True),
                patch(
                    "asyncio.create_subprocess_exec",
                    side_effect=OSError("Launch disabled for configuration test"),
                ) as launch,
            ):
                worker = InferenceWorker(Path("/unused"))
                with self.assertRaisesRegex(OSError, "Launch disabled"):
                    await worker.start()
                launch.assert_awaited_once()
                self.assertEqual(
                    launch.call_args.kwargs["env"],
                    {**environment, "KADAN_VIDEO_DIR": override}
                    if override is not None
                    else None,
                )
                self.assertEqual(dict(os.environ), environment)


class InferenceTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_upload_above_old_limit_reaches_worker_input(self):
        worker = InferenceWorker(Path("/unused"))
        request = InferenceTask(
            input={
                "feature": "image",
                "prompt": "sky",
                "image": "x" * (3 * 1024 * 1024),
            }
        )

        def write(payload):
            self.assertGreater(len(payload), 3 * 1024 * 1024)
            worker.jobs[request.id].accepted.set_result(None)

        worker.process = SimpleNamespace(
            returncode=None, stdin=SimpleNamespace(write=write, drain=AsyncMock())
        )
        job = await worker.submit(request)
        self.assertTrue(job.sent)

    def test_image_upload_still_has_a_bounded_transport(self):
        with self.assertRaisesRegex(Exception, "transport limit"):
            encode_request(
                {"input": {"feature": "image", "image": "x" * (64 * 1024 * 1024)}}
            )

    async def test_timing_lines_are_relayed_across_chunks_without_progress(self):
        callback = Mock()
        worker = InferenceWorker(Path("/unused"), callback)
        chunks = [
            b"ordinary diagnostic\nKAD",
            b'AN_IMAGE_TIMING {"step":',
            b'1}\nKADAN_IMAGE_TIMING {"step":2}\n',
            b"KADAN_ATEN_IMAGE_",
            b'TIMING {"phase":"load"}\nKADAN_ATEN_IMAGE_TIMING_EXTRA ignored\n',
            b'KADAN_ATEN_IMAGE_TIMING {"phase":"generation"}\n',
            b'KADAN_ATEN_IMAGE_TIMING {"phase":"unload"}\n',
            b'KADAN_ATEN_IMAGE_BATCH {"images":2,"host_peak_bytes":4096}\n',
        ]
        worker.process = SimpleNamespace(
            stderr=SimpleNamespace(read=AsyncMock(side_effect=[*chunks, b""]))
        )
        output = io.StringIO()
        with patch("sys.stderr", output):
            await worker._stderr()
        self.assertEqual(
            output.getvalue(),
            'KADAN_IMAGE_TIMING {"step":1}\nKADAN_IMAGE_TIMING {"step":2}\n'
            'KADAN_ATEN_IMAGE_TIMING {"phase":"load"}\n'
            'KADAN_ATEN_IMAGE_TIMING {"phase":"generation"}\n'
            'KADAN_ATEN_IMAGE_TIMING {"phase":"unload"}\n'
            'KADAN_ATEN_IMAGE_BATCH {"images":2,"host_peak_bytes":4096}\n',
        )
        self.assertEqual("".join(worker.diagnostics), b"".join(chunks).decode())
        callback.assert_not_called()
        self.assertIsNone(worker.memory)
        self.assertFalse(worker.jobs)

    async def test_oversized_and_incomplete_timing_lines_do_not_relay(self):
        worker = InferenceWorker(Path("/unused"))
        chunks = [
            b"KADAN_ATEN_IMAGE_TIMING " + b"x" * 4100 + b"\n",
            b"KADAN_IMAGE_TIMING " + b"x" * 4000,
            b"x" * 100 + b'\nKADAN_IMAGE_TIMING {"step":1}\nKADAN_ATEN_IMAGE_',
            b"TIMING incomplete",
        ]
        worker.process = SimpleNamespace(
            stderr=SimpleNamespace(read=AsyncMock(side_effect=[*chunks, b""]))
        )
        output = io.StringIO()
        with patch("sys.stderr", output):
            await worker._stderr()
        self.assertEqual(output.getvalue(), 'KADAN_IMAGE_TIMING {"step":1}\n')
        self.assertEqual("".join(worker.diagnostics), b"".join(chunks).decode())

    async def test_timing_log_failure_keeps_native_diagnostics(self):
        worker = InferenceWorker(Path("/unused"))
        line = b'KADAN_IMAGE_TIMING {"step":1}\n'
        worker.process = SimpleNamespace(
            stderr=SimpleNamespace(read=AsyncMock(side_effect=[line, b""]))
        )
        with patch("sys.stderr.write", side_effect=OSError("log unavailable")):
            await worker._stderr()
        self.assertEqual("".join(worker.diagnostics), line.decode())

    def job(self):
        return InferenceJob(
            InferenceTask(input={"feature": "image", "prompt": "test"}), False
        )

    async def test_cancellation_before_write_does_not_wait_for_unknown_native_job(self):
        worker = InferenceWorker(Path("/unused"))
        worker.process = SimpleNamespace(
            returncode=None, stdin=SimpleNamespace(write=Mock(), drain=AsyncMock())
        )
        request = self.job().request
        await worker.write_lock.acquire()
        submission = asyncio.create_task(worker.submit(request))
        await asyncio.sleep(0)
        submission.cancel()
        await asyncio.sleep(0)
        completed = submission.done()
        # Release the old implementation's stuck cleanup without a real worker.
        worker.write_lock.release()
        if not completed:
            worker.jobs[request.id].result.set_result(None)
        with self.assertRaises(asyncio.CancelledError):
            await submission
        self.assertTrue(completed)
        worker.process.stdin.write.assert_not_called()
        self.assertFalse(worker.jobs)

    async def test_cancellation_after_write_waits_for_native_cleanup(self):
        worker = InferenceWorker(Path("/unused"))
        request = self.job().request
        draining, cancelled = asyncio.Event(), asyncio.Event()
        commands = []

        def write(data):
            commands.append(json.loads(data)["command"])
            if commands[-1] == "cancel":
                cancelled.set()

        async def drain():
            if commands == ["submit"]:
                draining.set()
                await asyncio.Future()

        worker.process = SimpleNamespace(
            returncode=None, stdin=SimpleNamespace(write=write, drain=drain)
        )
        submission = asyncio.create_task(worker.submit(request))
        await draining.wait()
        submission.cancel()
        await cancelled.wait()
        job = worker.jobs[request.id]
        submission.cancel()
        await asyncio.sleep(0)
        self.assertFalse(submission.done())
        self.assertIs(worker.jobs[request.id], job)
        job.result.set_result(None)
        with self.assertRaises(asyncio.CancelledError):
            await submission
        self.assertEqual(commands, ["submit", "cancel"])
        self.assertFalse(worker.jobs)

    async def test_failed_write_does_not_cancel_an_unsent_job(self):
        worker = InferenceWorker(Path("/unused"))
        worker.process = SimpleNamespace(
            returncode=None,
            stdin=SimpleNamespace(
                write=Mock(side_effect=BrokenPipeError), drain=AsyncMock()
            ),
        )
        with self.assertRaises(BrokenPipeError):
            await worker.submit(self.job().request)
        worker.process.stdin.write.assert_called_once()
        self.assertFalse(worker.jobs)

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
