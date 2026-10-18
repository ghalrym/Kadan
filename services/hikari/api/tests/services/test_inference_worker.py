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
from api.services.inference_errors import InferenceFailure


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
            b'KADAN_ATEN_IMAGE_CACHE {"hits":10,"source_bytes":0}\n',
            b'KADAN_TEXT_TIMING {"phase":"decode","tokens":3}\n',
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
            'KADAN_ATEN_IMAGE_BATCH {"images":2,"host_peak_bytes":4096}\n'
            'KADAN_ATEN_IMAGE_CACHE {"hits":10,"source_bytes":0}\n'
            'KADAN_TEXT_TIMING {"phase":"decode","tokens":3}\n',
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


class ConcurrentWorkerEventsTests(unittest.IsolatedAsyncioTestCase):
    async def test_readiness_requires_residency_protocol(self):
        for protocol in (2, 3):
            with self.subTest(protocol=protocol):
                worker = InferenceWorker(Path("/unused"))
                worker.closing = True
                worker.ready = asyncio.get_running_loop().create_future()
                stdout = asyncio.StreamReader()
                stdout.feed_data(
                    (
                        json.dumps({"state": "ready", "protocol": protocol, "pid": 123})
                        + "\n"
                    ).encode()
                )
                stdout.feed_eof()
                worker.process = SimpleNamespace(
                    stdout=stdout,
                    pid=123,
                    returncode=0,
                    wait=AsyncMock(return_value=0),
                )
                await worker._read()
                if protocol == 3:
                    self.assertEqual(await worker.ready, 123)
                    self.assertIsNone(worker.last_exit.protocol_error)
                else:
                    with self.assertRaisesRegex(
                        Exception, "Invalid worker readiness response"
                    ):
                        await worker.ready

    async def test_interleaved_events_route_to_their_job_ids(self):
        observed = []
        worker = InferenceWorker(
            Path("/unused"), lambda job, event: observed.append((job.request.id, event))
        )
        worker.closing = True
        worker.ready = asyncio.get_running_loop().create_future()
        worker.ready.set_result(123)
        first, second = [
            InferenceJob(
                InferenceTask(
                    operation="completion",
                    input={
                        "feature": "llm",
                        "messages": [{"role": "user", "text": text}],
                    },
                ),
                True,
            )
            for text in ("first", "second")
        ]
        for job in (first, second):
            worker.jobs[job.request.id] = job
        stdout = asyncio.StreamReader()
        for job, event in (
            (first, {"state": "running"}),
            (second, {"state": "running"}),
            (second, {"state": "content", "content": "second chunk"}),
            (first, {"state": "content", "content": "first chunk"}),
            (second, {"state": "progress", "phase": "decode_token", "value": 2}),
            (first, {"state": "progress", "phase": "decode_token", "value": 1}),
            (second, {"state": "succeeded", "result": {"text": "second result"}}),
            (first, {"state": "succeeded", "result": {"text": "first result"}}),
        ):
            stdout.feed_data(
                (json.dumps({"id": job.request.id, **event}) + "\n").encode()
            )
        stdout.feed_eof()
        worker.process = SimpleNamespace(
            stdout=stdout, pid=123, returncode=0, wait=AsyncMock(return_value=0)
        )
        await worker._read()
        for job, name, value in ((first, "first", 1), (second, "second", 2)):
            self.assertEqual(await job.result, {"text": f"{name} result"})
            self.assertEqual(job.events.get_nowait(), {"content": f"{name} chunk"})
            progress = [
                event
                for job_id, event in observed
                if job_id == job.request.id and event.state == "progress"
            ]
            self.assertEqual([event.value for event in progress], [value])
        self.assertIsNone(worker.last_exit.protocol_error)


class WorkerShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_close_reaps_late_spawn_even_when_start_caller_is_cancelled(self):
        for cancel_start in (False, True):
            with self.subTest(cancel_start=cancel_start):
                worker = InferenceWorker(Path("/unused"))
                spawning, spawned, stopped = (
                    asyncio.Event(),
                    asyncio.Event(),
                    asyncio.Event(),
                )
                process = SimpleNamespace(
                    returncode=None,
                    stdin=SimpleNamespace(close=Mock(side_effect=stopped.set)),
                    wait=AsyncMock(return_value=0),
                )

                async def spawn(*args, **kwargs):
                    spawning.set()
                    await spawned.wait()
                    return process

                async def read():
                    await stopped.wait()
                    worker.ready.set_exception(InferenceFailure("Stopped before ready"))

                with (
                    patch("pathlib.Path.mkdir"),
                    patch("pathlib.Path.is_file", return_value=True),
                    patch(
                        "asyncio.create_subprocess_exec", side_effect=spawn
                    ) as launch,
                    patch.object(worker, "_read", side_effect=read),
                    patch.object(worker, "_stderr", new=AsyncMock()),
                ):
                    starting = asyncio.create_task(worker.start())
                    await spawning.wait()
                    closing = asyncio.create_task(worker.close())
                    if cancel_start:
                        starting.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(closing.done())
                    self.assertIsNone(worker.process)
                    spawned.set()
                    await closing
                    with self.assertRaises(
                        asyncio.CancelledError if cancel_start else InferenceFailure
                    ):
                        await starting
                    await worker.close()
                launch.assert_awaited_once()
                process.stdin.close.assert_called_once()
                process.wait.assert_awaited_once()
                self.assertIsNone(worker.process)

    async def test_close_preserves_each_terminal_callback_and_cancellation_error(self):
        events = []
        worker = InferenceWorker(
            Path("/unused"), on_event=lambda job, event: events.append((job, event))
        )
        worker.ready = asyncio.get_running_loop().create_future()
        worker.ready.set_result(123)
        jobs = [
            InferenceJob(
                InferenceTask(input={"feature": "image", "prompt": "test"}), False
            )
            for _ in range(2)
        ]
        for job in jobs:
            job.accepted.set_result(None)
            worker.jobs[job.request.id] = job
        stdout = asyncio.StreamReader()

        def stop():
            for job in jobs:
                stdout.feed_data(
                    (
                        json.dumps({"id": job.request.id, "state": "cancelled"}) + "\n"
                    ).encode()
                )
            stdout.feed_eof()
            worker.process.returncode = 0

        worker.process = SimpleNamespace(
            pid=123,
            returncode=None,
            stdin=SimpleNamespace(close=Mock(side_effect=stop)),
            stdout=stdout,
            wait=AsyncMock(return_value=0),
        )
        worker.reader = asyncio.create_task(worker._read())
        await worker.close()
        self.assertEqual(
            [(job, event.state) for job, event in events],
            [(job, "cancelled") for job in jobs],
        )
        for job in jobs:
            with self.assertRaises(InferenceFailure) as failure:
                await job.result
            self.assertEqual(failure.exception.status_code, 499)
        self.assertIsNone(worker.last_exit.protocol_error)

    async def test_concurrent_close_and_cancelled_caller_share_teardown(self):
        worker = InferenceWorker(Path("/unused"))
        stopped, reaped = asyncio.Event(), asyncio.Event()

        async def wait():
            await reaped.wait()
            return 0

        process = SimpleNamespace(
            returncode=None,
            stdin=SimpleNamespace(close=Mock(side_effect=stopped.set)),
            wait=AsyncMock(side_effect=wait),
        )
        worker.process = process
        first = asyncio.create_task(worker.close())
        await stopped.wait()
        second = asyncio.create_task(worker.close())
        first.cancel()
        await asyncio.sleep(0)
        self.assertFalse(first.done())
        self.assertFalse(second.done())
        with self.assertRaisesRegex(InferenceFailure, "stopping"):
            await worker.submit(
                InferenceTask(input={"feature": "image", "prompt": "test"})
            )
        reaped.set()
        with self.assertRaises(asyncio.CancelledError):
            await first
        await second
        await worker.close()
        process.stdin.close.assert_called_once()
        process.wait.assert_awaited_once()
        self.assertIsNone(worker.process)

    async def test_submission_waiting_for_write_lock_is_rejected_on_shutdown(self):
        worker = InferenceWorker(Path("/unused"))
        process = SimpleNamespace(
            returncode=None,
            stdin=SimpleNamespace(close=Mock(), write=Mock(), drain=AsyncMock()),
            wait=AsyncMock(return_value=0),
        )
        worker.process = process
        await worker.write_lock.acquire()
        submission = asyncio.create_task(
            worker.submit(InferenceTask(input={"feature": "image", "prompt": "test"}))
        )
        await asyncio.sleep(0)
        self.assertEqual(len(worker.jobs), 1)
        await worker.close()
        worker.write_lock.release()
        with self.assertRaisesRegex(InferenceFailure, "stopping"):
            await submission
        process.stdin.write.assert_not_called()
        self.assertFalse(worker.jobs)

    async def test_forced_reap_waits_for_reader_before_releasing_process(self):
        worker = InferenceWorker(Path("/unused"))
        killed, drained = asyncio.Event(), asyncio.Event()
        process = SimpleNamespace(
            pid=123,
            returncode=None,
            stdin=SimpleNamespace(close=Mock()),
            wait=AsyncMock(side_effect=[TimeoutError(), -9]),
        )
        worker.process = process

        async def read():
            await killed.wait()
            await drained.wait()
            self.assertIs(worker.process, process)

        worker.reader = asyncio.create_task(read())
        with patch(
            "api.services.inference_worker.os.killpg",
            side_effect=lambda *_: killed.set(),
        ) as kill:
            closing = asyncio.create_task(worker.close())
            await killed.wait()
            self.assertFalse(closing.done())
            drained.set()
            await closing
        kill.assert_called_once_with(123, 9)
        self.assertEqual(process.wait.await_count, 2)
        self.assertIsNone(worker.process)


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
