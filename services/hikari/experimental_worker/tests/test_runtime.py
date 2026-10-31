import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from experimental_worker.runtime import GIB, Runtime, execute_child, machine


class MachineTests(unittest.TestCase):
    def test_cpu_machine_preserves_ram_identity_without_gpu_probe(self):
        files = {
            "/proc/meminfo": f"MemTotal: {64 * GIB // 1024} kB\nMemAvailable: {48 * GIB // 1024} kB\n",
            "/proc/sys/kernel/random/boot_id": "shared-boot",
            "/sys/fs/cgroup/memory.max": str(24 * GIB),
            "/sys/fs/cgroup/memory.current": str(2 * GIB),
        }
        with (
            patch("pathlib.Path.read_text", new=lambda path: files[str(path)]),
            patch("pathlib.Path.exists", new=lambda path: str(path) in files),
            patch("subprocess.check_output", side_effect=AssertionError("GPU probe")),
        ):
            info = machine(cpu_only=True)
        self.assertEqual(info["id"], "shared-boot")
        self.assertEqual(info["devices"], [])
        self.assertEqual(info["ram_free"], 21 * GIB)
        self.assertEqual(info["ram_limit"], 23 * GIB)
        self.assertEqual(info["physical_ram_free"], 47 * GIB)

    def test_physical_ram_and_container_headroom_are_reported_separately(self):
        cases = (
            (str(24 * GIB), 400, 1, 22 * GIB, 23 * GIB),
            ("max", 400, 1, 399 * GIB, None),
            (None, 400, 1, 399 * GIB, None),
            (str(24 * GIB), 2, 1, GIB, 23 * GIB),
            (str(24 * GIB), 400, 20, 3 * GIB, 23 * GIB),
            (str(GIB // 2), 400, 1, 0, 0),
        )
        for limit, available, current, expected_free, expected_limit in cases:
            with self.subTest(limit=limit, available=available, current=current):
                files = {
                    "/proc/meminfo": (
                        f"MemTotal: {440 * GIB // 1024} kB\n"
                        f"MemAvailable: {available * GIB // 1024} kB\n"
                    ),
                    "/proc/sys/kernel/random/boot_id": "test-boot",
                }
                if limit is not None:
                    files["/sys/fs/cgroup/memory.max"] = limit
                    files["/sys/fs/cgroup/memory.current"] = str(current * GIB)
                with (
                    patch(
                        "pathlib.Path.read_text",
                        new=lambda path, files=files: files[str(path)],
                    ),
                    patch("pathlib.Path.exists", new=lambda path: str(path) in files),
                    patch(
                        "subprocess.check_output", return_value="GPU-test,24576,24000\n"
                    ),
                    patch.dict("os.environ", {"HIKARI_EXPERIMENTAL_GPU": "0"}),
                ):
                    info = machine()
                self.assertEqual(info["id"], "test-boot")
                self.assertEqual(info["devices"][0]["uuid"], "GPU-test")
                self.assertEqual(info["ram_free"], expected_free)
                self.assertEqual(info["ram_limit"], expected_limit)
                self.assertEqual(
                    info["ram_total"],
                    int(limit) if limit not in (None, "max") else 440 * GIB,
                )
                self.assertEqual(info["physical_ram_total"], 440 * GIB)
                self.assertEqual(info["physical_ram_free"], (available - 1) * GIB)


class Socket:
    def __init__(self):
        self.messages = []
        self.on_send = lambda message: None

    async def send(self, text):
        message = json.loads(text)
        self.messages.append(message)
        self.on_send(message)


class CpuExecutorTests(unittest.TestCase):
    def test_cpu_child_uses_backend_without_importing_torch_or_cuda_metrics(self):
        messages = []
        connection = SimpleNamespace(send=messages.append, close=lambda: None)
        backend = SimpleNamespace(execute=lambda *args: {"answers": []})
        with (
            patch("experimental_worker.runtime.ModelStore") as store,
            patch.dict("sys.modules", {"torch": None}),
            patch("builtins.print"),
            patch.dict("os.environ", {}, clear=True),
            patch("tempfile.tempdir", None),
            patch(
                "experimental_worker.runtime.importlib.import_module",
                return_value=backend,
            ),
        ):
            store.return_value.root = Path("/checkpoint")
            execute_child(
                connection,
                "d1",
                {"id": "d1-2b"},
                {},
                "/models",
                {"id": "a", "operation": "decision", "input": {}},
                None,
                [16 * GIB, 0],
                "/scratch",
            )
        self.assertEqual(messages[-1]["state"], "result")
        timing = messages[-1]["result"]["timing"]
        self.assertIn("peak_rss_bytes", timing)
        self.assertIn("executor_seconds", timing)
        self.assertFalse(any("cuda" in name for name in timing))

    def test_cpu_typed_answer_list_is_preserved_with_separate_metrics(self):
        answers = [{"key": "choice", "kind": "Choice", "value": "a"}]
        messages = []
        connection = SimpleNamespace(send=messages.append, close=lambda: None)
        backend = SimpleNamespace(execute=lambda *args: answers)
        with (
            patch("experimental_worker.runtime.ModelStore") as store,
            patch.dict("sys.modules", {"torch": None}),
            patch("builtins.print") as metric,
            patch.dict("os.environ", {}, clear=True),
            patch("tempfile.tempdir", None),
            patch(
                "experimental_worker.runtime.importlib.import_module",
                return_value=backend,
            ),
        ):
            store.return_value.root = Path("/checkpoint")
            execute_child(
                connection,
                "d1",
                {"id": "d1-2b-cpu-f32"},
                {},
                "/models",
                {"id": "a", "operation": "decision", "input": {}},
                None,
                [16 * GIB, 0],
                "/scratch",
            )
        self.assertIs(messages[-1]["result"], answers)
        record = json.loads(metric.call_args.args[0].removeprefix("LFM3_METRIC "))
        self.assertEqual(record["model"], "d1-2b-cpu-f32")
        self.assertIn("peak_rss_bytes", record["timing"])
        self.assertNotIn("answers", record)
        self.assertFalse(any("cuda" in name for name in record["timing"]))


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        env = patch.dict(
            "os.environ",
            {"HIKARI_ENABLE_LFM3_TEXT": "true", "KADAN_MODEL_DIR": self.directory.name},
        )
        env.start()
        self.addCleanup(env.stop)
        self.runtime = Runtime(
            "text", {"devices": [{"uuid": "GPU-test", "vram_total": 24 * 1024**3}]}
        )
        self.runtime.socket = Socket()

    def request(self, **fields):
        return {
            "id": "a" * 32,
            "command": "submit",
            "operation": "completion",
            "input": {"feature": "llm", "model": "lfm3-1.7b-bf16"},
            **fields,
        }

    async def test_default_audio_worker_reports_checkpoint_without_probe_or_download(
        self,
    ):
        with (
            patch.dict(
                "os.environ", {"KADAN_MODEL_DIR": self.directory.name}, clear=True
            ),
            patch("experimental_worker.runtime.machine") as probe,
            patch("multiprocessing.get_context") as process,
            patch("experimental_worker.store.ModelStore.download") as download,
        ):
            runtime = Runtime(
                "audio", {"devices": [{"uuid": "GPU-test", "vram_total": 24 * GIB}]}
            )
            runtime.socket = Socket()
            await runtime.models()
            self.assertEqual(runtime.kind, "lfm3_audio")
            model = runtime.socket.messages[-1]["models"]["models"][0]
            self.assertEqual(model["id"], "lfm3-audio-2b-bf16")
            self.assertEqual(model["status"], "not_downloaded")
            self.assertTrue(model["inference_available"])
            await runtime.run_job(
                {
                    "id": "a" * 32,
                    "command": "submit",
                    "operation": "completion",
                    "input": {
                        "feature": "audio_call",
                        "model_id": model["id"],
                        "mode": "transcribe",
                        "audio": "wav",
                    },
                }
            )
            self.assertEqual(runtime.socket.messages[-1]["status_code"], 409)
            probe.assert_not_called()
            process.assert_not_called()
            download.assert_not_called()

    async def test_unselected_model_does_not_enter_queue(self):
        await self.runtime.handle(self.request(input={"feature": "llm"}))
        self.assertTrue(self.runtime.queue.empty())
        self.assertEqual(self.runtime.socket.messages[-1]["state"], "rejected")

    async def test_missing_approved_snapshot_does_not_spawn_executor(self):
        with patch("multiprocessing.get_context") as process:
            await self.runtime.run_job(self.request())
        process.assert_not_called()
        self.assertEqual(self.runtime.socket.messages[-1]["status_code"], 409)

    async def test_cancellation_joins_executor_before_releasing_grant(self):
        runtime = self.runtime
        runtime.budget = list(runtime.backend.REQUIREMENT)
        runtime.store.complete = lambda: True
        process = SimpleNamespace(pid=0, live=False, joined=False)
        process.start = lambda: setattr(process, "live", True)
        process.is_alive = lambda: process.live
        process.terminate = lambda: setattr(process, "live", False)
        process.kill = process.terminate
        process.join = lambda *args: setattr(process, "joined", True)
        pipe = SimpleNamespace(poll=lambda: False, close=lambda: None)
        context = SimpleNamespace(
            Pipe=lambda **kwargs: (pipe, pipe), Process=lambda **kwargs: process
        )

        def observe(message):
            if message["state"] in ("loading", "running"):
                runtime.cancelled.add("a" * 32)
            if message["state"] in ("worker", "cancelled"):
                self.assertTrue(process.joined)
                self.assertFalse(process.live)

        runtime.socket.on_send = observe
        with patch("multiprocessing.get_context", return_value=context):
            await runtime.run_job(self.request())
        self.assertEqual(runtime.budget, [0, 0])
        self.assertIn(
            "cancelled", [message["state"] for message in runtime.socket.messages]
        )

    async def test_text_startup_waits_for_executor_running_event(self):
        runtime = self.runtime
        runtime.budget = list(runtime.backend.REQUIREMENT)
        runtime.store.complete = lambda: True
        messages = iter(
            [
                {"state": "loading"},
                {"state": "running"},
                {"state": "result", "result": {"text": "answer"}},
            ]
        )
        pending = 3

        def receive():
            nonlocal pending
            pending -= 1
            return next(messages)

        pipe = SimpleNamespace(
            poll=lambda: pending > 0, recv=receive, close=lambda: None
        )
        process = SimpleNamespace(
            start=lambda: None, is_alive=lambda: False, join=lambda: None
        )
        context = SimpleNamespace(
            Pipe=lambda **kwargs: (pipe, pipe), Process=lambda **kwargs: process
        )

        def observe(message):
            if message["state"] == "running":
                self.assertEqual(pending, 1)

        runtime.socket.on_send = observe
        with patch("multiprocessing.get_context", return_value=context):
            await runtime.run_job(self.request())
        states = [message["state"] for message in runtime.socket.messages]
        self.assertEqual(states[:4], ["preparing", "loading", "loading", "running"])
        self.assertEqual(states.count("running"), 1)
        self.assertIn("succeeded", states)

    async def test_requirement_keeps_whole_model_on_one_gpu(self):
        await self.runtime.handle({"id": "req", "command": "requirement"})
        reply = self.runtime.socket.messages[-1]
        self.assertEqual(reply["device_floor"], reply["minimum"][1])

    async def test_cancelled_executor_scratch_is_removed_after_join(self):
        runtime = self.runtime
        runtime.budget = list(runtime.backend.REQUIREMENT)
        runtime.store.complete = lambda: True
        process = SimpleNamespace(pid=0, live=False, joined=False)
        pipe = SimpleNamespace(poll=lambda: False, close=lambda: None)
        scratch = None

        def create_process(**kwargs):
            nonlocal scratch
            scratch = Path(kwargs["args"][-1])

            def start():
                process.live = True
                directory = scratch / "lfm3-audio-interrupted"
                directory.mkdir()
                (directory / "utterance.wav").write_bytes(b"utterance")

            def join():
                self.assertFalse(process.live)
                self.assertTrue(
                    (scratch / "lfm3-audio-interrupted/utterance.wav").exists()
                )
                process.joined = True

            process.start = start
            process.join = join
            return process

        process.is_alive = lambda: process.live
        process.terminate = lambda: setattr(process, "live", False)
        process.kill = process.terminate
        context = SimpleNamespace(
            Pipe=lambda **kwargs: (pipe, pipe), Process=create_process
        )

        def observe(message):
            if message["state"] in ("loading", "running"):
                runtime.cancelled.add("a" * 32)
            if message["state"] == "cancelled":
                self.assertTrue(process.joined)
                self.assertFalse(scratch.exists())

        runtime.socket.on_send = observe
        with patch("multiprocessing.get_context", return_value=context):
            await runtime.run_job(self.request())
        self.assertFalse(scratch.exists())
        self.assertTrue(process.joined)

    async def test_disconnect_waits_for_cleanup_despite_repeated_cancellation(self):
        runtime = self.runtime
        granted = list(runtime.backend.REQUIREMENT)
        runtime.budget = granted.copy()
        runtime.store.complete = lambda: True
        cleanup_waiting, exited = asyncio.Event(), asyncio.Event()
        process = SimpleNamespace(pid=0, live=False, joined=False, terminating=False)
        process.start = lambda: setattr(process, "live", True)
        process.is_alive = lambda: process.live
        process.terminate = lambda: setattr(process, "terminating", True)
        process.kill = process.terminate

        def join():
            self.assertFalse(process.live)
            process.joined = True

        process.join = join
        pipe = SimpleNamespace(poll=lambda: False, close=lambda: None)
        context = SimpleNamespace(
            Pipe=lambda **kwargs: (pipe, pipe), Process=lambda **kwargs: process
        )
        original_sleep = asyncio.sleep

        async def cleanup_pause(delay):
            self.assertTrue(process.terminating)
            cleanup_waiting.set()
            await exited.wait()

        def cancel_request(message):
            if message["state"] in ("loading", "running"):
                runtime.cancelled.add("a" * 32)

        runtime.socket.on_send = cancel_request
        with (
            patch("multiprocessing.get_context", return_value=context),
            patch("experimental_worker.runtime.asyncio.sleep", new=cleanup_pause),
        ):
            job = asyncio.create_task(runtime.run_job(self.request()))
            try:
                await cleanup_waiting.wait()
                job.cancel()
                await original_sleep(0)
                first_finished_early = job.done()
                job.cancel()
                await original_sleep(0)
                second_finished_early = job.done()
                self.assertIs(runtime.child, process)
                self.assertEqual(runtime.budget, granted)
            finally:
                process.live = False
                exited.set()
                with self.assertRaises(asyncio.CancelledError):
                    await job
        self.assertFalse(first_finished_early)
        self.assertFalse(second_finished_early)
        self.assertTrue(process.joined)
        self.assertIsNone(runtime.child)
        self.assertEqual(runtime.budget, [0, 0])
        self.assertFalse(
            {"cancelled", "succeeded", "failed", "released"}
            & {message["state"] for message in runtime.socket.messages}
        )

    async def test_grant_arriving_during_wait_event_is_not_lost(self):
        runtime = self.runtime
        runtime.store.complete = lambda: True

        def grant(message):
            if message["state"] == "waiting_for_resources":
                runtime.budget = list(runtime.backend.REQUIREMENT)
                runtime.granted.set()

        runtime.socket.on_send = grant
        with patch(
            "multiprocessing.get_context", side_effect=RuntimeError("grant observed")
        ):
            with self.assertRaisesRegex(RuntimeError, "grant observed"):
                await asyncio.wait_for(runtime.run_job(self.request()), 0.1)

    def test_disabled_worker_fails_before_probe_or_connection(self):
        with patch.dict("os.environ", {"HIKARI_ENABLE_LFM3_TEXT": "false"}):
            with self.assertRaisesRegex(ValueError, "requires HIKARI_ENABLE_LFM3_TEXT"):
                Runtime("text")

    async def test_reconnect_discards_stale_work_and_requires_a_fresh_grant(self):
        runtime = self.runtime
        calls = 0

        async def session():
            nonlocal calls
            calls += 1
            if calls == 1:
                runtime.budget = list(runtime.backend.REQUIREMENT)
                runtime.queue.put_nowait(self.request())
                runtime.cancelled.add("stale")
                runtime.granted.set()
                raise OSError("API restarted")
            self.assertEqual(runtime.budget, [0, 0])
            self.assertTrue(runtime.queue.empty())
            self.assertFalse(runtime.cancelled)
            self.assertFalse(runtime.granted.is_set())
            self.assertIsNone(runtime.socket)
            raise asyncio.CancelledError()

        runtime.serve_session = session
        with patch("asyncio.sleep", new_callable=AsyncMock) as pause:
            with self.assertRaises(asyncio.CancelledError):
                await runtime.serve()
        self.assertEqual(calls, 2)
        pause.assert_awaited_once_with(1)


class CpuRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.backend = SimpleNamespace(
            MODEL={"id": "d1-2b-cpu-f32", "revision": "pinned"},
            REQUIREMENT=(16 * GIB, 0),
            MAX_EXECUTOR_SECONDS=180,
        )
        with (
            patch.dict(
                "os.environ",
                {
                    "KADAN_MODEL_DIR": self.directory.name,
                },
            ),
            patch("pathlib.Path.read_text", return_value='{"artifacts": []}'),
            patch(
                "experimental_worker.runtime.importlib.import_module",
                return_value=self.backend,
            ),
        ):
            self.runtime = Runtime("d1", {"devices": []})
        self.runtime.socket = Socket()
        self.runtime.models = AsyncMock()

    def request(self, **fields):
        return {
            "id": "a" * 32,
            "command": "submit",
            "operation": "generate",
            "input": {"feature": "decisions", "model_id": "d1-2b-cpu-f32"},
            **fields,
        }

    def test_default_d1_needs_no_flag_and_never_probes_gpu(self):
        with (
            patch.dict("os.environ", {"HIKARI_ENABLE_D1_DECISIONS": "false"}),
            patch(
                "experimental_worker.runtime.machine", return_value={"devices": []}
            ) as probe,
            patch("pathlib.Path.read_text", return_value='{"artifacts": []}'),
            patch(
                "experimental_worker.runtime.importlib.import_module",
                return_value=self.backend,
            ),
        ):
            runtime = Runtime("d1")
        probe.assert_called_once_with(cpu_only=True)
        self.assertTrue(runtime.cpu_only)
        self.assertEqual(runtime.kind, "lfm3_decision")

    async def test_cpu_grant_reports_loaded_without_vram(self):
        await self.runtime.handle(
            {"id": "grant", "command": "load", "budget": [16 * GIB]}
        )
        self.assertEqual(self.runtime.budget, [16 * GIB, 0])
        await self.runtime.handle({"id": "status", "command": "status"})
        self.assertTrue(self.runtime.socket.messages[0]["loaded"])
        self.assertTrue(self.runtime.socket.messages[-1]["workers"][0]["loaded"])
        await self.runtime.handle({"id": "requirement", "command": "requirement"})
        self.assertEqual(self.runtime.socket.messages[-1]["minimum"], [16 * GIB, 0])
        self.assertEqual(self.runtime.socket.messages[-1]["device_floor"], 0)

    async def test_cpu_grant_cannot_allocate_device_memory(self):
        await self.runtime.handle(
            {"id": "bad", "command": "load", "budget": [16 * GIB, GIB]}
        )
        self.assertEqual(self.runtime.socket.messages[-1]["state"], "rejected")
        self.assertEqual(self.runtime.budget, [0, 0])

    async def test_missing_checkpoint_does_not_spawn_cpu_executor(self):
        with patch("multiprocessing.get_context") as context:
            await self.runtime.run_job(self.request())
        context.assert_not_called()
        self.assertEqual(self.runtime.socket.messages[-1]["status_code"], 409)

    async def test_queued_cancel_does_not_spawn_or_release_unrelated_grant(self):
        self.runtime.budget = [16 * GIB, 0]
        await self.runtime.handle({"id": "a" * 32, "command": "cancel"})
        with patch("multiprocessing.get_context") as context:
            await self.runtime.run_job(self.request())
        context.assert_not_called()
        self.assertEqual(self.runtime.budget, [16 * GIB, 0])
        self.assertEqual(self.runtime.socket.messages[-1]["state"], "cancelled")

    async def test_missing_ram_grant_waits_and_cancel_never_spawns(self):
        self.runtime.store.complete = lambda: True
        self.runtime.socket.on_send = lambda message: (
            (self.runtime.cancelled.add("a" * 32), self.runtime.granted.set())
            if message["state"] == "waiting_for_resources"
            else None
        )
        with patch("multiprocessing.get_context") as context:
            await self.runtime.run_job(self.request())
        context.assert_not_called()
        self.assertEqual(
            [message["state"] for message in self.runtime.socket.messages],
            ["waiting_for_resources", "cancelled"],
        )

    async def test_cpu_active_cancel_joins_before_grant_release(self):
        await RuntimeTests.test_cancellation_joins_executor_before_releasing_grant(self)

    async def test_cpu_repeated_disconnect_cancellation_keeps_grant_until_join(self):
        await RuntimeTests.test_disconnect_waits_for_cleanup_despite_repeated_cancellation(
            self
        )

    async def test_cpu_timeout_and_ram_overrun_stop_and_join_before_release(self):
        runtime = self.runtime
        for failure in ("time", "RAM"):
            with self.subTest(failure=failure):
                runtime.socket.messages.clear()
                runtime.budget = list(self.backend.REQUIREMENT)
                runtime.store.complete = lambda: True
                process = SimpleNamespace(pid=123, live=False, joined=False)
                process.start = lambda process=process: setattr(process, "live", True)
                process.is_alive = lambda process=process: process.live
                process.terminate = lambda process=process: setattr(
                    process, "live", False
                )
                process.kill = process.terminate
                process.join = lambda process=process: setattr(process, "joined", True)
                pipe = SimpleNamespace(poll=lambda: False, close=lambda: None)

                def create_process(process=process, **kwargs):
                    self.assertIsNone(kwargs["args"][6])
                    return process

                context = SimpleNamespace(
                    Pipe=lambda pipe=pipe, **kwargs: (pipe, pipe),
                    Process=create_process,
                )

                def observe(message, process=process):
                    if message["state"] in ("failed", "worker"):
                        self.assertTrue(process.joined)
                        self.assertFalse(process.live)
                        self.assertEqual(runtime.budget, [0, 0])

                runtime.socket.on_send = observe
                with (
                    patch("multiprocessing.get_context", return_value=context),
                    patch(
                        "experimental_worker.runtime.time",
                        SimpleNamespace(
                            monotonic=iter(
                                [0, 181] if failure == "time" else [0, 1]
                            ).__next__
                        ),
                    ),
                    patch(
                        "pathlib.Path.read_text",
                        return_value=f"VmRSS: {17 * GIB // 1024} kB",
                    ),
                ):
                    await runtime.run_job(self.request())
                failed = next(
                    message
                    for message in runtime.socket.messages
                    if message["state"] == "failed"
                )
                self.assertIn(failure, failed["error"])
                self.assertEqual(failed["status_code"], 502)
                self.assertIsNone(runtime.child)
