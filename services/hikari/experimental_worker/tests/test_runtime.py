import asyncio
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from experimental_worker.runtime import GIB, Runtime, machine


class MachineTests(unittest.TestCase):
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
            if message["state"] == "running":
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

    async def test_requirement_keeps_whole_model_on_one_gpu(self):
        await self.runtime.handle({"id": "req", "command": "requirement"})
        reply = self.runtime.socket.messages[-1]
        self.assertEqual(reply["device_floor"], reply["minimum"][1])

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
            if message["state"] == "running":
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
