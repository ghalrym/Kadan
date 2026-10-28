import asyncio
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

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


class D1RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        env = patch.dict(
            "os.environ",
            {"HIKARI_ENABLE_D1": "true", "KADAN_MODEL_DIR": self.directory.name},
        )
        env.start()
        self.addCleanup(env.stop)
        self.runtime = Runtime(
            "d1", {"devices": [{"uuid": "GPU-test", "vram_total": 24 * GIB}]}
        )
        self.runtime.socket = Socket()

    def request(self, **fields):
        return {
            "id": "d" * 32,
            "command": "submit",
            "operation": "generate",
            "input": {
                "feature": "decisions",
                "model": "d1-2b-bf16",
                "state": "State",
                "questions": [{"key": "q", "type": "Noul", "instructions": "True?"}],
            },
            **fields,
        }

    def test_disabled_d1_fails_before_machine_probe_or_connection(self):
        with (
            patch.dict("os.environ", {"HIKARI_ENABLE_D1": "false"}),
            patch("experimental_worker.runtime.machine") as probe,
        ):
            with self.assertRaisesRegex(ValueError, "HIKARI_ENABLE_D1"):
                Runtime("d1")
        probe.assert_not_called()

    async def test_d1_validation_precedes_queuing_and_never_accepts_chat(self):
        request = self.request()
        request["input"]["questions"][0]["type"] = "Text"
        await self.runtime.handle(request)
        self.assertTrue(self.runtime.queue.empty())
        self.assertEqual(self.runtime.socket.messages[-1]["status_code"], 422)
        await self.runtime.handle(self.request())
        self.assertEqual(self.runtime.queue.qsize(), 1)

    async def test_missing_d1_snapshot_requires_explicit_download_before_executor_spawn(
        self,
    ):
        with patch("multiprocessing.get_context") as process:
            await self.runtime.run_job(self.request())
        process.assert_not_called()
        self.assertEqual(self.runtime.socket.messages[-1]["status_code"], 409)

    async def test_native_settings_broadcast_still_succeeds_without_changing_d1(self):
        request = self.request(
            operation="configure", input={"feature": "tier_selection", "tier": "go"}
        )
        await self.runtime.handle(request)
        self.assertEqual(self.runtime.queue.qsize(), 1)
        await self.runtime.run_job(request)
        message = self.runtime.socket.messages[-1]
        self.assertEqual(message["state"], "succeeded")
        self.assertIsNone(message["result"]["selected_tier"])
        self.assertEqual(message["result"]["models"][0]["id"], "d1-2b-bf16")

    async def test_d1_requirement_is_distinct_from_laya_and_uses_one_gpu(self):
        await self.runtime.handle({"id": "req", "command": "requirement"})
        message = self.runtime.socket.messages[-1]
        self.assertEqual(message["model"], "d1-2b-bf16")
        self.assertEqual(message["minimum"], [8 * GIB, 22 * GIB])
        self.assertEqual(message["preferred"], message["minimum"])
        self.assertEqual(message["device_floor"], message["minimum"][1])

    async def test_d1_input_errors_keep_422_after_executor_cleanup(self):
        runtime = self.runtime
        runtime.store.complete = lambda: True
        runtime.budget = list(runtime.backend.REQUIREMENT)
        messages = [
            {
                "state": "error",
                "error": "Full prompt exceeds context",
                "status_code": 422,
            }
        ]
        process = SimpleNamespace(
            start=lambda: None, is_alive=lambda: False, join=Mock()
        )
        reader = SimpleNamespace(
            poll=lambda: bool(messages), recv=lambda: messages.pop(), close=lambda: None
        )
        writer = SimpleNamespace(close=lambda: None)
        context = SimpleNamespace(
            Pipe=lambda **kwargs: (reader, writer), Process=lambda **kwargs: process
        )
        with patch("multiprocessing.get_context", return_value=context):
            await runtime.run_job(self.request())
        process.join.assert_called_once_with()
        failed = next(
            message
            for message in runtime.socket.messages
            if message["state"] == "failed"
        )
        self.assertEqual(failed["status_code"], 422)
        self.assertEqual(runtime.budget, [0, 0])
