import asyncio
import io
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from api.services.inference_worker import (
    InferenceJob,
    InferenceTask,
    LocalMachine,
    encode_request,
)


class LocalMachineConfigurationTests(unittest.IsolatedAsyncioTestCase):
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
                self.assertEqual(LocalMachine().root, Path(expected))

    def test_explicit_root_overrides_model_directory_environment(self):
        with patch.dict(
            os.environ,
            {"HIKARI_MODEL_DIR": "/hikari-models", "KADAN_MODEL_DIR": "/legacy-models"},
            clear=True,
        ):
            self.assertEqual(LocalMachine(Path("/explicit")).root, Path("/explicit"))

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
                with self.assertRaisesRegex(OSError, "Launch disabled"):
                    await LocalMachine(Path("/unused")).start(
                        "ws://api/v1/workers", "t"
                    )
                launch.assert_awaited_once()
                self.assertEqual(launch.call_args.args[0], expected)

    async def test_launch_connects_back_with_token_without_changing_parent_environment(
        self,
    ):
        environment = {"UNRELATED_SETTING": "retained"}
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("pathlib.Path.mkdir"),
            patch("pathlib.Path.is_file", return_value=True),
            patch(
                "asyncio.create_subprocess_exec",
                side_effect=OSError("Launch disabled for configuration test"),
            ) as launch,
        ):
            with self.assertRaisesRegex(OSError, "Launch disabled"):
                await LocalMachine(Path("/models")).start(
                    "ws://127.0.0.1:8000/v1/workers", "secret"
                )
            self.assertEqual(
                launch.call_args.args[1:],
                ("/models/worker.lock", "ws://127.0.0.1:8000/v1/workers"),
            )
            self.assertEqual(
                launch.call_args.kwargs["env"],
                {**environment, "HIKARI_WORKER_TOKEN": "secret"},
            )
            self.assertEqual(dict(os.environ), environment)

    async def test_missing_executable_is_reported(self):
        with (
            patch.dict(os.environ, {"HIKARI_INFERENCE_WORKER": "relative"}, clear=True),
            self.assertRaisesRegex(Exception, "executable is unavailable"),
        ):
            LocalMachine(Path("/unused")).executable()


class LocalMachineSupervisionTests(unittest.IsolatedAsyncioTestCase):
    def process(self, returncode, stderr=b""):
        reader = asyncio.StreamReader()
        reader.feed_data(stderr)
        reader.feed_eof()
        return SimpleNamespace(
            pid=123,
            returncode=returncode,
            stderr=reader,
            wait=AsyncMock(return_value=returncode),
        )

    async def test_exit_is_recorded_and_the_process_restarted(self):
        machine = LocalMachine(Path("/unused"))
        machine.process = self.process(-11, b"x" * 10000 + b"last native diagnostic")
        restarted = self.process(0)

        async def launch(url, token):
            self.assertEqual((url, token), ("ws://api", "t"))
            machine.closing = True
            machine.process = restarted

        with (
            patch.object(machine, "launch", side_effect=launch),
            patch("api.services.inference_worker.os.killpg"),
            patch("api.services.inference_worker.asyncio.sleep", AsyncMock()),
            self.assertLogs("api.services.inference_worker", level="ERROR") as logs,
        ):
            await machine._supervise("ws://api", "t")
        self.assertIn("signal 11", logs.output[0])
        self.assertEqual(machine.last_exit.returncode, 0)
        self.assertIs(machine.process, restarted)

    async def test_signal_exit_keeps_bounded_stderr(self):
        machine = LocalMachine(Path("/unused"))
        machine.closing = True
        machine.process = self.process(-11, b"x" * 10000 + b"last native diagnostic")
        with patch("api.services.inference_worker.os.killpg"):
            await machine._supervise("ws://api", "t")
        self.assertEqual(machine.last_exit.signal, 11)
        self.assertEqual(len(machine.last_exit.stderr), 8192)
        self.assertTrue(machine.last_exit.stderr.endswith("last native diagnostic"))
        self.assertIn("signal 11", machine.last_exit.describe())

    async def test_close_stops_supervision_before_terminating(self):
        machine = LocalMachine(Path("/unused"))
        machine.supervisor = asyncio.create_task(asyncio.sleep(3600))
        process = self.process(None)
        process.terminate = lambda: setattr(process, "returncode", -15)
        machine.process = process
        with patch("api.services.inference_worker.os.killpg") as kill:
            await machine.close()
        self.assertTrue(machine.supervisor.cancelled())
        self.assertEqual(process.returncode, -15)
        kill.assert_called_once()
        self.assertIsNone(machine.process)

    async def test_no_restart_once_closing_during_backoff(self):
        machine = LocalMachine(Path("/unused"))
        machine.process = self.process(1)

        async def close_during_sleep(_):
            machine.closing = True

        with (
            patch.object(machine, "launch", new=AsyncMock()) as launch,
            patch("api.services.inference_worker.os.killpg"),
            patch(
                "api.services.inference_worker.asyncio.sleep",
                side_effect=close_during_sleep,
            ),
            self.assertLogs("api.services.inference_worker", level="ERROR"),
        ):
            await machine._supervise("ws://api", "t")
        launch.assert_not_awaited()


class TimingRelayTests(unittest.IsolatedAsyncioTestCase):
    def machine(self, chunks):
        machine = LocalMachine(Path("/unused"))
        machine.process = SimpleNamespace(
            stderr=SimpleNamespace(read=AsyncMock(side_effect=[*chunks, b""]))
        )
        return machine

    async def test_timing_lines_are_relayed_across_chunks(self):
        chunks = [
            b"ordinary diagnostic\nKAD",
            b'AN_IMAGE_TIMING {"step":',
            b'1}\nKADAN_IMAGE_TIMING {"step":2}\n',
            b"KADAN_ATEN_IMAGE_",
            b'TIMING {"phase":"load"}\nKADAN_ATEN_IMAGE_TIMING_EXTRA ignored\n',
            b'KADAN_ATEN_IMAGE_BATCH {"images":2,"host_peak_bytes":4096}\n',
            b'KADAN_ATEN_IMAGE_CACHE {"hits":10,"source_bytes":0}\n',
            b'KADAN_TEXT_TIMING {"phase":"decode","tokens":3}\n',
        ]
        machine = self.machine(chunks)
        output = io.StringIO()
        with patch("sys.stderr", output):
            await machine._stderr()
        self.assertEqual(
            output.getvalue(),
            'KADAN_IMAGE_TIMING {"step":1}\nKADAN_IMAGE_TIMING {"step":2}\n'
            'KADAN_ATEN_IMAGE_TIMING {"phase":"load"}\n'
            'KADAN_ATEN_IMAGE_BATCH {"images":2,"host_peak_bytes":4096}\n'
            'KADAN_ATEN_IMAGE_CACHE {"hits":10,"source_bytes":0}\n'
            'KADAN_TEXT_TIMING {"phase":"decode","tokens":3}\n',
        )
        self.assertEqual("".join(machine.diagnostics), b"".join(chunks).decode())

    async def test_oversized_and_incomplete_timing_lines_do_not_relay(self):
        chunks = [
            b"KADAN_ATEN_IMAGE_TIMING " + b"x" * 4100 + b"\n",
            b"KADAN_IMAGE_TIMING " + b"x" * 4000,
            b"x" * 100 + b'\nKADAN_IMAGE_TIMING {"step":1}\nKADAN_ATEN_IMAGE_',
            b"TIMING incomplete",
        ]
        machine = self.machine(chunks)
        output = io.StringIO()
        with patch("sys.stderr", output):
            await machine._stderr()
        self.assertEqual(output.getvalue(), 'KADAN_IMAGE_TIMING {"step":1}\n')

    async def test_timing_log_failure_keeps_native_diagnostics(self):
        line = b'KADAN_IMAGE_TIMING {"step":1}\n'
        machine = self.machine([line])
        with patch("sys.stderr.write", side_effect=OSError("log unavailable")):
            await machine._stderr()
        self.assertEqual("".join(machine.diagnostics), line.decode())


class RequestBoundTests(unittest.IsolatedAsyncioTestCase):
    def test_image_upload_has_a_larger_bounded_transport(self):
        encode_request(
            {"input": {"feature": "image", "image": "x" * (4 * 1024 * 1024)}}
        )
        with self.assertRaisesRegex(Exception, "transport limit"):
            encode_request(
                {"input": {"feature": "image", "image": "x" * (64 * 1024 * 1024)}}
            )
        with self.assertRaisesRegex(Exception, "transport limit"):
            encode_request(
                {"input": {"feature": "llm", "text": "x" * (4 * 1024 * 1024)}}
            )

    async def test_image_progress_is_initialized_with_the_job(self):
        job = InferenceJob(
            InferenceTask(
                input={"feature": "image", "prompt": "test", "count": 2, "steps": 20}
            ),
            False,
        )
        self.assertEqual(job.image_progress.image_count, 2)
        self.assertEqual(job.image_progress.steps_total, 20)
