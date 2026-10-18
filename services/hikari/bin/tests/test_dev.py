"""Exercise reload decisions without starting an API or inference worker."""

import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from bin import dev


class DevTests(unittest.TestCase):
    def test_start_bounds_request_drain_before_worker_shutdown(self):
        with patch.object(dev.subprocess, "Popen") as launch:
            dev.start({"KADAN_WORKER_DATA": "/source"})
        command = launch.call_args.args[0]
        timeout = command.index("--timeout-graceful-shutdown")
        self.assertEqual(command[timeout + 1], "5")
        self.assertEqual(launch.call_args.kwargs["cwd"], dev.ROOT)
        self.assertEqual(
            launch.call_args.kwargs["env"], {"KADAN_WORKER_DATA": "/source"}
        )

    def test_stop_waits_for_server_cleanup_before_returning(self):
        server = Mock()
        server.poll.return_value = None
        dev.stop(server)
        self.assertEqual(
            server.mock_calls, [call.poll(), call.terminate(), call.wait()]
        )

    def test_failed_compile_does_not_install(self):
        with (
            patch.object(dev.subprocess, "run", return_value=Mock(returncode=1)) as run,
            patch.object(dev.typer, "echo"),
        ):
            self.assertFalse(dev.build({}))
        self.assertEqual(run.call_count, 1)
        self.assertIn("--build", run.call_args.args[0])

    def check_reload(self, filename, build_succeeds, expected_builds, expected_starts):
        before = {Path(filename): 1}
        after = {Path(filename): 2}
        with (
            patch.object(dev.signal, "signal"),
            patch.object(dev, "environment", return_value={}),
            patch.object(dev, "snapshot", side_effect=[before, after]),
            patch.object(dev, "settled", return_value=after),
            patch.object(dev, "build", side_effect=[True, build_succeeds]) as build,
            patch.object(dev, "start") as start,
            patch.object(dev, "stop") as stop,
            patch.object(dev.time, "sleep", side_effect=[None, KeyboardInterrupt]),
            patch.object(dev.typer, "echo"),
            self.assertRaises(KeyboardInterrupt),
        ):
            lifecycle = Mock()
            lifecycle.attach_mock(start, "start")
            lifecycle.attach_mock(stop, "stop")
            dev.serve()
        self.assertEqual(build.call_count, expected_builds)
        self.assertEqual(start.call_count, expected_starts)
        self.assertEqual(stop.call_count, expected_starts)
        self.assertEqual(
            [event[0] for event in lifecycle.mock_calls],
            ["start", "stop"] * expected_starts,
        )

    def test_failed_rebuild_keeps_running_server(self):
        self.check_reload("worker.cpp", False, 2, 1)

    def test_successful_rebuild_restarts_server(self):
        self.check_reload("worker.cpp", True, 2, 2)

    def test_catalog_edit_restarts_without_rebuild(self):
        self.check_reload("catalog.json", True, 1, 2)

    def test_test_source_edit_does_not_restart(self):
        self.check_reload("tests/worker.cpp", True, 1, 1)
