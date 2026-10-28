"""Exercise reload decisions without starting an API or inference worker."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import typer
from bin import run


class BuildTests(unittest.TestCase):
    def test_failed_compile_does_not_install(self):
        with (
            patch.object(
                run.subprocess, "run", return_value=Mock(returncode=1)
            ) as execute,
            patch.object(run.typer, "echo"),
        ):
            self.assertFalse(run.build({}, "image"))
        self.assertEqual(execute.call_count, 1)
        command = execute.call_args.args[0]
        self.assertIn("--build", command)
        self.assertEqual(command[command.index("--target") + 1], "kadan-worker-image")

    def test_install_is_limited_to_the_worker_component(self):
        with patch.object(
            run.subprocess, "run", return_value=Mock(returncode=0)
        ) as execute:
            self.assertTrue(run.build({}, "video"))
        install = execute.call_args_list[1].args[0]
        self.assertEqual(
            install[install.index("--component") + 1], "kadan-worker-video"
        )

    def test_unknown_worker_is_rejected(self):
        with self.assertRaises(typer.BadParameter):
            run.worker("gpu")

    def test_retired_workers_cannot_start_or_build(self):
        with (
            patch.object(run, "build") as build,
            patch.object(run, "start_worker") as start,
        ):
            for name in ("llm", "tts", "formatting"):
                with self.subTest(name=name), self.assertRaises(typer.BadParameter):
                    run.worker(name)
        build.assert_not_called()
        start.assert_not_called()


class WorkerReloadTests(unittest.TestCase):
    def run_worker(self, filename, builds, fingerprints, exits=False):
        """Runs one reload cycle; returns (builds, starts, stops)."""
        before = {Path(filename): 1}
        after = {Path(filename): 2}
        process = Mock()
        process.poll.return_value = 0 if exits else None
        with (
            patch.object(run.signal, "signal"),
            patch.object(run, "environment", return_value={}),
            patch.object(run, "snapshot", side_effect=[before, after]),
            patch.object(run, "settled", return_value=after),
            patch.object(run, "build", side_effect=builds) as build,
            patch.object(run, "fingerprint", side_effect=fingerprints),
            patch.object(run, "start_worker", return_value=process) as start,
            patch.object(run, "stop") as stop,
            patch.object(run.time, "sleep", side_effect=[None, KeyboardInterrupt]),
            patch.object(run.typer, "echo"),
            self.assertRaises(KeyboardInterrupt),
        ):
            run.worker("image")
        return build.call_count, start.call_count, stop.call_count

    def test_a_changed_binary_restarts_only_this_worker(self):
        self.assertEqual(self.run_worker("worker.cpp", [True, True], [1, 2]), (2, 2, 2))

    def test_an_edit_that_leaves_this_binary_unchanged_does_not_restart(self):
        self.assertEqual(self.run_worker("worker.cpp", [True, True], [1, 1]), (2, 1, 1))

    def test_failed_rebuild_keeps_the_running_worker(self):
        self.assertEqual(self.run_worker("worker.cpp", [True, False], [1]), (2, 1, 1))

    def test_catalog_edit_restarts_without_rebuild(self):
        self.assertEqual(self.run_worker("catalog.json", [True], []), (1, 2, 2))

    def test_test_source_edit_does_not_restart(self):
        self.assertEqual(self.run_worker("tests/worker.cpp", [True], []), (1, 1, 1))

    def test_an_exited_worker_is_restarted(self):
        builds, starts, _ = self.run_worker("tests/worker.cpp", [True], [], exits=True)
        self.assertEqual((builds, starts), (1, 2))

    def test_source_fix_rebuilds_a_worker_after_repeated_early_exits(self):
        broken = {Path("worker.cpp"): 1}
        fixed = {Path("worker.cpp"): 2}
        crashed = Mock()
        crashed.poll.return_value = 1
        recovered = Mock()
        recovered.poll.return_value = None
        with (
            patch.object(run.signal, "signal"),
            patch.object(run, "environment", return_value={}),
            patch.object(run, "snapshot", side_effect=[broken, broken, fixed, fixed]),
            patch.object(run, "settled", return_value=fixed),
            patch.object(run, "build", return_value=True) as build,
            patch.object(run, "fingerprint", side_effect=[1, 2]),
            patch.object(
                run, "start_worker", side_effect=[crashed, crashed, recovered]
            ) as start,
            patch.object(run, "stop"),
            patch.object(
                run.time, "sleep", side_effect=[None, None, None, KeyboardInterrupt]
            ),
            patch.object(run.typer, "echo"),
            self.assertRaises(KeyboardInterrupt),
        ):
            run.worker("image")
        self.assertEqual(build.call_count, 2)
        self.assertEqual(start.call_count, 3)

    def test_failed_rebuild_still_restarts_an_exited_worker(self):
        self.assertEqual(
            self.run_worker("worker.cpp", [True, False], [1], exits=True), (2, 2, 2)
        )
