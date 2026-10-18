"""Exercise reload decisions without starting an API or inference worker."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import typer
from bin import dev


class BuildTests(unittest.TestCase):
    def test_failed_compile_does_not_install(self):
        with (
            patch.object(dev.subprocess, "run", return_value=Mock(returncode=1)) as run,
            patch.object(dev.typer, "echo"),
        ):
            self.assertFalse(dev.build({}, "image"))
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertIn("--build", command)
        self.assertEqual(command[command.index("--target") + 1], "kadan-worker-image")

    def test_install_is_limited_to_the_worker_component(self):
        with patch.object(
            dev.subprocess, "run", return_value=Mock(returncode=0)
        ) as run:
            self.assertTrue(dev.build({}, "video"))
        install = run.call_args_list[1].args[0]
        self.assertEqual(
            install[install.index("--component") + 1], "kadan-worker-video"
        )

    def test_unknown_worker_is_rejected(self):
        with self.assertRaises(typer.BadParameter):
            dev.worker("gpu")


class WorkerReloadTests(unittest.TestCase):
    def run_worker(self, filename, builds, fingerprints, exits=False):
        """Runs one reload cycle; returns (builds, starts, stops)."""
        before = {Path(filename): 1}
        after = {Path(filename): 2}
        process = Mock()
        process.poll.return_value = 0 if exits else None
        with (
            patch.object(dev.signal, "signal"),
            patch.object(dev, "environment", return_value={}),
            patch.object(dev, "snapshot", side_effect=[before, after]),
            patch.object(dev, "settled", return_value=after),
            patch.object(dev, "build", side_effect=builds) as build,
            patch.object(dev, "fingerprint", side_effect=fingerprints),
            patch.object(dev, "start_worker", return_value=process) as start,
            patch.object(dev, "stop") as stop,
            patch.object(dev.time, "sleep", side_effect=[None, KeyboardInterrupt]),
            patch.object(dev.typer, "echo"),
            self.assertRaises(KeyboardInterrupt),
        ):
            dev.worker("llm")
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
        builds, starts, _ = self.run_worker("worker.cpp", [True], [], exits=True)
        self.assertEqual((builds, starts), (1, 2))
