"""Dispatcher ordering and restart behavior with GitHub, Git and Docker replaced."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from bin import dispatch, rollout
from bin.tests.test_rollout import manifest


def run(number=2, source="b"):
    return {
        "id": number * 10,
        "run_number": number,
        "run_attempt": 1,
        "path": ".github/workflows/compose-release.yml",
        "head_branch": "master",
        "event": "push",
        "status": "completed",
        "conclusion": "success",
        "repository": {"full_name": "ghalrym/Kadan"},
        "head_sha": source * 40,
    }


class DispatcherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.state = Path(self.directory.name)
        self.backend = Mock(spec=rollout.Compose)
        self.current = manifest()
        self.target = manifest("b", "2")
        self.resume = patch.object(dispatch, "resume_locked", return_value=self.current)
        self.resume.start()
        self.addCleanup(self.resume.stop)

    def test_newest_successful_publication_is_selected(self):
        with patch.object(
            dispatch.subprocess,
            "run",
            return_value=Mock(
                stdout=json.dumps([{"workflow_runs": [run(1), run(3), run(2)]}])
            ),
        ):
            self.assertEqual(
                [value["id"] for value in dispatch.successful_runs()], [30, 20, 10]
            )

    def test_untrusted_run_is_refused(self):
        value = {**run(), "event": "pull_request"}
        with (
            patch.object(
                dispatch.subprocess,
                "run",
                return_value=Mock(stdout=json.dumps([{"workflow_runs": [value]}])),
            ),
            self.assertRaises(ValueError),
        ):
            dispatch.successful_runs()

    def test_success_advances_cursor_only_after_apply(self):
        def apply(*args):
            self.assertFalse((self.state / "dispatcher.json").exists())

        with (
            patch.object(dispatch, "successful_runs", return_value=[run()]),
            patch.object(dispatch, "download_release", return_value=self.target),
            patch.object(dispatch.subprocess, "run"),
            patch.object(dispatch, "ancestor", return_value=False),
            patch.object(dispatch, "rollout_locked", side_effect=apply) as apply,
        ):
            self.assertEqual(
                dispatch.poll_once(self.state, self.state, self.backend), 20
            )
        apply.assert_called_once_with(self.target, self.state, self.backend)
        self.assertEqual(
            json.loads((self.state / "dispatcher.json").read_text())["run_id"], 20
        )

    def test_failed_apply_remains_unconsumed_for_retry(self):
        with (
            patch.object(dispatch, "successful_runs", return_value=[run()]),
            patch.object(dispatch, "download_release", return_value=self.target),
            patch.object(dispatch.subprocess, "run"),
            patch.object(dispatch, "ancestor", return_value=False),
            patch.object(
                dispatch, "rollout_locked", side_effect=RuntimeError("failed health")
            ),
            self.assertRaises(RuntimeError),
        ):
            dispatch.poll_once(self.state, self.state, self.backend)
        self.assertFalse((self.state / "dispatcher.json").exists())

    def test_restart_cursor_makes_delivery_idempotent(self):
        rollout.atomic_json(
            self.state / "dispatcher.json", {"processed": ["20:1"], "run_id": 20}
        )
        with (
            patch.object(dispatch, "successful_runs", return_value=[run()]),
            patch.object(dispatch, "download_release") as download,
        ):
            self.assertIsNone(dispatch.poll_once(self.state, self.state, self.backend))
        download.assert_not_called()

    def test_proven_older_release_is_consumed_without_downgrade(self):
        with (
            patch.object(dispatch, "successful_runs", return_value=[run()]),
            patch.object(dispatch, "download_release", return_value=self.target),
            patch.object(dispatch.subprocess, "run"),
            patch.object(dispatch, "ancestor", return_value=True),
            patch.object(dispatch, "rollout_locked") as apply,
        ):
            self.assertIsNone(dispatch.poll_once(self.state, self.state, self.backend))
        apply.assert_not_called()

    def test_pending_recovery_precedes_network(self):
        with (
            patch.object(
                dispatch,
                "resume_locked",
                side_effect=RuntimeError("recovery incomplete"),
            ),
            patch.object(dispatch, "successful_runs") as network,
            self.assertRaises(RuntimeError),
        ):
            dispatch.poll_once(self.state, self.state, self.backend)
        network.assert_not_called()

    def test_termination_does_not_advance_cursor_or_run_recovery(self):
        with (
            patch.object(dispatch, "successful_runs", return_value=[run()]),
            patch.object(dispatch, "download_release", return_value=self.target),
            patch.object(dispatch.subprocess, "run"),
            patch.object(dispatch, "ancestor", return_value=False),
            patch.object(dispatch, "rollout_locked", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            dispatch.poll_once(self.state, self.state, self.backend)
        self.assertFalse((self.state / "dispatcher.json").exists())

    def test_git_daemon_failure_is_not_misclassified_as_stale(self):
        with (
            patch.object(dispatch.subprocess, "run", return_value=Mock(returncode=128)),
            self.assertRaises(RuntimeError),
        ):
            dispatch.ancestor(self.state, "a" * 40, "b" * 40)

    def test_poll_failure_is_visible_and_backs_off(self):
        with (
            patch.object(dispatch, "Compose", return_value=self.backend),
            patch.object(
                dispatch,
                "poll_once",
                side_effect=dispatch.DiscoveryError("gh unavailable"),
            ),
            patch.object(dispatch.logger, "exception") as log,
            patch.object(
                dispatch.time, "sleep", side_effect=KeyboardInterrupt
            ) as sleep,
            self.assertRaises(KeyboardInterrupt),
        ):
            dispatch.watch(self.state, self.state)
        log.assert_called_once()
        sleep.assert_called_once_with(120)

    def test_high_number_stale_run_does_not_hide_lower_number_descendant(self):
        stale = run(9, "c")
        descendant = run(2, "b")
        with (
            patch.object(dispatch, "successful_runs", return_value=[stale, descendant]),
            patch.object(dispatch, "download_release", return_value=self.target),
            patch.object(dispatch.subprocess, "run"),
            patch.object(dispatch, "ancestor", side_effect=[True, False]),
            patch.object(dispatch, "rollout_locked") as apply,
        ):
            self.assertEqual(
                dispatch.poll_once(self.state, self.state, self.backend), 20
            )
        apply.assert_called_once_with(self.target, self.state, self.backend)
        self.assertEqual(
            json.loads((self.state / "dispatcher.json").read_text())["processed"],
            ["90:1", "20:1"],
        )

    def test_late_lower_number_completion_is_not_filtered_by_cursor(self):
        rollout.atomic_json(
            self.state / "dispatcher.json", {"processed": ["90:1"], "run_id": 90}
        )
        with (
            patch.object(
                dispatch, "successful_runs", return_value=[run(9, "c"), run(2, "b")]
            ),
            patch.object(dispatch, "download_release", return_value=self.target),
            patch.object(dispatch.subprocess, "run"),
            patch.object(dispatch, "ancestor", return_value=False),
            patch.object(dispatch, "rollout_locked") as apply,
        ):
            self.assertEqual(
                dispatch.poll_once(self.state, self.state, self.backend), 20
            )
        apply.assert_called_once()

    def test_deployment_failure_exits_watch_without_automatic_redeploy(self):
        with (
            patch.object(dispatch, "Compose", return_value=self.backend),
            patch.object(
                dispatch,
                "poll_once",
                side_effect=RuntimeError("health rollback completed"),
            ) as poll,
            patch.object(dispatch.logger, "exception") as log,
            patch.object(dispatch.time, "sleep") as sleep,
            self.assertRaises(RuntimeError),
        ):
            dispatch.watch(self.state, self.state)
        poll.assert_called_once()
        sleep.assert_not_called()
        log.assert_called_once()
