"""The chosen release keeps its proof across both sides of admission reopening."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from bin import rollout
from bin.tests.test_rollout import manifest


class OpeningTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.state = Path(self.directory.name)
        self.current, self.target = manifest(), manifest("b", "2")
        rollout.atomic_json(
            self.state / "state.json",
            {"good": self.current.model_dump(), "pending": None},
        )
        rollout.fence(self.state, [])
        self.backend = Mock(spec=rollout.Compose)
        self.backend.snapshot_epochs.return_value = {"recorded": "epochs"}
        self.backend.drain.side_effect = lambda *args: self.actual_fence(
            self.state, list(rollout.OWNERS)
        )
        self.actual_fence = rollout.fence

    def record(self):
        return json.loads((self.state / "state.json").read_text())

    def interrupt_before_open(self):
        def fence(state, closed):
            if not closed:
                raise KeyboardInterrupt
            self.actual_fence(state, closed)

        with (
            patch.object(rollout, "fence", side_effect=fence),
            self.assertRaises(KeyboardInterrupt),
        ):
            rollout.rollout(self.target, self.state, self.backend)
        return self.record()["pending"]

    def test_crash_before_open_preserves_resolved_release_and_epoch_receipt(self):
        pending = self.interrupt_before_open()
        self.assertEqual(pending["phase"], "opening")
        self.assertEqual(pending["opening"]["release"], self.target.model_dump())
        self.assertEqual(pending["opening"]["baseline"], {"recorded": "epochs"})
        self.assertEqual(self.record()["good"], self.target.model_dump())
        self.assertEqual(
            set(json.loads((self.state / "gate/fence.json").read_text())["closed"]),
            set(rollout.OWNERS),
        )

    def test_closed_gate_resume_revalidates_before_and_after_health_and_runner(self):
        journal = self.interrupt_before_open()
        journal["runner"] = {"receipt": "retained"}
        self.backend.reset_mock()
        rollout.recover(self.state, self.backend, journal)
        calls = [call[0] for call in self.backend.mock_calls]
        self.assertEqual(
            calls,
            [
                "unchanged_epochs",
                "verify",
                "verify_runner",
                "unchanged_epochs",
                "verify_open",
            ],
        )
        self.backend.verify_runner.assert_called_once_with(journal, self.target)
        self.backend.replace.assert_not_called()
        self.backend.drain.assert_not_called()
        self.assertIsNone(self.record()["pending"])

    def test_unknown_restarted_epoch_keeps_opening_receipt_and_fence(self):
        journal = self.interrupt_before_open()
        self.backend.unchanged_epochs.side_effect = RuntimeError("unknown restart")
        with self.assertRaises(RuntimeError):
            rollout.recover(self.state, self.backend, journal)
        self.assertEqual(self.record()["pending"]["phase"], "opening")
        self.backend.replace.assert_called_once()  # Original target only; no rollback.
        self.assertTrue(
            json.loads((self.state / "gate/fence.json").read_text())["closed"]
        )

    def test_changed_lease_on_resume_cannot_bypass_eligibility(self):
        journal = self.interrupt_before_open()
        journal["runner"] = {"receipt": "retained"}
        self.backend.verify_runner.side_effect = ValueError("foreign lease owner")
        with self.assertRaises(ValueError):
            rollout.recover(self.state, self.backend, journal)
        self.assertEqual(self.record()["pending"]["phase"], "opening")
        self.assertTrue(
            json.loads((self.state / "gate/fence.json").read_text())["closed"]
        )

    def test_crash_after_open_only_finalizes_without_touching_admitted_work(self):
        journal = self.interrupt_before_open()
        self.actual_fence(self.state, [])
        self.backend.reset_mock()
        self.backend.verify.side_effect = RuntimeError("new admitted work is active")
        rollout.recover(self.state, self.backend, journal)
        self.assertEqual([call[0] for call in self.backend.mock_calls], ["verify_open"])
        self.assertIsNone(self.record()["pending"])
        self.assertEqual(self.record()["good"], self.target.model_dump())

    def test_process_restart_during_final_health_cannot_become_opening_baseline(self):
        self.backend.unchanged_epochs.side_effect = [
            None,
            RuntimeError("restart after final health"),
            RuntimeError("restart remains unproven during recovery"),
        ]
        self.backend.safe_reopen.side_effect = RuntimeError("original baseline gone")
        with self.assertRaises(RuntimeError):
            rollout.rollout(self.target, self.state, self.backend)
        self.assertTrue(
            json.loads((self.state / "gate/fence.json").read_text())["closed"]
        )
        self.assertIsNotNone(self.record()["pending"])
