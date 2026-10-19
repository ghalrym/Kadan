"""Natural runner lease handoff with server-clock observations and no real waits."""

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import UUID

from bin import rollout
from bin.tests.test_rollout import manifest

OLD, NEW = UUID(int=1), UUID(int=2)
NOW = datetime(2026, 10, 19, tzinfo=UTC)


def row(owner=OLD, expiry=None, capacity=5, assigned=2):
    return rollout.RunnerLease(
        runner_id="runner-1",
        worker_id=owner,
        lease_expires_at=expiry or NOW + timedelta(seconds=60),
        capacity=capacity,
        assigned_agent_count=assigned,
    )


def identity(owner=OLD, capacity=5):
    return {
        "fenced": True,
        "active": 0,
        "ready": True,
        "runner": {
            "runner_id": "runner-1",
            "worker_id": str(owner),
            "capacity": capacity,
        },
    }


class RunnerHandoffTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.state = Path(self.directory.name)
        rollout.fence(self.state, list(rollout.OWNERS))
        self.current = manifest("a", "1", "kadan-api")
        self.target = manifest("b", "2", "kadan-api")
        self.old = rollout.ContainerIdentity(
            id="old-runner",
            started="old-time",
            image=self.current.images["kadan-runner"],
            hostname="old-host",
            generation=None,
        )
        self.new = rollout.ContainerIdentity(
            id="new-runner",
            started="new-time",
            image=self.target.images["kadan-runner"],
            hostname="new-host",
            generation="transaction",
        )
        self.runner = {
            "runner_id": "runner-1",
            "capacity": 5,
            "assigned_agent_count": 2,
            "old": self.old.model_dump(),
            "leases": {str(OLD): row().lease_expires_at.isoformat()},
            "generations": [],
        }
        self.journal = {
            "previous": self.current.model_dump(),
            "target": self.target.model_dump(),
            "affected": ["kadan-api", "kadan-runner"],
            "phase": "replacing",
            "runner": self.runner,
            "native": None,
        }
        self.backend = object.__new__(rollout.Compose)
        self.backend.state, self.backend.timeout = self.state, 300
        self.backend.containers = Mock(return_value={"kadan-runner": self.old})
        raw = {"kadan-runner": {"Id": self.old.id, "State": {"Running": True}}}
        stopped = {"kadan-runner": {"Id": self.old.id, "State": {"Running": False}}}
        self.backend.persisted_old = Mock(side_effect=[raw, stopped])
        self.backend.status = Mock(return_value=identity())
        self.backend.stopped = Mock(return_value=False)
        self.backend.unaffected_running = Mock()
        self.backend.stop_ids = Mock()
        self.backend.lease_snapshot = Mock(
            side_effect=[(NOW, row()), (NOW + timedelta(seconds=61), row())]
        )

    def test_idle_runner_stops_only_exact_id_then_waits_natural_expiry(self):
        with patch.object(rollout.time, "sleep") as sleep:
            self.backend.handoff_runner(self.journal)
        self.backend.stop_ids.assert_called_once_with((self.old.id,))
        sleep.assert_called_once_with(0.2)
        self.assertEqual(self.backend.lease_snapshot.call_count, 2)
        self.assertEqual(
            self.journal["runner"]["leases"][str(OLD)],
            row().lease_expires_at.isoformat(),
        )

    def test_stop_intent_and_original_lease_are_durable_before_stop(self):
        def stop(ids):
            pending = json.loads((self.state / "state.json").read_text())["pending"]
            self.assertEqual(pending["runner"]["old"]["id"], self.old.id)
            self.assertEqual(
                pending["runner"]["leases"][str(OLD)],
                row().lease_expires_at.isoformat(),
            )

        self.backend.stop_ids.side_effect = stop
        with patch.object(rollout.time, "sleep"):
            self.backend.handoff_runner(self.journal)

    def test_renewed_expiry_is_not_silently_waited_out(self):
        self.backend.lease_snapshot.side_effect = [
            (NOW, row(expiry=NOW + timedelta(seconds=90)))
        ]
        with self.assertRaisesRegex(ValueError, "expiry changed"):
            self.backend.handoff_runner(self.journal)
        self.assertFalse((self.state / "gate/runner.json").exists())

    def test_foreign_owner_change_fails_closed(self):
        self.backend.lease_snapshot.side_effect = [(NOW, row(owner=UUID(int=3)))]
        with self.assertRaisesRegex(ValueError, "owner or expiry"):
            self.backend.handoff_runner(self.journal)

    def test_same_container_id_with_restarted_epoch_is_never_stopped(self):
        self.backend.containers.return_value = {
            "kadan-runner": self.old.model_copy(
                update={"started": "unexpected-restart"}
            )
        }
        with self.assertRaisesRegex(ValueError, "Unknown runner"):
            self.backend.handoff_runner(self.journal)
        self.backend.stop_ids.assert_not_called()

    def test_supervisor_stopping_unaffected_stack_preserves_pending_and_fence(self):
        self.backend.unaffected_running.side_effect = RuntimeError(
            "stack supervisor stopped Hikari"
        )
        with self.assertRaises(RuntimeError):
            self.backend.handoff_runner(self.journal)
        self.assertIsNotNone(
            json.loads((self.state / "state.json").read_text())["pending"]
        )
        self.assertEqual(
            set(json.loads((self.state / "gate/fence.json").read_text())["closed"]),
            set(rollout.OWNERS),
        )
        self.backend.lease_snapshot.assert_not_called()

    def test_unavailable_lease_observation_never_authorizes_start(self):
        self.backend.lease_snapshot.side_effect = ValueError("malformed server clock")
        with self.assertRaises(ValueError):
            self.backend.handoff_runner(self.journal)
        self.assertFalse((self.state / "gate/runner.json").exists())

    def test_active_runner_is_never_stopped(self):
        self.backend.status.return_value = {**identity(), "active": 1}
        with self.assertRaisesRegex(ValueError, "no longer drained"):
            self.backend.handoff_runner(self.journal)
        self.backend.stop_ids.assert_not_called()

    def test_rollback_waits_for_legitimate_target_lease_too(self):
        self.runner["generations"] = ["transaction"]
        self.backend.containers.return_value = {"kadan-runner": self.new}
        self.backend.persisted_old.side_effect = [{}, {}]
        self.backend.status.return_value = identity(NEW)
        target = row(owner=NEW)
        self.backend.lease_snapshot.side_effect = [
            (NOW, target),
            (NOW, target),
            (NOW + timedelta(seconds=61), target),
        ]
        with patch.object(rollout.time, "sleep") as sleep:
            self.backend.handoff_runner(self.journal)
        self.backend.stop_ids.assert_called_once_with((self.new.id,))
        self.assertIn(str(NEW), self.runner["leases"])
        sleep.assert_called_once()

    def prepare_verify(self, lease=None, capacity=5):
        self.runner["generations"] = ["transaction"]
        self.backend.containers.return_value = {"kadan-runner": self.new}
        value = lease or row()
        snapshot = {
            "observed_at": (NOW + timedelta(seconds=61)).isoformat(),
            "runners": [value.model_dump(mode="json")],
        }
        self.backend.status.side_effect = [
            identity(NEW, capacity),
            {"runner_leases": snapshot},
        ]
        return snapshot

    def test_fresh_runner_is_eligible_after_expiry_without_claiming_registration(self):
        self.prepare_verify()
        self.backend.verify_runner(self.journal, self.target)

    def test_unexpired_old_lease_prevents_reopen(self):
        self.prepare_verify(row(expiry=NOW + timedelta(seconds=120)))
        with self.assertRaises(ValueError):
            self.backend.verify_runner(self.journal, self.target)

    def test_changed_configured_capacity_prevents_reopen(self):
        self.prepare_verify(capacity=1)
        with self.assertRaises(ValueError):
            self.backend.verify_runner(self.journal, self.target)

    def test_fresh_uuid_owned_by_other_runner_prevents_reopen(self):
        snapshot = self.prepare_verify()
        snapshot["runners"].append(
            {**row(owner=NEW).model_dump(mode="json"), "runner_id": "runner-other"}
        )
        with self.assertRaisesRegex(ValueError, "already owns"):
            self.backend.verify_runner(self.journal, self.target)

    def test_pre_mutation_supervisor_failure_never_reopens(self):
        backend = Mock(spec=rollout.Compose)
        backend.safe_reopen.side_effect = RuntimeError("old stack stopped")
        self.journal["phase"] = "draining"
        rollout.save_pending(self.state, self.journal)
        with self.assertRaises(RuntimeError):
            rollout.recover(self.state, backend, self.journal)
        self.assertIsNotNone(
            json.loads((self.state / "state.json").read_text())["pending"]
        )
        self.assertEqual(
            set(json.loads((self.state / "gate/fence.json").read_text())["closed"]),
            set(rollout.OWNERS),
        )
        backend.replace.assert_not_called()

    def test_supervision_enrollment_is_explicitly_required(self):
        rollout.atomic_json(
            self.state / "installation.json",
            {"project": "kadan", "env_sha256": "a" * 64},
        )
        with self.assertRaisesRegex(ValueError, "detached-independent"):
            rollout.Compose(self.state, self.state)

    def test_unaffected_same_id_restarted_native_epoch_is_rejected(self):
        baseline = {
            name: self.old.model_copy(update={"id": name}).model_dump()
            for name in rollout.SERVICES
        }
        names = [
            name for name in rollout.SERVICES if name not in self.journal["affected"]
        ]
        cohort = {name: rollout.ContainerIdentity(**baseline[name]) for name in names}
        cohort["hikari-worker-llm"] = cohort["hikari-worker-llm"].model_copy(
            update={"started": "supervisor-restarted"}
        )
        self.backend.containers.return_value = cohort
        with self.assertRaisesRegex(RuntimeError, "epoch changed"):
            self.backend.unchanged_epochs(baseline, tuple(self.journal["affected"]))

    def test_pre_mutation_same_image_api_restart_preserves_journal(self):
        baseline = {
            name: self.old.model_copy(update={"id": name}).model_dump()
            for name in rollout.SERVICES
        }
        self.journal["baseline"] = baseline
        self.journal["phase"] = "draining"
        cohort = {
            name: rollout.ContainerIdentity(**value) for name, value in baseline.items()
        }
        cohort["kadan-api"] = cohort["kadan-api"].model_copy(
            update={"started": "supervisor-restarted"}
        )
        self.backend.containers.return_value = cohort
        rollout.save_pending(self.state, self.journal)
        with self.assertRaisesRegex(RuntimeError, "epoch changed"):
            rollout.recover(self.state, self.backend, self.journal)
        self.assertIsNotNone(
            json.loads((self.state / "state.json").read_text())["pending"]
        )
        self.assertEqual(
            set(json.loads((self.state / "gate/fence.json").read_text())["closed"]),
            set(rollout.OWNERS),
        )

    def test_stopped_unaffected_service_is_rejected_even_with_same_epoch(self):
        baseline = {
            name: self.old.model_copy(update={"id": name}).model_dump()
            for name in rollout.SERVICES
        }
        names = [
            name for name in rollout.SERVICES if name not in self.journal["affected"]
        ]
        self.backend.containers.return_value = {
            name: rollout.ContainerIdentity(**baseline[name]) for name in names
        }
        observed = [
            {"Id": name, "State": {"Running": name != "hikari-api"}} for name in names
        ]
        with (
            patch.object(
                rollout.subprocess,
                "run",
                return_value=Mock(stdout=json.dumps(observed)),
            ),
            self.assertRaisesRegex(RuntimeError, "stopped unexpectedly"),
        ):
            self.backend.unchanged_epochs(baseline, tuple(self.journal["affected"]))

    def test_changed_assignment_count_prevents_new_runner_eligibility(self):
        self.prepare_verify(row(assigned=3))
        with self.assertRaisesRegex(ValueError, "capacity is incompatible"):
            self.backend.verify_runner(self.journal, self.target)

    def test_affected_restart_during_drain_prevents_any_replacement(self):
        backend = Mock(spec=rollout.Compose)
        backend.snapshot_epochs.return_value = {}
        backend.drain.side_effect = lambda: rollout.fence(
            self.state, list(rollout.OWNERS)
        )
        backend.unchanged_epochs.side_effect = RuntimeError(
            "affected API restarted during drain"
        )
        backend.safe_reopen.side_effect = RuntimeError(
            "original baseline no longer exists"
        )
        rollout.atomic_json(
            self.state / "state.json",
            {"good": self.current.model_dump(), "pending": None},
        )
        rollout.fence(self.state, [])
        with self.assertRaises(RuntimeError):
            rollout.rollout(self.target, self.state, backend)
        backend.runner_transition.assert_not_called()
        backend.native_transition.assert_not_called()
        backend.replace.assert_not_called()
        backend.handoff_runner.assert_not_called()
        self.assertEqual(
            json.loads((self.state / "state.json").read_text())["pending"]["phase"],
            "draining",
        )
        self.assertEqual(
            set(json.loads((self.state / "gate/fence.json").read_text())["closed"]),
            set(rollout.OWNERS),
        )
