"""Proven-idle cohort replacement and recovery without native execution or Docker."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import UUID

from bin import rollout
from bin.tests.test_rollout import manifest

TOKEN = UUID(int=999)


def proof(fresh=False):
    return rollout.NativeProof(
        native_idle=True,
        api_generation=UUID(int=200 if fresh else 100),
        cold_start_generation=str(TOKEN) if fresh else None,
        workers=tuple(
            rollout.NativeWorker(
                kind=name.removeprefix("hikari-worker-"),
                instance=("new-" if fresh else "old-") + name,
                generation=UUID(int=index + (101 if fresh else 1)),
            )
            for index, name in enumerate(rollout.WORKERS)
        ),
    )


def containers(release, fresh=False):
    return {
        name: rollout.ContainerIdentity(
            id=("new-" if fresh else "old-") + name,
            started="new-time" if fresh else "old-time",
            image=release.images[name],
            hostname=("new-" if fresh else "old-") + name,
            generation=str(TOKEN) if fresh else None,
        )
        for name in rollout.NATIVE
    }


class NativeRolloutTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.state = Path(self.directory.name)
        rollout.fence(self.state, list(rollout.OWNERS))
        self.current = manifest("a", "1", "hikari-api")
        self.target = manifest("b", "2", "hikari-api")
        self.old = containers(self.current)
        self.new = containers(self.target, True)
        self.backend = object.__new__(rollout.Compose)
        self.backend.state = self.state
        self.backend.timeout = 0
        self.backend.command = Mock(return_value="")
        self.backend.overlay = Mock()
        self.backend.stopped = Mock(return_value=False)
        self.backend.stop_native_ids = Mock()
        self.native = {
            "proof": proof().model_dump(mode="json"),
            "old": {name: value.model_dump() for name, value in self.old.items()},
            "generations": [],
            "fresh_proof": None,
            "stop_authorized": False,
        }
        self.journal = {
            "previous": self.current.model_dump(),
            "target": self.target.model_dump(),
            "phase": "native_stopping",
            "affected": rollout.NATIVE,
            "native": self.native,
        }
        self.backend.containers = Mock(side_effect=[self.old, self.old, self.new])
        raw = {
            name: {"Id": value.id, "State": {"Running": True}}
            for name, value in self.old.items()
        }
        stopped = {
            name: {**value, "State": {"Running": False}} for name, value in raw.items()
        }
        self.backend.persisted_old = Mock(side_effect=[raw, stopped])
        self.backend.barrier = Mock(side_effect=[proof(), proof(True)])

    def test_old_idle_proof_is_revalidated_before_any_stop(self):
        self.backend.barrier.side_effect = ValueError("connection lost")
        with self.assertRaises(ValueError):
            self.backend.replace_native(self.target, self.journal)
        self.backend.stop_native_ids.assert_not_called()
        self.backend.overlay.assert_not_called()

    def test_workers_stop_before_api_and_new_cohort_has_fresh_proof(self):
        with patch.object(rollout, "uuid4", return_value=TOKEN):
            self.backend.replace_native(self.target, self.journal)
        self.assertEqual(
            self.backend.stop_native_ids.call_args_list[0].args,
            (tuple(self.old[name].id for name in rollout.WORKERS),),
        )
        self.assertEqual(
            self.backend.stop_native_ids.call_args_list[1].args,
            ((self.old["hikari-api"].id,),),
        )
        self.assertFalse(
            any(call.args[0] == "pull" for call in self.backend.command.call_args_list)
        )
        self.assertEqual(self.journal["phase"], "replacing")
        self.assertEqual(
            self.native["fresh_proof"], proof(True).model_dump(mode="json")
        )
        self.assertEqual(
            json.loads((self.state / "gate/native.json").read_text())["generation"],
            str(TOKEN),
        )

    def test_stop_authorization_is_durable_before_first_stop(self):
        def stopped(ids):
            record = json.loads((self.state / "state.json").read_text())
            self.assertTrue(record["pending"]["native"]["stop_authorized"])
            self.assertEqual(
                record["pending"]["native"]["proof"], proof().model_dump(mode="json")
            )
            self.assertEqual(
                set(record["pending"]["native"]["old"]), set(rollout.NATIVE)
            )

        self.backend.stop_native_ids.side_effect = stopped
        with patch.object(rollout, "uuid4", return_value=TOKEN):
            self.backend.replace_native(self.target, self.journal)

    def test_cold_marker_is_not_written_if_worker_stop_unconfirmed(self):
        self.backend.stop_native_ids.side_effect = RuntimeError("still running")
        with self.assertRaises(RuntimeError):
            self.backend.replace_native(self.target, self.journal)
        self.assertFalse((self.state / "gate/native.json").exists())
        self.backend.overlay.assert_not_called()

    def test_hidden_running_old_container_cannot_be_skipped(self):
        self.backend.containers.side_effect = [
            {
                name: value
                for name, value in self.old.items()
                if name != rollout.WORKERS[0]
            }
        ]
        with self.assertRaisesRegex(ValueError, "hidden"):
            self.backend.replace_native(self.target, self.journal)
        self.backend.stop_native_ids.assert_not_called()

    def test_unknown_restarted_container_epoch_is_never_stopped(self):
        self.backend.persisted_old.side_effect = None
        self.backend.persisted_old.return_value = {}
        unknown = {
            **self.old,
            rollout.WORKERS[0]: self.old[rollout.WORKERS[0]].model_copy(
                update={"started": "unknown"}
            ),
        }
        self.backend.containers.side_effect = [unknown]
        with self.assertRaisesRegex(ValueError, "Unknown native"):
            self.backend.replace_native(self.target, self.journal)
        self.backend.stop_native_ids.assert_not_called()

    def test_old_native_generation_in_cold_cohort_is_refused(self):
        self.backend.barrier.side_effect = [
            proof(),
            proof(True).model_copy(update={"workers": proof().workers}),
        ]
        with (
            patch.object(rollout, "uuid4", return_value=TOKEN),
            self.assertRaisesRegex(ValueError, "previous native"),
        ):
            self.backend.replace_native(self.target, self.journal)
        self.assertIsNone(self.native["fresh_proof"])

    def test_wrong_cold_marker_generation_is_refused(self):
        self.backend.barrier.side_effect = [
            proof(),
            proof(True).model_copy(update={"cold_start_generation": "wrong"}),
        ]
        with (
            patch.object(rollout, "uuid4", return_value=TOKEN),
            self.assertRaisesRegex(ValueError, "controller generation"),
        ):
            self.backend.replace_native(self.target, self.journal)

    def test_native_transition_rejects_identity_change_during_barrier(self):
        self.backend.containers.side_effect = [self.old, self.new]
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.backend.native_transition(self.current)

    def test_api_change_restarts_unchanged_workers_without_changing_their_pins(self):
        affected = rollout.affected_services(self.current, self.target)
        self.assertEqual(set(affected), set(rollout.NATIVE))
        self.assertTrue(
            all(
                self.current.images[name] == self.target.images[name]
                for name in rollout.WORKERS
            )
        )

    def test_invalid_native_idle_receipt_is_rejected(self):
        value = proof().model_dump(mode="json")
        value["native_idle"] = False
        with self.assertRaises(ValueError):
            rollout.NativeProof(**value)

    def test_unknown_disconnect_during_first_barrier_never_stops_anything(self):
        backend = Mock(spec=rollout.Compose)
        backend.drain.side_effect = lambda: rollout.fence(
            self.state, list(rollout.OWNERS)
        )
        backend.native_transition.side_effect = ValueError(
            "uncertain native connection"
        )
        rollout.atomic_json(
            self.state / "state.json",
            {"good": self.current.model_dump(), "pending": None},
        )
        rollout.fence(self.state, [])
        with self.assertRaises(ValueError):
            rollout.rollout(self.target, self.state, backend)
        backend.replace_native.assert_not_called()
        backend.replace.assert_not_called()
        self.assertEqual(
            json.loads((self.state / "state.json").read_text())["good"],
            self.current.model_dump(),
        )

    def test_interrupted_native_stop_keeps_proof_for_restart_recovery(self):
        backend = Mock(spec=rollout.Compose)
        backend.drain.side_effect = lambda: rollout.fence(
            self.state, list(rollout.OWNERS)
        )
        backend.native_transition.return_value = self.native
        backend.replace_native.side_effect = KeyboardInterrupt
        rollout.atomic_json(
            self.state / "state.json",
            {"good": self.current.model_dump(), "pending": None},
        )
        rollout.fence(self.state, [])
        with self.assertRaises(KeyboardInterrupt):
            rollout.rollout(self.target, self.state, backend)
        pending = json.loads((self.state / "state.json").read_text())["pending"]
        self.assertEqual(pending["phase"], "native_stopping")
        self.assertEqual(pending["native"]["proof"], self.native["proof"])
        self.assertEqual(
            set(json.loads((self.state / "gate/fence.json").read_text())["closed"]),
            set(rollout.OWNERS),
        )

    def test_failed_fresh_health_restores_complete_previous_cohort(self):
        backend = Mock(spec=rollout.Compose)
        backend.native_transition.return_value = self.native
        backend.verify.side_effect = [RuntimeError("bad smoke"), None]
        rollout.atomic_json(
            self.state / "state.json",
            {"good": self.current.model_dump(), "pending": None},
        )
        rollout.fence(self.state, [])
        with self.assertRaises(RuntimeError):
            rollout.rollout(self.target, self.state, backend)
        self.assertEqual(backend.replace_native.call_args_list[0].args[0], self.target)
        self.assertEqual(backend.replace_native.call_args_list[1].args[0], self.current)
        self.assertEqual(
            json.loads((self.state / "state.json").read_text())["good"],
            self.current.model_dump(),
        )
