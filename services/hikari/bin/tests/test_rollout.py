"""Exercise release ordering and recovery without Docker, a network, or inference."""

import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from bin import release, rollout


def manifest(source="a", image="1", service="kadan-frontend"):
    images = {
        name: "ghcr.io/ghalrym/kadan/runtime@sha256:" + "0" * 64
        for name in rollout.SERVICES
    }
    images[service] = "ghcr.io/ghalrym/kadan/runtime@sha256:" + image * 64
    if service in {"kadan-api", "kadan-runner"}:
        images["kadan-api"] = images[service]
        images["kadan-runner"] = images[service]
    return rollout.Release(
        source=source * 40, compose_sha256="f" * 64, compatibility="1", images=images
    )


class RolloutTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.state = Path(self.directory.name)
        self.current = manifest()
        self.target = manifest("b", "2")
        rollout.atomic_json(
            self.state / "state.json",
            {"good": self.current.model_dump(), "pending": None},
        )
        rollout.fence(self.state, [])
        self.backend = Mock(spec=rollout.Compose)
        self.backend.snapshot_epochs.return_value = {}

    def record(self):
        return json.loads((self.state / "state.json").read_text())

    def test_fence_remains_readable_after_replacement_under_private_umask(self):
        previous = os.umask(0o077)
        try:
            for closed in (list(rollout.OWNERS), [], list(rollout.OWNERS), []):
                rollout.fence(self.state, closed)
                gate = self.state / "gate"
                self.assertEqual(stat.S_IMODE(gate.stat().st_mode), 0o755)
                self.assertEqual(
                    stat.S_IMODE((gate / "fence.json").stat().st_mode), 0o644
                )
                self.assertEqual(
                    json.loads((gate / "fence.json").read_text())["closed"], closed
                )
            rollout.atomic_json(self.state / "private.json", {"private": True})
            self.assertEqual(
                stat.S_IMODE((self.state / "private.json").stat().st_mode), 0o600
            )
        finally:
            os.umask(previous)

    def test_unreadable_open_gate_preserves_receipt_until_owners_observe_open(self):
        journal = {
            "previous": self.current.model_dump(),
            "target": self.target.model_dump(),
            "phase": "replacing",
            "affected": ["kadan-frontend"],
            "native": None,
            "runner": None,
            "baseline": {},
        }
        rollout.fence(self.state, list(rollout.OWNERS))
        self.backend.verify_open.side_effect = RuntimeError("still fenced")
        with self.assertRaisesRegex(RuntimeError, "still fenced"):
            rollout.complete_opening(self.state, self.backend, journal, self.target, {})
        self.assertEqual(self.record()["pending"]["phase"], "opening")
        self.assertEqual(
            json.loads((self.state / "gate/fence.json").read_text())["closed"], []
        )
        self.backend.reset_mock()
        self.backend.verify_open.side_effect = None
        self.assertEqual(rollout.resume_locked(self.state, self.backend), self.target)
        self.assertIsNone(self.record()["pending"])
        self.backend.verify.assert_not_called()
        self.backend.replace.assert_not_called()

    def test_open_verification_checks_each_owner_without_requiring_idle(self):
        backend = object.__new__(rollout.Compose)
        backend.status = Mock(return_value={"fenced": False, "active": 3})
        backend.verify_open()
        self.assertEqual(backend.status.call_count, len(rollout.OWNERS))
        for owner in rollout.OWNERS:
            backend.status = Mock(
                side_effect=lambda name, owner=owner: {"fenced": name == owner}
            )
            with self.assertRaisesRegex(RuntimeError, "preserve opening receipt"):
                backend.verify_open()

    def test_only_changed_service_is_replaced(self):
        rollout.rollout(self.target, self.state, self.backend)
        self.backend.replace.assert_called_once_with(self.target, ("kadan-frontend",))
        self.assertEqual(self.record()["good"], self.target.model_dump())
        self.assertEqual(
            json.loads((self.state / "gate/fence.json").read_text()), {"closed": []}
        )

    def test_failed_drain_never_replaces_containers(self):
        self.backend.drain.side_effect = TimeoutError("active jobs")
        with self.assertRaises(TimeoutError):
            rollout.rollout(self.target, self.state, self.backend)
        self.backend.replace.assert_not_called()
        self.assertIsNone(self.record()["pending"])
        self.assertEqual(self.record()["good"], self.current.model_dump())

    def test_failed_health_restores_previous_pins(self):
        self.backend.verify.side_effect = [RuntimeError("unhealthy"), None]
        with self.assertRaises(RuntimeError):
            rollout.rollout(self.target, self.state, self.backend)
        self.assertEqual(
            self.backend.replace.call_args_list[-1].args,
            (self.current, ("kadan-frontend",)),
        )
        self.assertEqual(self.record()["good"], self.current.model_dump())
        self.assertIsNone(self.record()["pending"])

    def test_failed_rollback_keeps_journal_and_fence(self):
        self.backend.verify.side_effect = RuntimeError("unhealthy")
        self.backend.drain.side_effect = lambda *args: rollout.fence(
            self.state, list(rollout.OWNERS)
        )
        with self.assertRaises(RuntimeError):
            rollout.rollout(self.target, self.state, self.backend)
        self.assertEqual(self.record()["pending"]["phase"], "replacing")
        self.assertTrue(
            json.loads((self.state / "gate/fence.json").read_text())["closed"]
        )

    def test_interrupted_replacement_is_recovered_before_next_release(self):
        pending = {
            "previous": self.current.model_dump(),
            "target": self.target.model_dump(),
            "affected": ["kadan-frontend"],
            "phase": "replacing",
        }
        rollout.atomic_json(
            self.state / "state.json",
            {"good": self.current.model_dump(), "pending": pending},
        )
        rollout.rollout(self.target, self.state, self.backend)
        self.assertEqual(self.backend.replace.call_args_list[0].args[0], self.current)
        self.assertEqual(self.backend.replace.call_args_list[1].args[0], self.target)

    def test_closed_gate_without_opening_receipt_fails_closed(self):
        rollout.fence(self.state, list(rollout.OWNERS))
        with self.assertRaisesRegex(ValueError, "no durable opening receipt"):
            rollout.rollout(self.current, self.state, self.backend)
        self.backend.drain.assert_not_called()
        self.backend.replace.assert_not_called()
        self.assertTrue(
            json.loads((self.state / "gate/fence.json").read_text())["closed"]
        )

    def test_stale_release_refusal_leaves_state_and_containers_unchanged(self):
        self.backend.preflight.side_effect = ValueError("not descendant")
        with self.assertRaises(ValueError):
            rollout.rollout(self.target, self.state, self.backend)
        self.backend.drain.assert_not_called()
        self.backend.replace.assert_not_called()
        self.assertEqual(self.record()["good"], self.current.model_dump())

    def test_digest_unchanged_only_advances_commit(self):
        target = self.current.model_copy(update={"source": "b" * 40})
        rollout.rollout(target, self.state, self.backend)
        self.backend.replace.assert_not_called()
        self.assertEqual(self.record()["good"]["source"], target.source)


class ValidationTests(unittest.TestCase):
    def test_mutable_tags_are_refused(self):
        value = manifest().model_dump()
        value["images"]["kadan-frontend"] = "ghcr.io/ghalrym/kadan/web:latest"
        with self.assertRaises(ValueError):
            rollout.Release(**value)

    def test_other_registry_is_refused(self):
        value = manifest().model_dump()
        value["images"]["kadan-frontend"] = "evil.example/web@sha256:" + "1" * 64
        with self.assertRaises(ValueError):
            rollout.Release(**value)

    def test_native_change_expands_only_coupled_inference_group(self):
        self.assertEqual(
            set(
                rollout.affected_services(
                    manifest("a", "1", "hikari-api"), manifest("b", "2", "hikari-api")
                )
            ),
            set(rollout.NATIVE),
        )

    def test_republished_commit_is_refused(self):
        backend = object.__new__(rollout.Compose)
        with self.assertRaisesRegex(ValueError, "republished"):
            backend.preflight(manifest(), manifest("a", "2"))

    def test_ci_requires_all_exact_master_workflows(self):
        with (
            patch.object(release, "github", return_value={"workflow_runs": []}),
            self.assertRaises(ValueError),
        ):
            release.require_ci("a" * 40)

    def test_shared_tracing_rebuilds_all_python_owners(self):
        self.assertEqual(
            release.affected_services(["services/tracing/kadan_tracing/__init__.py"]),
            {"hikari-api", "kadan-api", "kadan-runner", "tsuchi-api"},
        )

    def test_frontend_change_does_not_rebuild_api_or_native(self):
        self.assertEqual(
            release.affected_services(["services/kadan/web/src/main.ts"]),
            {"kadan-frontend"},
        )

    def test_download_rejects_untrusted_workflow_before_artifact_read(self):
        value = {
            "path": ".github/workflows/evil.yml",
            "head_branch": "master",
            "event": "push",
            "status": "completed",
            "conclusion": "success",
            "repository": {"full_name": "ghalrym/Kadan"},
        }
        with (
            patch.object(
                rollout.subprocess, "run", return_value=Mock(stdout=json.dumps(value))
            ) as run,
            self.assertRaises(ValueError),
        ):
            rollout.download_release(123, Path("/tmp/unused"))
        self.assertEqual(run.call_count, 1)


class ComposeTests(unittest.TestCase):
    def setUp(self):
        self.backend = object.__new__(rollout.Compose)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.backend.state = Path(self.directory.name)
        rollout.fence(self.backend.state, [])
        self.backend.timeout = 0

    def test_drain_closes_upstream_first_and_waits_before_downstream(self):
        self.backend.status = Mock(return_value={"fenced": True, "active": 0})
        with patch.object(rollout, "fence") as fence:
            self.backend.drain()
        self.assertEqual(
            [call.args[1][:] for call in fence.call_args_list],
            [list(rollout.OWNERS[:2]), list(rollout.OWNERS[:3]), list(rollout.OWNERS)],
        )

    def test_active_upstream_prevents_downstream_fence(self):
        self.backend.status = Mock(return_value={"fenced": True, "active": 1})
        with patch.object(rollout, "fence") as fence, self.assertRaises(TimeoutError):
            self.backend.drain()
        self.assertEqual(fence.call_count, 1)

    def test_ndjson_health_is_parsed(self):
        self.backend.command = Mock(
            return_value=json.dumps(
                {"Service": "kadan-frontend", "State": "running", "Health": "healthy"}
            )
        )
        self.backend.status = Mock(
            return_value={"fenced": True, "active": 0, "ready": True}
        )
        self.backend.verify(("kadan-frontend",))

    def test_missing_owner_status_never_passes_readiness(self):
        self.backend.command = Mock(return_value="[]")
        self.backend.status = Mock(side_effect=RuntimeError("unavailable"))
        with self.assertRaises(RuntimeError):
            self.backend.verify(("kadan-frontend",))


class PublicationTests(unittest.TestCase):
    def test_all_exact_successful_runs_admit_publication(self):
        runs = [
            {
                "path": path,
                "head_sha": "a" * 40,
                "head_branch": "master",
                "event": "push",
                "repository": {"full_name": "ghalrym/Kadan"},
                "run_number": 1,
                "run_attempt": 1,
                "status": "completed",
                "conclusion": "success",
            }
            for path in release.WORKFLOWS
        ]
        with patch.object(release, "github", return_value={"workflow_runs": runs}):
            release.require_ci("a" * 40)

    def test_failed_latest_attempt_cannot_reuse_older_success(self):
        runs = [
            {
                "path": path,
                "head_sha": "a" * 40,
                "head_branch": "master",
                "event": "push",
                "repository": {"full_name": "ghalrym/Kadan"},
                "run_number": 1,
                "run_attempt": 1,
                "status": "completed",
                "conclusion": "success",
            }
            for path in release.WORKFLOWS
        ]
        runs.append({**runs[0], "run_attempt": 2, "conclusion": "failure"})
        with (
            patch.object(release, "github", return_value={"workflow_runs": runs}),
            self.assertRaises(ValueError),
        ):
            release.require_ci("a" * 40)

    def test_outsider_approval_does_not_admit_publication(self):
        pr = {
            "merged_at": "now",
            "base": {"ref": "master"},
            "merge_commit_sha": "a" * 40,
            "number": 1,
            "head": {"sha": "b" * 40},
            "user": {"login": "author"},
        }
        review = {
            "state": "APPROVED",
            "user": {"login": "outsider"},
            "commit_id": "b" * 40,
        }
        with patch.object(
            release, "github", side_effect=[[pr], [review], {"permission": "read"}]
        ):
            self.assertFalse(release.approved("a" * 40))

    def test_stale_head_approval_does_not_admit_publication(self):
        pr = {
            "merged_at": "now",
            "base": {"ref": "master"},
            "merge_commit_sha": "a" * 40,
            "number": 1,
            "head": {"sha": "b" * 40},
            "user": {"login": "author"},
        }
        review = {
            "state": "APPROVED",
            "user": {"login": "owner"},
            "commit_id": "c" * 40,
        }
        with patch.object(release, "github", side_effect=[[pr], [review]]) as github:
            self.assertFalse(release.approved("a" * 40))
        self.assertEqual(github.call_count, 2)

    def test_build_uses_versioned_tag_and_captures_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "compose.yaml").write_text("name: kadan\n")
            previous = manifest()

            def run(command, **kwargs):
                if command[0] == "git":
                    if command[1] == "rev-parse":
                        return Mock(stdout="b" * 40)
                    if command[1] == "diff":
                        return Mock(stdout="services/kadan/web/main.ts\n")
                    return Mock(stdout="")
                if "imagetools" in command:
                    return Mock(stdout=json.dumps("sha256:" + "2" * 64))
                return Mock(stdout="")

            with (
                patch.object(
                    release, "github", return_value={"commit": {"sha": "b" * 40}}
                ),
                patch.object(release, "approved", return_value=True),
                patch.object(release, "require_ci"),
                patch.object(release.subprocess, "run", side_effect=run) as calls,
            ):
                target = release.publish(
                    root, previous, root / "release.json", "b" * 40
                )
            builds = [call for call in calls.call_args_list if "build" in call.args[0]]
            self.assertEqual(len(builds), 1)
            self.assertEqual(builds[0].args[0][-1], "kadan-frontend")
            self.assertNotIn("--push", builds[0].args[0])
            docker_calls = [
                call.args[0]
                for call in calls.call_args_list
                if call.args[0][0] == "docker"
            ]
            self.assertEqual(
                docker_calls[1],
                ["docker", "push", "ghcr.io/ghalrym/kadan/kadan-frontend:" + "b" * 40],
            )
            self.assertEqual(docker_calls[2][1:4], ["buildx", "imagetools", "inspect"])
            self.assertEqual(target.images["hikari-api"], previous.images["hikari-api"])
            self.assertTrue(
                target.images["kadan-frontend"].endswith("@sha256:" + "2" * 64)
            )

    def test_failed_push_does_not_resolve_or_publish_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "compose.yaml").write_text("name: kadan\n")

            def run(command, **kwargs):
                if command[:2] == ["git", "rev-parse"]:
                    return Mock(stdout="b" * 40)
                if command[:2] == ["docker", "push"]:
                    raise subprocess.CalledProcessError(1, command)
                return Mock(stdout="")

            with (
                patch.object(
                    release, "github", return_value={"commit": {"sha": "b" * 40}}
                ),
                patch.object(release, "approved", return_value=True),
                patch.object(release, "require_ci"),
                patch.object(release.subprocess, "run", side_effect=run) as calls,
                self.assertRaises(subprocess.CalledProcessError),
            ):
                release.publish(root, None, root / "release.json", "b" * 40)
            self.assertFalse((root / "release.json").exists())
            self.assertFalse(
                any("imagetools" in call.args[0] for call in calls.call_args_list)
            )


class RecoveryBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.backend = object.__new__(rollout.Compose)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.backend.state = Path(self.directory.name)
        rollout.fence(self.backend.state, [])
        self.backend.timeout = 0

    def test_live_drift_is_refused(self):
        self.backend.command = Mock(return_value="container")
        observed = [
            {
                "Config": {
                    "Labels": {"com.docker.compose.service": "kadan-frontend"},
                    "Image": "unexpected",
                },
                "State": {"Running": True},
            }
        ]
        with (
            patch.object(
                rollout.subprocess,
                "run",
                return_value=Mock(stdout=json.dumps(observed)),
            ),
            self.assertRaises(ValueError),
        ):
            self.backend.assert_running(manifest())

    def test_stopped_affected_owner_can_drain_during_recovery(self):
        self.backend.status = Mock(
            side_effect=lambda service: (
                (_ for _ in ()).throw(ValueError("no socket"))
                if service == "kadan-api"
                else {"fenced": True, "active": 0}
            )
        )
        self.backend.stopped = Mock(return_value=True)
        with patch.object(rollout, "fence"):
            self.backend.drain(("kadan-api",))
        self.backend.stopped.assert_called_once_with("kadan-api")

    def test_running_unobservable_old_owner_cannot_be_drained(self):
        self.backend.status = Mock(side_effect=ValueError("no socket"))
        self.backend.stopped = Mock(return_value=False)
        with patch.object(rollout, "fence"), self.assertRaises(ValueError):
            self.backend.drain(("kadan-api",))

    def test_only_proven_new_fenced_target_is_stopped_for_recovery(self):
        target = manifest("b", "2", "kadan-api").model_dump()
        target["images"]["kadan-runner"] = target["images"]["kadan-api"]
        journal = {"target": target, "affected": ["kadan-api"]}
        rollout.fence(self.backend.state, list(rollout.OWNERS))
        self.backend.status = Mock(side_effect=ValueError("no socket"))
        self.backend.command = Mock(return_value="container")
        observed = [
            {
                "Config": {
                    "Image": target["images"]["kadan-api"],
                    "Env": ["KADAN_ROLLOUT_SERVICE=kadan-api"],
                },
                "Mounts": [
                    {
                        "Source": str(self.backend.state / "gate"),
                        "Destination": "/run/kadan/rollout",
                        "RW": False,
                    }
                ],
            }
        ]
        with patch.object(
            rollout.subprocess, "run", return_value=Mock(stdout=json.dumps(observed))
        ):
            self.backend.stop_failed_replacements(journal)
        self.assertEqual(
            self.backend.command.call_args.args,
            ("stop", "--timeout", "10", "kadan-api"),
        )

    def test_unproven_mount_never_stops_container(self):
        rollout.fence(self.backend.state, list(rollout.OWNERS))
        target = manifest("b", "2", "kadan-api").model_dump()
        target["images"]["kadan-runner"] = target["images"]["kadan-api"]
        self.backend.status = Mock(side_effect=ValueError("no socket"))
        self.backend.command = Mock(return_value="container")
        observed = [
            {
                "Config": {"Image": target["images"]["kadan-api"], "Env": []},
                "Mounts": [],
            }
        ]
        with (
            patch.object(
                rollout.subprocess,
                "run",
                return_value=Mock(stdout=json.dumps(observed)),
            ),
            self.assertRaises(RuntimeError),
        ):
            self.backend.stop_failed_replacements(
                {"target": target, "affected": ["kadan-api"]}
            )
        self.assertEqual(self.backend.command.call_count, 1)


class FinalSafetyTests(unittest.TestCase):
    def test_recovery_never_reopens_downstream_fences(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = object.__new__(rollout.Compose)
            backend.state = Path(directory)
            backend.timeout = 0
            backend.status = Mock(return_value={"fenced": True, "active": 0})
            rollout.fence(backend.state, list(rollout.OWNERS))
            with patch.object(rollout, "fence") as fence:
                backend.drain(("kadan-api",))
            self.assertTrue(
                all(
                    set(call.args[1]) == set(rollout.OWNERS)
                    for call in fence.call_args_list
                )
            )

    def test_recovery_configuration_drift_blocks_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            rollout.atomic_json(
                state / "state.json",
                {
                    "good": manifest().model_dump(),
                    "pending": {"phase": "replacing"},
                },
            )
            backend = Mock(spec=rollout.Compose)
            backend.snapshot_epochs.return_value = {}
            backend.configuration.side_effect = ValueError("environment changed")
            with self.assertRaises(ValueError):
                rollout.rollout(manifest("b"), state, backend)
            backend.drain.assert_not_called()
            backend.replace.assert_not_called()
            backend.stop_failed_replacements.assert_not_called()

    def test_truncated_review_history_fails_closed(self):
        pr = {
            "merged_at": "now",
            "base": {"ref": "master"},
            "merge_commit_sha": "a" * 40,
            "number": 1,
            "head": {"sha": "b" * 40},
            "user": {"login": "author"},
        }
        with (
            patch.object(release, "github", side_effect=[[pr], [{}] * 100]),
            self.assertRaises(ValueError),
        ):
            release.approved("a" * 40)


class CachedRecoveryTests(unittest.TestCase):
    def test_registry_outage_during_prefetch_never_closes_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            current, target = manifest(), manifest("b", "2")
            rollout.atomic_json(
                state / "state.json", {"good": current.model_dump(), "pending": None}
            )
            rollout.fence(state, [])
            backend = Mock(spec=rollout.Compose)
            backend.snapshot_epochs.return_value = {}
            backend.prepare_images.side_effect = OSError("registry unavailable")
            with self.assertRaises(OSError):
                rollout.rollout(target, state, backend)
            backend.drain.assert_not_called()
            backend.replace.assert_not_called()
            backend.replace_native.assert_not_called()
            self.assertIsNone(json.loads((state / "state.json").read_text())["pending"])
            self.assertEqual(
                json.loads((state / "gate/fence.json").read_text())["closed"], []
            )

    def test_cached_rollback_is_verified_before_target_registry_access(self):
        backend = object.__new__(rollout.Compose)
        backend.timeout = 1
        backend.command = Mock()
        backend.overlay = Mock()
        with (
            patch.object(
                rollout.subprocess, "run", side_effect=OSError("rollback image missing")
            ),
            self.assertRaises(OSError),
        ):
            backend.prepare_images(manifest(), manifest("b", "2"), ("kadan-frontend",))
        backend.command.assert_not_called()
        backend.overlay.assert_not_called()

    def test_failed_target_pull_restores_previous_overlay(self):
        backend = object.__new__(rollout.Compose)
        backend.timeout = 1
        current, target = manifest(), manifest("b", "2")
        backend.command = Mock(side_effect=OSError("registry unavailable"))
        backend.overlay = Mock()
        with patch.object(rollout.subprocess, "run"), self.assertRaises(OSError):
            backend.prepare_images(current, target, ("kadan-frontend",))
        self.assertEqual(
            [call.args[0] for call in backend.overlay.call_args_list], [target, current]
        )

    def test_replacement_uses_cached_pins_without_registry(self):
        backend = object.__new__(rollout.Compose)
        backend.timeout = 1
        backend.command = Mock()
        backend.overlay = Mock()
        backend.replace(manifest(), ("kadan-frontend",))
        commands = [call.args for call in backend.command.call_args_list]
        self.assertFalse(any(command[0] == "pull" for command in commands))
        self.assertEqual(commands[0][commands[0].index("--pull") + 1], "never")

    def test_missing_cached_recovery_pin_never_stops_any_container(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = Mock(spec=rollout.Compose)
            backend.snapshot_epochs.return_value = {}
            backend.cached_images.side_effect = OSError("cached old image removed")
            journal = {
                "previous": manifest().model_dump(),
                "target": manifest("b", "2").model_dump(),
                "affected": ["kadan-frontend"],
                "phase": "replacing",
                "native": None,
            }
            state = Path(directory)
            rollout.fence(state, list(rollout.OWNERS))
            rollout.save_pending(state, journal)
            with self.assertRaises(OSError):
                rollout.recover(state, backend, journal)
            self.assertEqual(
                set(json.loads((state / "gate/fence.json").read_text())["closed"]),
                set(rollout.OWNERS),
            )
            self.assertIsNotNone(
                json.loads((state / "state.json").read_text())["pending"]
            )
            backend.stop_failed_replacements.assert_not_called()
            backend.drain.assert_not_called()
            backend.replace.assert_not_called()

    def test_recovery_never_prefetches_or_contacts_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            current = manifest()
            backend = Mock(spec=rollout.Compose)
            backend.snapshot_epochs.return_value = {}
            journal = {
                "previous": current.model_dump(),
                "target": manifest("b", "2").model_dump(),
                "affected": ["kadan-frontend"],
                "phase": "replacing",
                "native": None,
            }
            rollout.fence(state, list(rollout.OWNERS))
            rollout.recover(state, backend, journal)
            backend.cached_images.assert_called_once_with(current, ("kadan-frontend",))
            backend.prepare_images.assert_not_called()
            backend.replace.assert_called_once_with(current, ("kadan-frontend",))

    def test_failed_health_recovers_cached_old_images_during_registry_outage(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            current, target = manifest(), manifest("b", "2")
            rollout.atomic_json(
                state / "state.json", {"good": current.model_dump(), "pending": None}
            )
            rollout.fence(state, [])
            backend = object.__new__(rollout.Compose)
            backend.state, backend.timeout = state, 1
            backend.configuration = Mock()
            backend.verify_open = Mock()
            backend.snapshot_epochs = Mock(return_value={})
            backend.unchanged_epochs = Mock()
            backend.unaffected_running = Mock()
            backend.preflight = Mock()
            backend.prepare_images = Mock()
            backend.overlay = Mock()
            backend.drain = Mock(
                side_effect=lambda *args: rollout.fence(state, list(rollout.OWNERS))
            )
            backend.stop_failed_replacements = Mock()
            backend.verify = Mock(
                side_effect=[RuntimeError("failed target smoke"), None]
            )

            def command(*args):
                if args[0] == "pull":
                    raise OSError("registry offline after prefetch")
                return ""

            backend.command = Mock(side_effect=command)
            with (
                patch.object(rollout.subprocess, "run"),
                self.assertRaises(RuntimeError),
            ):
                rollout.rollout(target, state, backend)
            self.assertEqual(
                json.loads((state / "state.json").read_text())["good"],
                current.model_dump(),
            )
            self.assertEqual(
                json.loads((state / "gate/fence.json").read_text())["closed"], []
            )
            self.assertEqual(
                [call.args[0] for call in backend.overlay.call_args_list],
                [target, current],
            )
            self.assertTrue(
                all(call.args[0] == "up" for call in backend.command.call_args_list)
            )
