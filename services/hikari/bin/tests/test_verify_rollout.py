"""Unit checks for disposable-verifier safety and orchestration, without Docker."""

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bin import rollout, verify_rollout

PIN = "ghcr.io/ghalrym/kadan/kadan-api@sha256:" + "a" * 64


class CachedImageTests(unittest.TestCase):
    def test_requires_real_registry_digest_syntax_before_any_docker_call(self):
        for image in ("", "python:3.14", "local@sha256:" + "a" * 64):
            with (
                self.subTest(image=image),
                patch("bin.verify_rollout.subprocess.run") as run,
            ):
                with self.assertRaisesRegex(ValueError, "genuine cached GHCR"):
                    verify_rollout.cached_image(image)
                run.assert_not_called()

    def test_requires_exact_cached_repo_digest(self):
        with patch(
            "bin.verify_rollout.subprocess.run",
            return_value=SimpleNamespace(stdout=json.dumps([{"RepoDigests": [PIN]}])),
        ) as run:
            verify_rollout.cached_image(PIN)
        self.assertEqual(
            run.call_args.args[0],
            [
                "docker",
                "--host",
                "unix:///var/run/docker.sock",
                "image",
                "inspect",
                PIN,
            ],
        )
        self.assertEqual(run.call_args.kwargs["timeout"], 30)

    def test_missing_image_fails_without_pull_or_build(self):
        with (
            patch(
                "bin.verify_rollout.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "docker"),
            ),
            self.assertRaisesRegex(ValueError, "no resources were created"),
        ):
            verify_rollout.cached_image(PIN)

    def test_different_digest_is_rejected(self):
        with (
            patch(
                "bin.verify_rollout.subprocess.run",
                return_value=SimpleNamespace(stdout='[{"RepoDigests":[]}]'),
            ),
            self.assertRaisesRegex(ValueError, "did not attest"),
        ):
            verify_rollout.cached_image(PIN)


class DisposableTopologyTests(unittest.TestCase):
    def test_fixture_is_isolated_cpu_only_with_pinned_instruction_and_volume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch(
                "bin.verify_rollout.subprocess.run",
                return_value=SimpleNamespace(stdout="a" * 40),
            ) as commands:
                fixture = verify_rollout.DisposableInstallation.create(root, PIN)
            self.assertEqual(commands.call_count, 1)
            self.assertEqual(commands.call_args.args[0][:2], ["git", "rev-parse"])
            backend = fixture.backend
            self.assertTrue(backend.project.startswith("kadan-verify-"))
            self.assertNotEqual(backend.project, "kadan")
            topology = json.loads((backend.checkout / "compose.yaml").read_text())
            self.assertEqual(set(topology["services"]), set(rollout.SERVICES))
            for service in topology["services"].values():
                self.assertNotIn("ports", service)
                self.assertNotIn("devices", service)
                self.assertNotIn("build", service)
                self.assertEqual(service["entrypoint"], ["python"])
                self.assertEqual(service["command"], ["/fixture/service.py", "serve"])
                self.assertEqual(service["image"], PIN)
                self.assertLessEqual(service["cpus"], 0.1)
                self.assertEqual(service["mem_limit"], "64m")
                self.assertFalse(
                    any(
                        "docker.sock" in mount["source"] for mount in service["volumes"]
                    )
                )
            host = backend.host_overlay
            self.assertFalse(host.path.is_relative_to(backend.checkout))
            document = json.loads(host.path.read_text())
            self.assertEqual(
                document["volumes"]["verification"]["name"],
                backend.project + "-verification",
            )
            for name in ("kadan-api", "kadan-runner"):
                mount = document["services"][name]["volumes"][0]
                self.assertTrue(mount["read_only"])
                self.assertEqual(mount["target"], "/app/AGENT.md")
            self.assertEqual(
                host.content_pins[0].sha256,
                hashlib.sha256(fixture.instruction.encode()).hexdigest(),
            )
            host.validate_document(backend.checkout)


class DisposableOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.report = Path(self.temporary.name) / "report.json"
        self.backend = Mock(spec=rollout.Compose)
        self.backend.project = "kadan-verify-unit"
        self.backend.state = Path(self.temporary.name)
        self.backend.command.return_value = ""
        self.fixture = Mock(spec=verify_rollout.DisposableInstallation)
        self.fixture.backend = self.backend
        self.fixture.release = Mock(spec=rollout.Release)
        for name in ("preflight_resources", "assert_owned"):
            boundary = patch("bin.verify_rollout." + name)
            boundary.start()
            self.addCleanup(boundary.stop)
        resources = patch(
            "bin.verify_rollout.existing_resources",
            side_effect=lambda project: iter(()),
        )
        resources.start()
        self.addCleanup(resources.stop)

    def test_runs_explicit_checks_and_cleans_only_its_compose_project(self):
        with (
            patch("bin.verify_rollout.cached_image") as cached,
            patch(
                "bin.verify_rollout.DisposableInstallation.create",
                return_value=self.fixture,
            ),
            patch("bin.verify_rollout.serialization") as serial,
        ):
            result = verify_rollout.verify(PIN, self.report)
        cached.assert_called_once_with(PIN)
        self.fixture.preservation.assert_called_once_with()
        self.fixture.drain.assert_called_once_with()
        self.fixture.recovery.assert_called_once_with()
        serial.assert_called_once_with(self.backend.state)
        self.assertTrue(result["passed"])
        self.assertTrue(result["cleaned"])
        self.assertEqual(json.loads(self.report.read_text()), result)
        calls = [call.args for call in self.backend.command.call_args_list]
        self.assertEqual(
            calls[-2:],
            [("down", "--volumes", "--remove-orphans"), ("ps", "--all", "--quiet")],
        )
        self.assertFalse(
            any(
                "pull" == arg or "build" == arg or "prune" == arg
                for call in calls
                for arg in call
            )
        )
        self.assertIn("--pull", calls[1])
        self.assertEqual(calls[1][calls[1].index("--pull") + 1], "never")
        self.assertIn("prepared same-pin", result["scope"])

    def test_partial_startup_failure_still_cleans_and_records_failure(self):
        def command(*args):
            if args[0] == "up":
                raise subprocess.CalledProcessError(1, "docker")
            return ""

        self.backend.command.side_effect = command
        with (
            patch("bin.verify_rollout.cached_image"),
            patch(
                "bin.verify_rollout.DisposableInstallation.create",
                return_value=self.fixture,
            ),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            verify_rollout.verify(PIN, self.report)
        self.backend.command.assert_any_call("down", "--volumes", "--remove-orphans")
        report = json.loads(self.report.read_text())
        self.assertFalse(report["passed"])
        self.assertTrue(report["cleaned"])
        self.assertEqual(report["error"], "CalledProcessError")

    def test_project_collision_never_stops_existing_resources(self):
        self.backend.command.return_value = "existing-container"
        with (
            patch("bin.verify_rollout.cached_image"),
            patch(
                "bin.verify_rollout.DisposableInstallation.create",
                return_value=self.fixture,
            ),
            self.assertRaisesRegex(ValueError, "already exists"),
        ):
            verify_rollout.verify(PIN, self.report)
        self.backend.command.assert_called_once_with("ps", "--all", "--quiet")

    def test_no_cached_pin_means_no_fixture_or_docker_creation(self):
        with (
            patch(
                "bin.verify_rollout.cached_image", side_effect=ValueError("not cached")
            ),
            patch("bin.verify_rollout.DisposableInstallation.create") as create,
            self.assertRaisesRegex(ValueError, "not cached"),
        ):
            verify_rollout.verify(PIN, self.report)
        create.assert_not_called()
        self.assertFalse(self.report.exists())

    def test_cleanup_failure_cannot_be_reported_as_passed(self):
        def command(*args):
            if args[0] == "down":
                raise subprocess.CalledProcessError(1, "docker")
            return ""

        self.backend.command.side_effect = command
        with (
            patch("bin.verify_rollout.cached_image"),
            patch(
                "bin.verify_rollout.DisposableInstallation.create",
                return_value=self.fixture,
            ),
            patch("bin.verify_rollout.serialization"),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            verify_rollout.verify(PIN, self.report)
        report = json.loads(self.report.read_text())
        self.assertFalse(report["passed"])
        self.assertFalse(report["cleaned"])
        self.assertTrue(Path(report["workdir"]).is_dir())
        self.addCleanup(shutil.rmtree, report["workdir"], ignore_errors=True)

    def test_foreign_resource_blocks_cleanup_and_retains_evidence(self):
        with (
            patch("bin.verify_rollout.cached_image"),
            patch(
                "bin.verify_rollout.DisposableInstallation.create",
                return_value=self.fixture,
            ),
            patch("bin.verify_rollout.serialization"),
            patch(
                "bin.verify_rollout.assert_owned",
                side_effect=[None, ValueError("foreign")],
            ),
            self.assertRaisesRegex(ValueError, "foreign"),
        ):
            verify_rollout.verify(PIN, self.report)
        self.assertFalse(
            any(call.args[0] == "down" for call in self.backend.command.call_args_list)
        )
        report = json.loads(self.report.read_text())
        self.assertFalse(report["cleaned"])
        self.assertFalse(report["passed"])
        self.assertTrue(Path(report["workdir"]).is_dir())
        self.addCleanup(shutil.rmtree, report["workdir"], ignore_errors=True)


class ResourceOwnershipTests(unittest.TestCase):
    def test_preexisting_foreign_named_volume_is_never_adopted(self):
        project = "kadan-verify-unit"
        with (
            patch(
                "bin.verify_rollout.docker", return_value=project + "-verification\n"
            ) as docker,
            self.assertRaisesRegex(ValueError, "nothing may be adopted"),
        ):
            verify_rollout.preflight_resources(project)
        self.assertEqual(docker.call_args.args[:2], ("volume", "ls"))
        self.assertFalse(any("rm" in call.args for call in docker.call_args_list))

    def test_preexisting_foreign_network_is_never_adopted(self):
        project = "kadan-verify-unit"
        with (
            patch(
                "bin.verify_rollout.docker", side_effect=["", project + "_default\n"]
            ) as docker,
            self.assertRaisesRegex(ValueError, "nothing may be adopted"),
        ):
            verify_rollout.preflight_resources(project)
        self.assertEqual(docker.call_args.args[:2], ("network", "ls"))

    def test_cleanup_rejects_foreign_volume_label(self):
        backend = Mock(spec=rollout.Compose)
        backend.project = "kadan-verify-unit"
        backend.command.return_value = ""
        with (
            patch(
                "bin.verify_rollout.docker",
                side_effect=[
                    backend.project + "-verification\n",
                    json.dumps(
                        [{"Labels": {"com.docker.compose.project": "another-project"}}]
                    ),
                ],
            ),
            self.assertRaisesRegex(ValueError, "foreign; cleanup refused"),
        ):
            verify_rollout.assert_owned(backend)
        backend.command.assert_called_once_with("ps", "--all", "--quiet")

    def test_cleanup_rejects_foreign_container_label(self):
        backend = Mock(spec=rollout.Compose)
        backend.project = "kadan-verify-unit"
        backend.command.return_value = "container-id"
        with (
            patch(
                "bin.verify_rollout.docker",
                return_value=json.dumps(
                    [
                        {
                            "Id": "container-id",
                            "Config": {
                                "Labels": {
                                    "com.docker.compose.project": "another-project"
                                }
                            },
                        }
                    ]
                ),
            ),
            self.assertRaisesRegex(ValueError, "cleanup refused"),
        ):
            verify_rollout.assert_owned(backend)
