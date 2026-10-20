"""A local pinned storage overlay never comes from or weakens a release artifact."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from bin import rollout
from bin.tests.test_rollout import manifest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HostOverlayTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.checkout, self.host, self.state = (
            self.root / name for name in ("checkout", "host", "state")
        )
        for path in (self.checkout, self.host, self.state):
            path.mkdir()
        (self.checkout / "compose.yaml").write_text("name: kadan\nservices: {}\n")
        (self.checkout / ".env").write_text("HIKARI_WORKER_TOKEN=unit-placeholder\n")
        self.data = self.host / "postgres"
        self.data.mkdir()
        self.instructions = self.host / "AGENT.md"
        self.instructions.write_text("Host instructions\n")
        self.config = self.host / "compose.host.json"
        self.document = {
            "services": {
                "hikari-postgres": {
                    "volumes": [
                        {
                            "type": "bind",
                            "source": str(self.data),
                            "target": "/var/lib/postgresql",
                        }
                    ]
                },
                "kadan-runner": {
                    "volumes": [
                        {
                            "type": "bind",
                            "source": str(self.instructions),
                            "target": "/app/AGENT.md",
                            "read_only": True,
                        }
                    ],
                    "environment": {"KADAN_AGENT_INSTRUCTION_FILE": "/app/AGENT.md"},
                },
            }
        }
        self.installation = {
            "project": "existing-project",
            "env_sha256": digest(self.checkout / ".env"),
            "supervision": "detached-independent",
        }
        self.enroll(self.document)
        self.release = manifest().model_copy(
            update={"compose_sha256": digest(self.checkout / "compose.yaml")}
        )
        rollout.atomic_json(
            self.state / "state.json",
            {"good": self.release.model_dump(), "pending": None},
        )
        rollout.fence(self.state, [])

    def enroll(self, document, pins=None):
        self.config.write_text(json.dumps(document))
        self.installation["host_overlay"] = {
            "path": str(self.config),
            "sha256": digest(self.config),
            "content_pins": pins
            if pins is not None
            else [
                {"path": str(self.instructions), "sha256": digest(self.instructions)}
            ],
        }
        rollout.atomic_json(self.state / "installation.json", self.installation)

    def backend(self):
        return rollout.Compose(self.checkout, self.state)

    def test_files_layer_canonical_then_host_then_images(self):
        backend = self.backend()
        with patch.object(
            rollout.subprocess, "run", return_value=Mock(stdout="")
        ) as run:
            backend.command("config", "--quiet")
        command = run.call_args.args[0]
        files = [
            command[index + 1] for index, part in enumerate(command) if part == "-f"
        ]
        self.assertEqual(
            files,
            [
                str(self.checkout / "compose.yaml"),
                str(self.config),
                str(self.state / "images.json"),
            ],
        )
        self.assertEqual(
            command[command.index("--project-name") + 1], "existing-project"
        )

    def test_image_overlay_does_not_copy_or_erase_host_mounts(self):
        backend = self.backend()
        backend.overlay(self.release)
        generated = json.loads((self.state / "images.json").read_text())
        self.assertNotIn("hikari-postgres", generated["services"])
        self.assertEqual(
            generated["services"]["kadan-runner"]["volumes"][0]["target"],
            "/run/kadan/rollout",
        )
        self.assertEqual(json.loads(self.config.read_text()), self.document)

    def test_overlay_hash_drift_blocks_compose_mutation(self):
        backend = self.backend()
        self.config.write_text('{"services":{}}')
        with (
            patch.object(rollout.subprocess, "run") as run,
            self.assertRaises(ValueError),
        ):
            backend.command("up", "-d", "kadan-runner")
        run.assert_not_called()

    def test_instruction_content_drift_blocks_raw_docker_stop(self):
        backend = self.backend()
        self.instructions.write_text("Changed host instructions\n")
        with (
            patch.object(rollout.subprocess, "run") as run,
            self.assertRaises(ValueError),
        ):
            backend.stop_ids(("exact-container-id",))
        run.assert_not_called()

    def test_pending_recovery_checks_drift_before_docker_and_keeps_state(self):
        backend = self.backend()
        pending = {"phase": "replacing", "previous": self.release.model_dump()}
        rollout.atomic_json(
            self.state / "state.json",
            {"good": self.release.model_dump(), "pending": pending},
        )
        rollout.fence(self.state, list(rollout.OWNERS))
        self.config.write_text('{"services":{}}')
        with (
            patch.object(rollout.subprocess, "run") as run,
            self.assertRaises(ValueError),
        ):
            rollout.resume_locked(self.state, backend)
        run.assert_not_called()
        self.assertEqual(
            json.loads((self.state / "state.json").read_text())["pending"], pending
        )
        self.assertTrue(
            json.loads((self.state / "gate/fence.json").read_text())["closed"]
        )

    def test_unpinned_instruction_bind_is_refused(self):
        self.enroll(self.document, pins=[])
        with self.assertRaisesRegex(ValueError, "content pins"):
            self.backend()

    def test_jaeger_file_bind_requires_readonly_pin(self):
        self.document["services"]["jaeger"] = {
            "volumes": [
                {
                    "type": "bind",
                    "source": str(self.instructions),
                    "target": "/etc/jaeger/config.yaml",
                    "read_only": False,
                }
            ]
        }
        self.enroll(self.document)
        with self.assertRaisesRegex(ValueError, "read-only"):
            self.backend()

    def test_app_code_and_executable_mounts_are_refused(self):
        for target in (
            "/app",
            "/app/kadan",
            "/opt/kadan/bin",
            "/usr/lib",
            "/run/kadan/rollout",
            "/",
        ):
            with self.subTest(target=target):
                self.enroll(
                    {
                        "services": {
                            "kadan-api": {
                                "volumes": [
                                    {
                                        "type": "bind",
                                        "source": str(self.data),
                                        "target": target,
                                    }
                                ]
                            }
                        }
                    }
                )
                with self.assertRaises(ValueError):
                    self.backend()

    def test_api_preserves_readonly_checkpoint_bind(self):
        mounts = [
            {
                "type": "bind",
                "source": str(self.data),
                "target": target,
                "read_only": True,
            }
            for target in ("/var/lib/kadan/models", "/mnt/glm-models")
        ]
        self.document["services"]["hikari-api"] = {"volumes": mounts}
        self.enroll(self.document)
        self.backend()
        self.assertEqual(
            json.loads(self.config.read_text())["services"]["hikari-api"]["volumes"],
            mounts,
        )

    def test_api_checkpoint_bind_requires_explicit_readonly(self):
        for readonly in (None, False, "true"):
            with self.subTest(readonly=readonly):
                mount = {
                    "type": "bind",
                    "source": str(self.data),
                    "target": "/mnt/glm-models",
                }
                if readonly is not None:
                    mount["read_only"] = readonly
                self.enroll({"services": {"hikari-api": {"volumes": [mount]}}})
                with self.assertRaises(ValueError):
                    self.backend()

    def test_api_checkpoint_mount_keeps_source_and_target_validation(self):
        for changes in (
            {"source": "relative-models"},
            {"source": str(self.host / "missing-models")},
            {"source": str(self.host / "$MODEL_PATH")},
            {"target": "/mnt/glm-models/../app"},
            {"target": "/mnt/glm-models-other"},
            {"type": "volume", "source": "model_data"},
        ):
            with self.subTest(changes=changes):
                mount = {
                    "type": "bind",
                    "source": str(self.data),
                    "target": "/mnt/glm-models",
                    "read_only": True,
                    **changes,
                }
                self.enroll({"services": {"hikari-api": {"volumes": [mount]}}})
                with self.assertRaises(ValueError):
                    self.backend()

    def test_protected_runtime_overrides_are_refused(self):
        for key in (
            "image",
            "build",
            "command",
            "entrypoint",
            "depends_on",
            "healthcheck",
            "profiles",
            "scale",
            "network_mode",
        ):
            with self.subTest(key=key):
                self.enroll({"services": {"kadan-api": {key: "untrusted"}}})
                with self.assertRaises(ValueError):
                    self.backend()

    def test_reserved_environment_and_replication_are_refused(self):
        for service in (
            {"environment": {"KADAN_ROLLOUT_SERVICE": "other"}},
            {"deploy": {"replicas": 2}},
        ):
            with self.subTest(service=service):
                self.enroll({"services": {"kadan-api": service}})
                with self.assertRaises(ValueError):
                    self.backend()

    def test_incoming_release_cannot_supply_overlay(self):
        with self.assertRaises(ValueError):
            rollout.Release(
                **self.release.model_dump(),
                host_overlay=self.installation["host_overlay"],
            )

    def test_unknown_services_and_relative_binds_are_refused(self):
        for document in (
            {"services": {"runner-2": {}}},
            {"include": ["other.json"]},
            {
                "services": {
                    "tsuchi-api": {
                        "volumes": [
                            {"type": "bind", "source": "relative", "target": "/data"}
                        ]
                    }
                }
            },
        ):
            with self.subTest(document=document):
                self.enroll(document)
                with self.assertRaises(ValueError):
                    self.backend()

    def test_storage_roots_external_models_and_operational_settings_are_preserved(self):
        document = {
            "services": {
                "hikari-worker-llm": {
                    "volumes": [
                        {
                            "type": "bind",
                            "source": str(self.data),
                            "target": "/mnt/glm-models",
                            "read_only": True,
                        }
                    ],
                    "environment": {"LOCAL_SETTING": "${LOCAL_SETTING}"},
                    "restart": "unless-stopped",
                    "init": True,
                    "read_only": True,
                    "cap_drop": ["ALL"],
                    "security_opt": ["no-new-privileges:true"],
                    "mem_limit": "2g",
                    "cpus": 2,
                    "pids_limit": 64,
                    "tmpfs": ["/tmp:size=64m"],
                    "deploy": {"resources": {"limits": {"cpus": "2"}}},
                },
                "kadan-frontend": {
                    "ports": ["127.0.0.1:8088:8080", "192.0.2.10:8088:8080"]
                },
            },
            "volumes": {"host_store": {"external": True, "name": "existing_store"}},
        }
        self.enroll(document)
        self.backend().configuration(self.release)

    def test_data_directory_changes_are_not_confused_with_configuration_drift(self):
        backend = self.backend()
        (self.data / "database-file").write_text("owner-managed mutable storage")
        backend.configuration(self.release)

    def test_canonical_only_enrollment_remains_supported(self):
        del self.installation["host_overlay"]
        rollout.atomic_json(self.state / "installation.json", self.installation)
        backend = self.backend()
        with patch.object(
            rollout.subprocess, "run", return_value=Mock(stdout="")
        ) as run:
            backend.command("config", "--quiet")
        self.assertNotIn(str(self.config), run.call_args.args[0])

    def test_named_volume_names_and_external_flags_cannot_interpolate_or_hide_objects(
        self,
    ):
        for value in (
            {"name": "${HOST_VOLUME}"},
            {"name": ""},
            {"name": {"other": "volume"}},
            {"name": "/host/path"},
            {"external": "true"},
            {"external": {"name": "other"}},
            {"driver_opts": {"device": "/host/path"}},
        ):
            with self.subTest(value=value):
                self.enroll({"volumes": {"host_store": value}})
                with self.assertRaises(ValueError):
                    self.backend()

    def test_tmpfs_cannot_mask_application_code(self):
        self.enroll({"services": {"kadan-api": {"tmpfs": ["/app:size=1m"]}}})
        with self.assertRaisesRegex(ValueError, "tmpfs"):
            self.backend()

    def test_lan_port_addition_keeps_canonical_ports_and_final_image_pins_separate(
        self,
    ):
        canonical = {"services": {"kadan-frontend": {"ports": ["127.0.0.1:8630:8080"]}}}
        (self.checkout / "compose.yaml").write_text(json.dumps(canonical))
        local = {"services": {"kadan-frontend": {"ports": ["192.0.2.10:8630:8080"]}}}
        self.enroll(local)
        backend = self.backend()
        backend.overlay(self.release)
        self.assertEqual(
            json.loads((self.checkout / "compose.yaml").read_text()), canonical
        )
        self.assertEqual(json.loads(self.config.read_text()), local)
        images = json.loads((self.state / "images.json").read_text())
        self.assertEqual(
            images["services"]["kadan-frontend"],
            {"image": self.release.images["kadan-frontend"]},
        )
        with patch.object(
            rollout.subprocess, "run", return_value=Mock(stdout="")
        ) as run:
            backend.command("config", "--quiet")
        command = run.call_args.args[0]
        self.assertLess(
            command.index(str(self.checkout / "compose.yaml")),
            command.index(str(self.config)),
        )
        self.assertLess(
            command.index(str(self.config)),
            command.index(str(self.state / "images.json")),
        )
