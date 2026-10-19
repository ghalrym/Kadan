"""Offline log extraction units; no inference or external process."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bin.profiling import projection_samples


class ProjectionSamplesTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.report = Path(self.directory.name) / "worker.log"

    def test_samples_omit_unrelated_payloads(self):
        record = {
            "phase": "prefill",
            "tokens": 9,
            "seconds": 2,
            "projection_seconds": 1,
            "prepare_seconds": 0.1,
            "host_wait_seconds": 0.2,
            "projection_detail": {
                "version": 1,
                "groups": [],
                "dropped": 0,
                "batch_wall_seconds": 0.3,
            },
            "prompt": "private",
        }
        self.report.write_text(
            "unrelated secret\nworker | KADAN_TEXT_TIMING " + json.dumps(record)
        )
        result = projection_samples(self.report)
        self.assertEqual(result["samples"][0]["tokens"], 9)
        self.assertEqual(
            result["phases"]["prefill"]["spread"]["seconds"],
            {"min": 2, "median": 2, "max": 2},
        )
        self.assertNotIn("private", json.dumps(result))
        self.assertNotIn("unrelated", json.dumps(result))

    def test_samples_require_opt_in(self):
        self.report.write_text('KADAN_TEXT_TIMING {"phase":"prefill"}')
        with self.assertRaisesRegex(ValueError, "No opted-in"):
            projection_samples(self.report)

    def test_samples_reject_unknown_version(self):
        self.report.write_text('KADAN_TEXT_TIMING {"projection_detail":{"version":2}}')
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            projection_samples(self.report)
