import os
import unittest
from pathlib import Path
from unittest.mock import patch

from api.services.video_jobs import VideoJobs


class VideoJobsConfigurationTests(unittest.TestCase):
    def test_video_directory_uses_hikari_then_legacy_then_persistent_default(self):
        cases = (
            ({}, "/tmp/hikari-test-home/.local/share/kadan/videos"),
            (
                {"KADAN_VIDEO_DIR": "~/legacy-videos"},
                "/tmp/hikari-test-home/legacy-videos",
            ),
            (
                {
                    "HIKARI_VIDEO_DIR": "~/hikari-videos",
                    "KADAN_VIDEO_DIR": "~/legacy-videos",
                },
                "/tmp/hikari-test-home/hikari-videos",
            ),
        )
        for environment, expected in cases:
            with (
                self.subTest(environment=environment),
                patch.dict(
                    os.environ,
                    {"HOME": "/tmp/hikari-test-home", **environment},
                    clear=True,
                ),
            ):
                self.assertEqual(VideoJobs().root, Path(expected))

    def test_explicit_root_overrides_video_directory_environment(self):
        with patch.dict(
            os.environ,
            {"HIKARI_VIDEO_DIR": "/hikari-videos", "KADAN_VIDEO_DIR": "/legacy-videos"},
            clear=True,
        ):
            self.assertEqual(
                VideoJobs("~/explicit").root, Path("~/explicit").expanduser()
            )
