import unittest

from pydantic import ValidationError

from api.routes.v1.videos.generations import VideoGenerationRequest


class VideoGenerationRequestTests(unittest.TestCase):
    def test_omitted_settings_remain_unresolved_until_tier_selection(self):
        request = VideoGenerationRequest(prompt="Scene")
        self.assertIsNone(request.duration)
        self.assertIsNone(request.resolution)

    def test_explicit_clip_choices_keep_the_full_supported_range(self):
        for duration in (1, 5, 8, 15):
            for resolution in ("480p", "768p"):
                with self.subTest(duration=duration, resolution=resolution):
                    request = VideoGenerationRequest(
                        prompt="Scene", seconds=duration, resolution=resolution
                    )
                    self.assertEqual(request.duration, duration)
                    self.assertEqual(request.resolution, resolution)

    def test_invalid_clip_settings_are_rejected(self):
        for fields in (
            {"seconds": 0},
            {"seconds": 16},
            {"seconds": 1.5},
            {"resolution": "1080p"},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                VideoGenerationRequest(prompt="Scene", **fields)
