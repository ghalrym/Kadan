import unittest

from starlette.requests import Request

from api.pydantic_models.media import VideoJob
from api.routes.v1.videos import public_job


def job(output_url):
    return VideoJob(
        id="clip",
        prompt="A boat",
        duration="5s",
        resolution="480p",
        aspect="wide",
        fps="24",
        progress=100,
        progressText="Done",
        time="now",
        status="Done",
        thumbnail="",
        output_url=output_url,
    )


def request(root_path=""):
    return Request({"type": "http", "path": "/", "root_path": root_path, "headers": []})


class PublicJobTests(unittest.TestCase):
    def test_content_link_includes_the_mount_prefix(self):
        self.assertEqual(
            public_job(job("/v1/videos/clip/content"), request("/hikari")).output_url,
            "/hikari/v1/videos/clip/content",
        )

    def test_unmounted_or_missing_links_are_unchanged(self):
        self.assertEqual(
            public_job(job("/v1/videos/clip/content"), request()).output_url,
            "/v1/videos/clip/content",
        )
        self.assertIsNone(public_job(job(None), request("/hikari")).output_url)
