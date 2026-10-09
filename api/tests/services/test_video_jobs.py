"""Exercise real job ownership with tiny local output fixtures, never model weights."""
from pathlib import Path
import tempfile
import threading
import unittest

from api.inference.video import VideoSpec
from api.inference.resources import ResourceCancelled
from api.services.video_jobs import VideoJobs
from api.inference.errors import InferenceFailure


class Provider:
    def validate(self, spec):
        if spec.prompt == 'invalid':
            raise ValueError('invalid')

    def generate(self, spec, output, cancellation):
        if spec.prompt == 'fail':
            output.write_bytes(b'partial')
            raise RuntimeError('provider failed')
        if spec.prompt == 'wait':
            cancellation.wait(5)
            raise ResourceCancelled()
        output.write_bytes(b'test MP4 fixture')


class VideoJobsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.jobs = VideoJobs(Path(self.directory.name), factory=lambda _: Provider())

    def tearDown(self):
        self.jobs.close()
        self.directory.cleanup()

    def test_output_published_only_after_success(self):
        self.jobs.run('a' * 32, 'test', VideoSpec('hello'), threading.Event())
        finished = self.jobs.get('a' * 32)
        self.assertEqual(finished.status, 'Done')
        self.assertEqual(finished.progress, 100)
        self.assertEqual(self.jobs.content(finished.id).read_bytes(), b'test MP4 fixture')
        self.assertEqual(finished.output_url, f'/v1/videos/{finished.id}/content')

    def test_failure_cleans_partial_and_allows_retry(self):
        with self.assertRaises(InferenceFailure):
            self.jobs.run('a' * 32, 'test', VideoSpec('fail'), threading.Event())
        self.assertEqual(self.jobs.get('a' * 32).status, 'Failed')
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])
        self.jobs.run('b' * 32, 'test', VideoSpec('retry'), threading.Event())

    def test_cancel_before_execution_does_not_publish(self):
        job = self.jobs.prepare('a' * 32, VideoSpec('wait'))
        self.jobs.cancel(job.id)
        with self.assertRaises(ResourceCancelled):
            self.jobs.run(job.id, 'test', VideoSpec('wait'), threading.Event())
        self.assertEqual(self.jobs.get(job.id).status, 'Cancelled')
        with self.assertRaises(KeyError):
            self.jobs.content(job.id)

    def test_invalid_request_does_not_create_job(self):
        with self.assertRaises(ValueError):
            self.jobs.validate('test', VideoSpec('invalid'))
        self.assertEqual(self.jobs.list(), [])
