"""Exercise real job ownership with tiny local output fixtures, never model weights."""
from pathlib import Path
import tempfile
import threading
import unittest

from api.inference.video import VideoSpec
from api.inference.resources import ResourceCancelled
from api.services.video_jobs import VideoJobs


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
        job = self.jobs.submit('test', VideoSpec('hello'))
        self.jobs._thread.join(2)
        finished = self.jobs.get(job.id)
        self.assertEqual(finished.status, 'Done')
        self.assertEqual(finished.progress, 100)
        self.assertEqual(self.jobs.content(job.id).read_bytes(), b'test MP4 fixture')
        self.assertEqual(finished.output_url, f'/v1/videos/{job.id}/content')

    def test_failure_cleans_partial_and_allows_retry(self):
        job = self.jobs.submit('test', VideoSpec('fail'))
        self.jobs._thread.join(2)
        self.assertEqual(self.jobs.get(job.id).status, 'Failed')
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])
        self.jobs.submit('test', VideoSpec('retry'))

    def test_cancel_waits_for_worker_and_rejects_overlap(self):
        job = self.jobs.submit('test', VideoSpec('wait'))
        with self.assertRaises(ValueError):
            self.jobs.submit('test', VideoSpec('overlap'))
        self.jobs.cancel(job.id)
        self.jobs._thread.join(2)
        self.assertEqual(self.jobs.get(job.id).status, 'Cancelled')
        with self.assertRaises(KeyError):
            self.jobs.content(job.id)

    def test_invalid_request_does_not_create_job(self):
        with self.assertRaises(ValueError):
            self.jobs.submit('test', VideoSpec('invalid'))
        self.assertEqual(self.jobs.list(), [])
