import unittest

from api.services.inference_worker import (
    InferenceJob,
    InferenceTask,
    encode_request,
)


class RequestBoundTests(unittest.IsolatedAsyncioTestCase):
    def test_image_upload_has_a_larger_bounded_transport(self):
        encode_request(
            {"input": {"feature": "image", "image": "x" * (4 * 1024 * 1024)}}
        )
        with self.assertRaisesRegex(Exception, "transport limit"):
            encode_request(
                {"input": {"feature": "image", "image": "x" * (64 * 1024 * 1024)}}
            )
        with self.assertRaisesRegex(Exception, "transport limit"):
            encode_request(
                {"input": {"feature": "llm", "text": "x" * (4 * 1024 * 1024)}}
            )

    async def test_image_progress_is_initialized_with_the_job(self):
        job = InferenceJob(
            InferenceTask(
                input={"feature": "image", "prompt": "test", "count": 2, "steps": 20}
            ),
            False,
        )
        self.assertEqual(job.image_progress.image_count, 2)
        self.assertEqual(job.image_progress.steps_total, 20)
