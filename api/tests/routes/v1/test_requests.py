import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from starlette.requests import Request
from api.routes.v1.videos.generations import generate_video, VideoGenerationRequest
from api.pydantic_models.media import VideoJob

from api.pydantic_models.requests import RequestRecord
from api.routes.v1.requests import with_video_status


class VideoRequestStatusTests(unittest.TestCase):
    def record(self, **values):
        return RequestRecord(id='request', time='now', type='Video', status=202,
            latency='0.1 s', latency_ms=100, endpoint='/v1/videos/generations',
            prompt='', output='HTTP 202', request_bytes=1, response_bytes=1, **values)

    def test_accepted_request_tracks_actual_job_without_mutating_history(self):
        record = self.record(video_job_id='job')
        for state in ('Queued', 'Rendering', 'Done', 'Failed', 'Cancelled'):
            with self.subTest(state=state), patch('api.routes.v1.requests.video_jobs.get',
                    return_value=SimpleNamespace(status=state)) as get:
                self.assertEqual(with_video_status(record).video_status, state)
                get.assert_called_once_with('job')
                self.assertIsNone(record.video_status)
                self.assertEqual(record.status, 202)

    def test_missing_job_remains_unknown_not_done(self):
        self.assertIsNone(with_video_status(self.record()).video_status)
        with patch('api.routes.v1.requests.video_jobs.get', side_effect=KeyError):
            self.assertIsNone(with_video_status(self.record(video_job_id='expired')).video_status)


class VideoSubmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_submission_records_exact_job_id_for_telemetry(self):
        job = VideoJob(id='actual-job', prompt='a scene', duration='3s', resolution='768p',
                       aspect='wide', fps='24', progress=0, time='now', status='Queued',
                       thumbnail='', progressText='Queued')
        request = Request({'type': 'http'})
        with patch('api.routes.v1.videos.generations.inference.submit', new=AsyncMock(return_value=job)):
            response = await generate_video(VideoGenerationRequest(prompt='a scene'), request)
        self.assertEqual(response.job.id, 'actual-job')
        self.assertEqual(request.scope['kadan_measurement']['video_job_id'], 'actual-job')
        self.assertEqual(response.job.status, 'Queued')
