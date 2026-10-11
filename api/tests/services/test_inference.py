import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from api.routes.model_lifecycle import ModelLoadRequest
from api.services.inference_requests import PreparedRequest
from api.pydantic_models.inference import EmptyPayload
from api.services.inference import InferenceService
from api.services.native_worker import NativeJob, NativeRequest


class InferenceBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_workspace_failure_keeps_id_until_native_terminal(self):
        service = InferenceService()
        job = NativeJob(NativeRequest(feature='image', operation='generate', model='image'), False)
        job.preparing.set()
        service.worker.jobs[job.request.id] = job
        joining, terminal = asyncio.Event(), asyncio.Event()
        async def join(_):
            joining.set()
            await terminal.wait()
            job.result.set_result(None)
        with patch('api.services.inference.tempfile.mkdtemp', side_effect=OSError('disk full')), \
             patch.object(service.worker, 'cancel_and_join', side_effect=join):
            task = asyncio.create_task(service._complete(job, Mock()))
            await joining.wait()
            self.assertIn(job.request.id, service.worker.jobs)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertIn(job.request.id, service.worker.jobs)
            terminal.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertNotIn(job.request.id, service.worker.jobs)

    async def test_preparation_waits_for_native_head_notification(self):
        service = InferenceService()
        job = NativeJob(NativeRequest(feature='image', operation='generate', model='image'), False)
        with tempfile.TemporaryDirectory() as folder, \
             patch('api.services.inference.run_cancellable_thread', new=AsyncMock()) as prepare:
            service.worker.root = Path(folder)
            task = asyncio.create_task(service._complete(job, Mock()))
            await asyncio.sleep(0)
            prepare.assert_not_awaited()
            job.result.set_result({'cancelled': True})
            self.assertEqual(await task, {'cancelled': True})
            prepare.assert_not_awaited()

    async def test_successful_load_persists_selected_model(self):
        service = InferenceService()
        job = NativeJob(NativeRequest(feature='llm', operation='load', model='small'), False)
        job.preparing.set()
        async def prepared(*args):
            job.result.set_result({})
        with tempfile.TemporaryDirectory() as folder, \
             patch('api.services.inference.model_manager') as manager, \
             patch('api.services.inference.run_cancellable_thread', new=AsyncMock(return_value=PreparedRequest(
                 payload=EmptyPayload(), checkpoint_id='small', context_limit=8192, supported_context=32768))), \
             patch.object(service.worker, 'prepared', side_effect=prepared):
            service.worker.root = Path(folder)
            await service._complete(job, ModelLoadRequest(model_id='small'))
            manager.select.assert_called_once_with('small')

    async def test_cancellation_joins_presentation_before_removing_workspace(self):
        service = InferenceService()
        job = NativeJob(NativeRequest(feature='image', operation='generate', model='image'), False)
        job.preparing.set()
        entered, release = asyncio.Event(), asyncio.Event()
        async def prepared(*args):
            job.result.set_result({})
        async def present(*args):
            entered.set()
            await release.wait()
            return {}
        with tempfile.TemporaryDirectory() as folder, \
             patch('api.services.inference.run_cancellable_thread', new=AsyncMock(return_value=PreparedRequest(
                 payload=EmptyPayload(), checkpoint_id='image'))), \
             patch.object(service.worker, 'prepared', side_effect=prepared), \
             patch('api.services.inference.asyncio.to_thread', side_effect=present):
            service.worker.root = Path(folder)
            task = asyncio.create_task(service._complete(job, Mock()))
            await entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertTrue(list(Path(folder).iterdir()))
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(list(Path(folder).iterdir()))

    async def test_immediate_cancel_releases_native_job_before_task_first_step(self):
        service = InferenceService()
        job = NativeJob(NativeRequest(feature='image', operation='generate', model='image'), False)
        service.worker.jobs[job.request.id] = job
        task = asyncio.create_task(service._complete(job, Mock()))
        service.tasks[job.request.id] = task
        async def join(_):
            job.result.set_result(None)
        with patch.object(service.worker, 'cancel_and_join', side_effect=join) as cleanup:
            await service.cancel(job.request.id)
        cleanup.assert_awaited_once_with(job)
        self.assertTrue(task.cancelled())
        self.assertNotIn(job.request.id, service.worker.jobs)

    async def test_close_tolerates_job_completing_during_another_cancellation(self):
        service = InferenceService()
        head = NativeJob(NativeRequest(feature='image', operation='generate', model='image'), False)
        tail = NativeJob(NativeRequest(feature='image', operation='generate', model='image'), False)
        service.worker.jobs.update({head.request.id: head, tail.request.id: tail})
        async def join(job):
            tail.result.set_result(None)
            service.worker.jobs.pop(tail.request.id)
            job.result.set_result(None)
        with patch.object(service.worker, 'cancel_and_join', side_effect=join), \
             patch.object(service.worker, 'close', new=AsyncMock()) as close:
            await service.close()
        close.assert_awaited_once()
        self.assertFalse(service.worker.jobs)
