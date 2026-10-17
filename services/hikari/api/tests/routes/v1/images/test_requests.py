import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi import HTTPException

from api.routes.v1.images import requests


class ImageCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_acknowledgement_waits_for_cleanup(self):
        request_id = uuid4()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def cancel(job_id):
            self.assertEqual(job_id, request_id.hex)
            entered.set()
            await release.wait()

        service = SimpleNamespace(
            worker=SimpleNamespace(
                jobs={
                    request_id.hex: SimpleNamespace(
                        request=SimpleNamespace(feature="image")
                    )
                }
            ),
            cancel=cancel,
        )
        with patch.object(requests, "inference", service):
            call = asyncio.create_task(requests.cancel_image_request(request_id))
            await entered.wait()
            self.assertFalse(call.done())
            release.set()
            self.assertEqual((await call).status, "cancelled")

    async def test_absent_request_is_not_reported_as_cancelled(self):
        service = SimpleNamespace(worker=SimpleNamespace(jobs={}), cancel=AsyncMock())
        with patch.object(requests, "inference", service):
            result = await requests.cancel_image_request(uuid4())
        self.assertEqual(result.status, "not_active")
        service.cancel.assert_not_awaited()

    async def test_other_modalities_cannot_be_cancelled_through_image_route(self):
        request_id = uuid4()
        service = SimpleNamespace(
            worker=SimpleNamespace(
                jobs={
                    request_id.hex: SimpleNamespace(
                        request=SimpleNamespace(feature="llm")
                    )
                }
            ),
            cancel=AsyncMock(),
        )
        with patch.object(requests, "inference", service):
            with self.assertRaises(HTTPException) as raised:
                await requests.cancel_image_request(request_id)
        self.assertEqual(raised.exception.status_code, 409)
        service.cancel.assert_not_awaited()
