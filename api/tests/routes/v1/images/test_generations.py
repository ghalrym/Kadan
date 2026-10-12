import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID

from api.routes.v1.images import generations


class ImageRequestIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_header_uuid_is_normalized_to_exact_native_job_id(self):
        body = generations.ImageRequest(prompt="test")
        request = object()
        result = {"image": {}}

        async def infer(http_request, call):
            self.assertIs(http_request, request)
            return await call

        with (
            patch.object(generations, "infer", side_effect=infer),
            patch.object(
                generations.inference, "submit", new=AsyncMock(return_value=result)
            ) as submit,
        ):
            response = await generations.create_image(
                body, request, UUID("b2a9163e-90bf-4a7d-b68d-887ddbd71a7e")
            )
        submit.assert_awaited_once_with(
            body,
            feature="image",
            operation="generate",
            job_id="b2a9163e90bf4a7db68d887ddbd71a7e",
        )
        self.assertEqual(response, result)
