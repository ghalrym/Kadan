import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID

from pydantic import ValidationError

from api.pydantic_models.inference_task import InferenceTask
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


class ImageModelTests(unittest.TestCase):
    def test_omitted_model_defers_to_settings_selection(self):
        request = generations.ImageRequest(prompt="test")
        self.assertIsNone(request.model_id)
        self.assertEqual(request.steps, 50)
        task = InferenceTask(
            input={"feature": "image", **request.model_dump(exclude_unset=True)}
        )
        self.assertNotIn("model_id", task.model_dump(mode="json")["input"])

    def test_turbo_choice_survives_task_serialization(self):
        request = generations.ImageRequest(
            prompt="test", model_id="qwen-image-2.1-turbo"
        )
        task = InferenceTask(input={"feature": "image", **request.model_dump()})
        restored = InferenceTask.model_validate_json(task.model_dump_json())
        self.assertEqual(restored.input.model_id, "qwen-image-2.1-turbo")

    def test_unknown_image_model_is_rejected(self):
        with self.assertRaises(ValidationError):
            generations.ImageRequest(prompt="test", model_id="unknown")
        # The base model was removed; Turbo is the only image model.
        with self.assertRaises(ValidationError):
            generations.ImageRequest(prompt="test", model_id="qwen-image-2.1")
        with self.assertRaises(ValidationError):
            InferenceTask(
                input={"feature": "image", "prompt": "test", "model_id": "unknown"}
            )


class ImageCapacityResponseTests(unittest.TestCase):
    def test_optional_capacity_advice_survives_response_validation(self):
        image = {
            "id": "result",
            "mode": "Generate",
            "prompt": "otter",
            "aspect": "square",
            "seeds": [42],
            "meta": "384×416",
        }
        self.assertIsNone(generations.ImageResponse(image=image).next_batch_capacity)
        for capacity in (1, 5, 32):
            result = generations.ImageResponse(
                image=image, next_batch_capacity=capacity
            )
            self.assertEqual(result.model_dump()["next_batch_capacity"], capacity)
        for capacity in (0, 33):
            with self.assertRaises(ValidationError):
                generations.ImageResponse(image=image, next_batch_capacity=capacity)
