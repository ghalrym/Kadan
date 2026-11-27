import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID

from pydantic import ValidationError

from api.pydantic_models.inference_task import InferenceTask
from api.routes.v1.images import edits, generations


class ImageRequestIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_header_uuid_is_normalized_to_exact_native_job_id(self):
        body = generations.ImageRequest(prompt="test", batch_collection_target=3)
        request = object()
        result = {"image": {}}

        async def infer(http_request, call):
            self.assertIs(http_request, request)
            return await call

        with (
            patch.object(generations, "infer", side_effect=infer),
            patch(
                "api.services.subscription_images.subscriptions.active",
                return_value=None,
            ),
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
        self.assertEqual(request.steps, 4)
        task = InferenceTask(
            input={"feature": "image", **request.model_dump(exclude_unset=True)}
        )
        self.assertNotIn("model_id", task.model_dump(mode="json")["input"])

    def test_klein_choice_survives_task_serialization(self):
        request = generations.ImageRequest(
            prompt="test", model_id="flux.2-klein-4b"
        )
        task = InferenceTask(input={"feature": "image", **request.model_dump()})
        restored = InferenceTask.model_validate_json(task.model_dump_json())
        self.assertEqual(restored.input.model_id, "flux.2-klein-4b")

    def test_unknown_image_model_is_rejected(self):
        with self.assertRaises(ValidationError):
            generations.ImageRequest(prompt="test", model_id="unknown")
        # Qwen-Image 2.1 was removed; FLUX.2 Klein 4B is the only image model.
        with self.assertRaises(ValidationError):
            generations.ImageRequest(prompt="test", model_id="qwen-image-2.1-turbo")
        with self.assertRaises(ValidationError):
            InferenceTask(
                input={"feature": "image", "prompt": "test", "model_id": "unknown"}
            )

    def test_reference_images_validation_and_serialization(self):
        png = "data:image/png;base64,cG5n"
        request = edits.ImageRequest(
            prompt="pose transfer",
            image=png,
            reference_images=[png],
            mode="reference",
            width=1536,
            height=512,
        )
        task = InferenceTask(
            input={"feature": "image", **request.model_dump(exclude_unset=True)}
        )
        restored = InferenceTask.model_validate_json(task.model_dump_json())
        self.assertEqual(restored.input.reference_images, [png])

        with self.assertRaises(ValidationError):
            edits.ImageRequest(
                prompt="source edit",
                image=png,
                reference_images=[png],
                mode="source",
            )
        with self.assertRaises(ValidationError):
            edits.ImageRequest(
                prompt="too many refs",
                image=png,
                reference_images=[png, png, png, png],
                mode="reference",
            )
        with self.assertRaises(ValidationError):
            edits.ImageRequest(
                prompt="empty refs",
                image=png,
                reference_images=[],
                mode="reference",
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


class ImageCollectionTargetTests(unittest.TestCase):
    def test_optional_target_validates_and_survives_worker_task_serialization(self):
        for model, values in (
            (generations.ImageRequest, {"prompt": "fox"}),
            (
                edits.ImageRequest,
                {"prompt": "fox", "image": "data:image/png;base64,cG5n"},
            ),
        ):
            with self.subTest(model=model):
                body = model(**values)
                self.assertIsNone(body.batch_collection_target)
                task = InferenceTask(
                    input={"feature": "image", **body.model_dump(exclude_unset=True)}
                )
                self.assertNotIn(
                    "batch_collection_target", task.model_dump(mode="json")["input"]
                )
                for target in (None, 1, 3, 32):
                    body = model(**values, batch_collection_target=target)
                    task = InferenceTask(
                        input={
                            "feature": "image",
                            **body.model_dump(exclude_unset=True),
                        }
                    )
                    restored = InferenceTask.model_validate_json(task.model_dump_json())
                    self.assertEqual(restored.input.batch_collection_target, target)
                for invalid in (0, -1, 33, 1.5, True, "2"):
                    with (
                        self.subTest(invalid=invalid),
                        self.assertRaises(ValidationError),
                    ):
                        model(**values, batch_collection_target=invalid)
                    with self.assertRaises(ValidationError):
                        InferenceTask(
                            input={
                                "feature": "image",
                                **values,
                                "batch_collection_target": invalid,
                            }
                        )
