import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID

from pydantic import ValidationError

from api.pydantic_models.inference_task import InferenceTask
from api.routes.v1.images import edits


class ImageEditTests(unittest.IsolatedAsyncioTestCase):
    async def test_edit_uses_shared_queue_and_normalized_request_identity(self):
        body = edits.ImageRequest(
            prompt="A blue sky",
            image="data:image/png;base64,cG5n",
            mask="data:image/png;base64,bWFzaw==",
            strength=0.4,
            count=1,
            batch_collection_target=3,
        )
        request = object()
        result = {"image": {"mode": "Edit"}}

        async def infer(http_request, call):
            self.assertIs(http_request, request)
            return await call

        with (
            patch.object(edits, "infer", side_effect=infer),
            patch.object(
                edits.inference, "submit", new=AsyncMock(return_value=result)
            ) as submit,
        ):
            response = await edits.create_image(body, request, UUID("a" * 32))
        submit.assert_awaited_once_with(
            body, feature="image", operation="generate", job_id="a" * 32
        )
        self.assertEqual(response, result)
        task = InferenceTask(input={"feature": "image", **body.model_dump()})
        restored = InferenceTask.model_validate_json(task.model_dump_json())
        self.assertEqual(restored.input.image, body.image)
        self.assertEqual(restored.input.mask, body.mask)
        self.assertEqual(restored.input.strength, 0.4)
        self.assertEqual(restored.input.batch_collection_target, 3)

    def test_rejects_non_upload_references_and_invalid_strength(self):
        for image in (
            "https://example.com/source.png",
            "/tmp/source.png",
            "data:image/jpeg;base64,cG5n",
        ):
            with self.assertRaises(ValidationError):
                edits.ImageRequest(prompt="sky", image=image)
        for strength in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValidationError):
                edits.ImageRequest(
                    prompt="sky", image="data:image/png;base64,cG5n", strength=strength
                )
        with self.assertRaises(ValidationError):
            InferenceTask(input={"feature": "image", "prompt": "sky", "mask": "mask"})

    def test_zero_strength_and_settings_selection_survive_serialization(self):
        body = edits.ImageRequest(
            prompt="sky", image="data:image/png;base64,cG5n", strength=0
        )
        task = InferenceTask(
            input={"feature": "image", **body.model_dump(exclude_unset=True)}
        )
        self.assertEqual(task.input.strength, 0)
        self.assertNotIn("model_id", task.model_dump(mode="json")["input"])

    def test_reference_dimensions_survive_task_serialization(self):
        body = edits.ImageRequest(
            prompt="The same fox jumping",
            image="data:image/png;base64,cG5n",
            mode="reference",
            width=1920,
            height=416,
            count=1,
        )
        task = InferenceTask(
            input={"feature": "image", **body.model_dump(exclude_unset=True)}
        )
        restored = InferenceTask.model_validate_json(task.model_dump_json())
        self.assertEqual(restored.input.mode, "reference")
        self.assertEqual((restored.input.width, restored.input.height), (1920, 416))
        self.assertNotIn("strength", restored.model_dump(mode="json")["input"])

    def test_reference_rejects_source_edit_controls_and_invalid_target(self):
        for values in (
            {"mask": "data:image/png;base64,cG5n"},
            {"strength": 0.65},
            {"width": 384},
            {"width": 0, "height": 416},
            {"width": 385, "height": 416},
            {"width": 3072, "height": 3072},
        ):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                edits.ImageRequest(
                    prompt="fox",
                    image="data:image/png;base64,cG5n",
                    mode="reference",
                    **values,
                )
        with self.assertRaises(ValidationError):
            edits.ImageRequest(
                prompt="fox", image="data:image/png;base64,cG5n", width=384, height=416
            )
        with self.assertRaises(ValidationError):
            InferenceTask(
                input={"feature": "image", "prompt": "fox", "mode": "reference"}
            )

    def test_reference_defaults_to_source_dimensions(self):
        body = edits.ImageRequest(
            prompt="fox", image="data:image/png;base64,cG5n", mode="reference"
        )
        self.assertIsNone(body.width)
        self.assertIsNone(body.height)


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
        self.assertIsNone(edits.ImageResponse(image=image).next_batch_capacity)
        for capacity in (1, 5, 32):
            result = edits.ImageResponse(image=image, next_batch_capacity=capacity)
            self.assertEqual(result.model_dump()["next_batch_capacity"], capacity)
        for capacity in (0, 33):
            with self.assertRaises(ValidationError):
                edits.ImageResponse(image=image, next_batch_capacity=capacity)
