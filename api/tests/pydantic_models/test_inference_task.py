import unittest

from pydantic import ValidationError

from api.pydantic_models.inference_task import InferenceTask


class InferenceTaskTests(unittest.TestCase):
    def test_every_feature_round_trips(self):
        inputs = [
            {"feature": "llm", "messages": [{"role": "user", "text": "Hello"}]},
            {"feature": "image", "prompt": "Mountain"},
            {"feature": "decisions", "state": "Example", "questions": []},
            {
                "feature": "tts",
                "script": "Hello",
                "voice": {"mode": "custom", "speaker": "Ryan"},
            },
            {"feature": "stt", "audio": "data:audio/wav;base64,AAAA"},
            {"feature": "video", "prompt": "Mountain"},
        ]
        for value in inputs:
            with self.subTest(feature=value["feature"]):
                task = InferenceTask(input=value)
                self.assertEqual(
                    InferenceTask.model_validate_json(task.model_dump_json()), task
                )

    def test_download_round_trips(self):
        task = InferenceTask(
            operation="download",
            input={
                "feature": "models",
                "model_id": "small",
                "license_acknowledged": True,
            },
        )
        self.assertEqual(
            InferenceTask.model_validate_json(task.model_dump_json()), task
        )
        for operation, body in (
            ("generate", {"feature": "models", "model_id": "small"}),
            ("download", {"feature": "llm"}),
        ):
            with self.assertRaises(ValidationError):
                InferenceTask(operation=operation, input=body)

    def test_context_omitted_and_explicit_null_remain_distinct(self):
        for fields in ({}, {"context_limit": None}, {"context_limit": 4096}):
            task = InferenceTask(operation="load", input={"feature": "llm", **fields})
            restored = InferenceTask.model_validate_json(task.model_dump_json())
            self.assertEqual(
                "context_limit" in restored.input.model_fields_set,
                "context_limit" in fields,
            )

    def test_invalid_feature_operation_and_input_are_rejected(self):
        for value in (
            {"input": {"feature": "unknown"}},
            {
                "input": {
                    "feature": "image",
                    "prompt": "test",
                    "checkpoint": "/tmp/weights",
                }
            },
            {"operation": "load", "input": {"feature": "image", "prompt": "test"}},
            {"operation": "completion", "input": {"feature": "llm"}},
        ):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                InferenceTask.model_validate(value)
