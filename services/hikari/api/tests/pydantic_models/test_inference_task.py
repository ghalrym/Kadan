import unittest

from pydantic import ValidationError

from api.pydantic_models.inference_task import (
    InferenceTask,
    TranscriptionFormattingTask,
    resolve_inkling_variant_model,
    same_inkling_precision_family,
)


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

    def test_weight_variant_round_trips_and_resolves_precision_family(self):
        chat_task = InferenceTask(
            operation="completion",
            input={
                "feature": "llm",
                "messages": [{"role": "user", "text": "Hello"}],
                "weight_variant": "base",
            },
        )
        restored = InferenceTask.model_validate_json(chat_task.model_dump_json())
        self.assertEqual(restored.input.weight_variant, "base")

        dec_task = InferenceTask(
            input={
                "feature": "decisions",
                "state": "Example",
                "questions": [],
                "weight_variant": "abliterated",
            }
        )
        self.assertEqual(
            InferenceTask.model_validate_json(
                dec_task.model_dump_json()
            ).input.weight_variant,
            "abliterated",
        )

        with self.assertRaises(ValidationError):
            InferenceTask(
                operation="completion",
                input={
                    "feature": "llm",
                    "messages": [{"role": "user", "text": "Hello"}],
                    "model": "inkling-2-mini-abliterated-iq4-nl",
                    "weight_variant": "base",
                },
            )

        self.assertEqual(
            resolve_inkling_variant_model("inkling-2-mini-abliterated-iq4-nl", "base"),
            "inkling-2-mini-iq4-nl",
        )
        self.assertEqual(
            resolve_inkling_variant_model("inkling-2-mini-q8-0", "abliterated"),
            "inkling-2-mini-abliterated-q8-0",
        )
        self.assertTrue(
            same_inkling_precision_family(
                "inkling-2-mini-iq4-nl", "inkling-2-mini-abliterated-iq4-nl"
            )
        )
        self.assertFalse(
            same_inkling_precision_family(
                "inkling-2-mini-iq4-nl", "inkling-2-mini-q8-0"
            )
        )


class NativeTranscriptionFormattingLimitTests(unittest.TestCase):
    def test_native_transcript_formatting_keeps_its_32000_character_limit(self):
        self.assertEqual(len(TranscriptionFormattingTask(text="x" * 32000).text), 32000)
        with self.assertRaises(ValidationError):
            TranscriptionFormattingTask(text="x" * 32001)
