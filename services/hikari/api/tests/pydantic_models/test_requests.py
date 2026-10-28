import unittest
from typing import get_args

from pydantic import TypeAdapter, ValidationError

from api.pydantic_models.inference_task import LlmModel
from api.pydantic_models.requests import RequestRecord


class ObservedModelTests(unittest.TestCase):
    def record(self, model):
        return RequestRecord(
            id="request",
            time="now",
            type="LLM",
            model=model,
            status=200,
            latency="0.1 s",
            latency_ms=100,
            endpoint="/v1/chat/completions",
            prompt="1 message",
            output="HTTP 200",
            request_bytes=1,
            response_bytes=1,
        )

    def test_records_preserve_native_lfm_and_unknown_observations(self):
        for model in (*get_args(LlmModel), "lfm3-1.7b-bf16", None):
            with self.subTest(model=model):
                record = self.record(model)
                restored = RequestRecord.model_validate_json(record.model_dump_json())
                self.assertEqual(restored.model, model)

    def test_private_and_unreleased_model_ids_cannot_enter_a_record(self):
        for model in ("Private prompt used as model", "lfm3-unreleased", ""):
            with self.subTest(model=model), self.assertRaises(ValidationError):
                self.record(model)

    def test_observing_lfm_does_not_admit_it_as_a_native_language_checkpoint(self):
        with self.assertRaises(ValidationError):
            TypeAdapter(LlmModel).validate_python("lfm3-1.7b-bf16")
