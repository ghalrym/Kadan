"""Pure request/template boundary checks, without loading or running inference."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from pydantic import ValidationError

from api.pydantic_models.inference_task import InferenceTask
from api.routes.v1.chat.completions import CompletionRequest
from experimental_worker.text import template_inputs


class Lfm3ToolPromptTests(unittest.TestCase):
    def body(self):
        return {
            "model": "lfm3-1.7b-bf16",
            "messages": [
                {"role": "system", "content": "Owned agent instructions"},
                {"role": "user", "content": "Read the file"},
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "call_0", "type": "function",
                    "function": {"name": "read_file", "arguments": {"path": "/tmp/a", "lines": 3}},
                }]},
                {"role": "tool", "content": "file contents", "tool_call_id": "call_0"},
            ],
            "tools": [{"type": "function", "function": {
                "name": "read_file", "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            }}],
        }

    def test_typed_request_and_worker_queue_preserve_native_template_fields(self):
        body = CompletionRequest.model_validate(self.body())
        request = InferenceTask(operation="completion", input={
            "feature": "llm", **body.model_dump(exclude_unset=True),
        })
        task = request.model_dump(mode="json")["input"]
        tokenizer = SimpleNamespace(apply_chat_template=Mock(return_value="formatted"))
        self.assertEqual(template_inputs(tokenizer, task), "formatted")
        conversation = tokenizer.apply_chat_template.call_args.args[0]
        options = tokenizer.apply_chat_template.call_args.kwargs
        self.assertEqual(conversation[0], {"role": "system", "content": "Owned agent instructions"})
        self.assertEqual(conversation[-1], {"role": "tool", "content": "file contents", "tool_call_id": "call_0"})
        self.assertEqual(conversation[-2]["tool_calls"][0]["function"]["arguments"], {"path": "/tmp/a", "lines": 3})
        self.assertEqual(options["tools"][0]["function"]["name"], "read_file")
        self.assertFalse(options["enable_thinking"])

    def test_rich_tool_history_is_rejected_for_native_model(self):
        body = self.body()
        body["model"] = "native-model"
        with self.assertRaises(ValidationError):
            CompletionRequest.model_validate(body)
        with self.assertRaises(ValidationError):
            InferenceTask(input={"feature": "llm", **body})

    def test_json_encoded_arguments_cannot_reach_lfm3_template(self):
        body = self.body()
        body["messages"][2]["tool_calls"][0]["function"]["arguments"] = '{"path":"/tmp/a"}'
        with self.assertRaises(ValidationError):
            CompletionRequest.model_validate(body)

    def test_tool_results_keep_their_call_identity(self):
        body = self.body()
        del body["messages"][-1]["tool_call_id"]
        with self.assertRaises(ValidationError):
            CompletionRequest.model_validate(body)
