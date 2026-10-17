import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import Request

from api.routes.v1.chat.completions import CompletionRequest, chunks, create_completion
from api.services.inference_errors import InferenceFailure


class CompletionRequestTests(unittest.TestCase):
    def test_text_and_content_inputs_preserve_supported_message_fields(self):
        for field in ("text", "content"):
            with self.subTest(field=field):
                body = CompletionRequest(
                    messages=[{"role": "user", field: "Hello", "meta": "context"}]
                )
                self.assertEqual(
                    body.model_dump()["messages"],
                    [{"role": "user", "text": "Hello", "meta": "context"}],
                )

    def test_unsupported_message_fields_are_rejected_instead_of_discarded(self):
        for field, value in (
            (
                "tool_calls",
                [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": "{}"},
                    }
                ],
            ),
            ("function_call", {"name": "weather", "arguments": "{}"}),
            ("tool_call_id", "call_1"),
            ("name", "weather"),
            ("refusal", "I cannot help."),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValidationError) as error:
                    CompletionRequest(
                        messages=[
                            {"role": "assistant", "content": "Hello", field: value}
                        ]
                    )
                self.assertEqual(error.exception.errors()[0]["type"], "extra_forbidden")
                self.assertEqual(
                    error.exception.errors()[0]["loc"], ("messages", 0, field)
                )

    def test_ambiguous_text_and_content_are_rejected(self):
        with self.assertRaises(ValidationError):
            CompletionRequest(
                messages=[
                    {"role": "user", "text": "Hello", "content": "Different prompt"}
                ]
            )

    def test_unsupported_top_level_controls_are_rejected(self):
        for field, value in (
            ("tools", []),
            ("tool_choice", "auto"),
            ("reasoning_effort", "high"),
            ("stream_options", {"include_usage": True}),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValidationError) as error:
                    CompletionRequest(
                        messages=[{"role": "user", "content": "Hello"}],
                        **{field: value},
                    )
                self.assertEqual(error.exception.errors()[0]["type"], "extra_forbidden")


class CompletionResponseTests(unittest.IsolatedAsyncioTestCase):
    async def receive(self):
        await asyncio.Future()

    async def test_json_preserves_empty_and_nonempty_worker_answers(self):
        for text in ("", "Hello 🌍"):
            for reason in ("stop", "length"):
                with self.subTest(text=text, reason=reason):
                    request = Request({"type": "http"}, receive=self.receive)
                    body = CompletionRequest(
                        messages=[{"role": "user", "content": "Hello"}]
                    )
                    result = {"model": "small", "text": text, "finish_reason": reason}
                    with patch(
                        "api.routes.v1.chat.completions.inference.submit",
                        new=AsyncMock(return_value=result),
                    ) as submit:
                        response = (await create_completion(body, request)).model_dump()
                    submit.assert_awaited_once_with(
                        body, feature="llm", operation="completion"
                    )
                    self.assertEqual(
                        response["message"],
                        {"role": "assistant", "text": text, "meta": None},
                    )
                    self.assertEqual(
                        response["choices"],
                        [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": reason,
                            }
                        ],
                    )
                    self.assertEqual(response["model"], "small")
                    self.assertEqual(response["object"], "chat.completion")
                    self.assertTrue(response["id"].startswith("chatcmpl-"))

    async def test_empty_input_messages_remain_invalid(self):
        with self.assertRaises(ValidationError):
            CompletionRequest(messages=[{"role": "user", "content": ""}])

    async def test_pre_stream_inference_failure_retains_http_status_and_detail(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                request = Request({"type": "http"}, receive=self.receive)
                body = CompletionRequest(
                    messages=[{"role": "user", "content": "Hello"}], stream=streaming
                )
                method = "open_chat_stream" if streaming else "submit"
                with (
                    patch(
                        "api.routes.v1.chat.completions.inference." + method,
                        new=AsyncMock(
                            side_effect=InferenceFailure("Worker unavailable", 503)
                        ),
                    ),
                    self.assertRaises(HTTPException) as error,
                ):
                    await create_completion(body, request)
                self.assertEqual(error.exception.status_code, 503)
                self.assertEqual(error.exception.detail, "Worker unavailable")

    async def test_stream_preserves_text_identity_and_terminal_reason(self):
        for text in ("", "Hello 🌍"):
            for reason in ("stop", "length"):
                with self.subTest(text=text, reason=reason):

                    async def stream():
                        if text:
                            yield {"content": "Hello "}
                            yield {"content": "🌍"}
                        yield {"finish_reason": reason}

                    identity = {"id": "chatcmpl-unit", "created": 1, "model": "small"}
                    frames = [frame async for frame in chunks(stream(), identity)]
                    self.assertEqual(frames[-1], "data: [DONE]\n\n")
                    events = [
                        json.loads(frame.removeprefix("data: "))
                        for frame in frames[:-1]
                    ]
                    for event in events:
                        self.assertEqual(
                            {key: event[key] for key in identity}, identity
                        )
                        self.assertEqual(event["object"], "chat.completion.chunk")
                    self.assertEqual(
                        events[0]["choices"][0]["delta"],
                        {"role": "assistant", "content": ""},
                    )
                    self.assertEqual(
                        "".join(
                            event["choices"][0]["delta"].get("content", "")
                            for event in events
                        ),
                        text,
                    )
                    self.assertEqual(
                        events[-1]["choices"],
                        [{"index": 0, "delta": {}, "finish_reason": reason}],
                    )
                    self.assertTrue(
                        all(
                            event["choices"][0]["finish_reason"] is None
                            for event in events[:-1]
                        )
                    )

    async def test_incomplete_or_failed_stream_does_not_claim_success(self):
        for failure in (None, InferenceFailure("Worker failed", 502)):
            with self.subTest(failure=failure):

                async def stream():
                    yield {"content": "Partial"}
                    if failure:
                        raise failure

                frames = [
                    frame
                    async for frame in chunks(
                        stream(),
                        {"id": "chatcmpl-unit", "created": 1, "model": "small"},
                    )
                ]
                self.assertEqual(frames[-1], "data: [DONE]\n\n")
                events = [
                    json.loads(frame.removeprefix("data: ")) for frame in frames[:-1]
                ]
                self.assertIn("error", events[-1])
                self.assertEqual(events[-1]["error"]["code"], 502)
                self.assertTrue(
                    all(
                        event["choices"][0]["finish_reason"] is None
                        for event in events[:-1]
                    )
                )
