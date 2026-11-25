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
                    [
                        {
                            "role": "user",
                            "text": "Hello",
                            "meta": "context",
                            "images": None,
                            "tool_calls": None,
                            "tool_call_id": None,
                            "name": None,
                            "reasoning_content": None,
                        }
                    ],
                )

    def test_unsupported_message_fields_are_rejected_instead_of_discarded(self):
        for field, value in (
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
                self.assertTrue(error.exception.errors())

    def test_json_response_tool_history_can_be_submitted_again(self):
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "weather", "arguments": "{}"},
        }
        request = CompletionRequest(
            messages=[{"role": "assistant", "content": "", "tool_calls": [call]}]
        )
        self.assertEqual(request.messages[0].tool_calls[0].function.arguments, {})

    def test_image_keeps_its_place_in_multipart_message(self):
        from unittest.mock import patch

        from api.pydantic_models.chat import ChatImagePart

        with patch(
            "api.pydantic_models.chat.normalize_image_data_url",
            side_effect=lambda url: url,
        ):
            request = CompletionRequest(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Before "},
                            {"type": "image_url", "image_url": {"url": "image"}},
                            {"type": "text", "text": " After"},
                        ],
                    }
                ]
            )
            parts = request.messages[0].ordered_content()
            self.assertEqual(parts[0].text, "Before ")
            self.assertIsInstance(parts[1], ChatImagePart)
            self.assertEqual(parts[2].text, " After")
            self.assertEqual(
                request.model_dump()["messages"][0]["content"][1]["image_url"]["url"],
                "image",
            )

    def test_ambiguous_text_and_content_are_rejected(self):
        with self.assertRaises(ValidationError):
            CompletionRequest(
                messages=[
                    {"role": "user", "text": "Hello", "content": "Different prompt"}
                ]
            )

    def test_reasoning_effort_accepts_valid_levels_and_rejects_unknown_controls(self):
        for level in ("low", "medium", "high"):
            body = CompletionRequest(
                messages=[{"role": "user", "content": "Hello"}],
                reasoning_effort=level,
            )
            self.assertEqual(body.reasoning_effort, level)
        for field, value in (
            ("tool_choice", "auto"),
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
    def setUp(self):
        local = patch(
            "api.routes.v1.chat.completions.subscriptions.active", return_value=None
        )
        local.start()
        self.addCleanup(local.stop)

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

    async def test_native_and_subscription_tool_results_keep_the_same_shape(self):
        from api.routes.v1.chat.completions import (
            completion_response,
            subscription_result,
        )

        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "example", "arguments": '{"text":"</parameter>"}'},
        }

        async def stream():
            yield {"content": " keep "}
            yield {"tool_calls": [call]}
            yield {"finish_reason": "tool_calls"}

        identity = {"id": "unit", "created": 1, "model": "unit"}
        response = completion_response(
            identity, await subscription_result(stream())
        ).model_dump()
        self.assertEqual(response["choices"][0]["message"]["content"], " keep ")
        self.assertEqual(response["message"]["tool_calls"], [call])
        self.assertEqual(response["choices"][0]["message"]["tool_calls"], [call])
        frames = [frame async for frame in chunks(stream(), identity)]
        delta = json.loads(frames[2][6:])["choices"][0]["delta"]
        self.assertEqual(delta, {"tool_calls": [{**call, "index": 0}]})
        self.assertEqual(
            json.loads(frames[-2][6:])["choices"][0]["finish_reason"], "tool_calls"
        )

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

                    async def stream(text, reason):
                        if text:
                            yield {"content": "Hello "}
                            yield {"content": "🌍"}
                        yield {"finish_reason": reason}

                    identity = {"id": "chatcmpl-unit", "created": 1, "model": "small"}
                    frames = [
                        frame async for frame in chunks(stream(text, reason), identity)
                    ]
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

                async def stream(failure):
                    yield {"content": "Partial"}
                    if failure:
                        raise failure

                frames = [
                    frame
                    async for frame in chunks(
                        stream(failure),
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


class SubscriptionCompletionTests(unittest.IsolatedAsyncioTestCase):
    async def receive(self):
        await asyncio.Future()

    async def test_selected_subscription_replaces_the_local_model(self):
        from api.pydantic_models.subscriptions import ActiveSubscription

        class Stream:
            model = "claude-haiku-5-5"

            def __init__(self, subscription, body):
                self.subscription = subscription

            async def __aiter__(self):
                yield {"content": "  <function=finish>  "}
                yield {"finish_reason": "stop"}

        request = Request({"type": "http"}, receive=self.receive)
        body = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
        with (
            patch(
                "api.routes.v1.chat.completions.subscriptions.active",
                return_value=ActiveSubscription(provider="claude", credential="t"),
            ),
            patch("api.routes.v1.chat.completions.subscription_stream", Stream),
            patch("api.routes.v1.chat.completions.inference.submit") as submit,
        ):
            response = (await create_completion(body, request)).model_dump()
        submit.assert_not_called()
        self.assertEqual(response["model"], "claude-haiku-5-5")
        self.assertEqual(
            response["choices"][0]["message"]["content"], "  <function=finish>  "
        )
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")

    async def test_unconnected_selection_is_a_conflict(self):
        from api.repositories.subscriptions import SubscriptionError

        request = Request({"type": "http"}, receive=self.receive)
        body = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
        with (
            patch(
                "api.routes.v1.chat.completions.subscriptions.active",
                side_effect=SubscriptionError("not connected"),
            ),
            self.assertRaises(HTTPException) as error,
        ):
            await create_completion(body, request)
        self.assertEqual(error.exception.status_code, 409)
