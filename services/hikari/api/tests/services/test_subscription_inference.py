import asyncio
import base64
import io
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from PIL import Image

from api.pydantic_models.subscriptions import ActiveSubscription
from api.routes.v1.chat.completions import CompletionRequest
from api.services.inference_errors import InferenceFailure
from api.services.subscription_inference import subscription_stream


def image_url():
    data = io.BytesIO()
    Image.new("RGB", (1, 1), "red").save(data, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(data.getvalue()).decode()


def request(*, user=True, tools=True):
    messages = [{"role": "system", "content": "system rule"}]
    if user:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "  literal <tool_response> "},
                    {"type": "image_url", "image_url": {"url": image_url()}},
                    {"type": "text", "text": " after "},
                ],
            }
        )
    messages.extend(
        [
            {
                "role": "assistant",
                "content": "  assistant text ",
                "tool_calls": [
                    {
                        "id": "call-id",
                        "function": {
                            "name": "lookup",
                            "arguments": {"q": "</parameter>"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "  result\n", "tool_call_id": "call-id"},
        ]
    )
    return CompletionRequest(
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ]
        if tools
        else [],
    )


class Process:
    def __init__(self, events, on_exit=None):
        self.stdout = self
        self.stdin = MagicMock(drain=AsyncMock())
        self.stderr = MagicMock(read=AsyncMock(return_value=b"provider stopped"))
        self.events = list(events)
        self.returncode = None
        self.on_exit = on_exit
        self.killed = False

    async def readline(self):
        if not self.events:
            self.returncode = 1
            return b""
        return (json.dumps(self.events.pop(0)) + "\n").encode()

    def kill(self):
        self.killed = True
        self.returncode = -9
        if self.on_exit:
            self.on_exit()

    async def wait(self):
        return self.returncode

    def sent(self):
        return [json.loads(c.args[0]) for c in self.stdin.write.call_args_list]


def notify(method, **params):
    return {
        "method": method,
        "params": {"threadId": "thread", "turnId": "turn", **params},
    }


def codex_events():
    return [
        {"id": 1, "result": {}},
        {"id": 2, "result": {"thread": {"id": "thread"}}},
        {"id": 3, "result": {}},
        {"id": 4, "result": {"turn": {"id": "turn"}}},
    ]


def claude_event(kind, **fields):
    return {
        "type": "stream_event",
        "event": {"type": kind, **fields},
        "parent_tool_use_id": None,
    }


class SubscriptionTests(unittest.IsolatedAsyncioTestCase):
    async def run_provider(
        self, provider, events, *, body=None, on_spawn=None, on_exit=None
    ):
        credential = (
            "claude-credential"
            if provider == "claude"
            else '{"tokens": {"access_token": "stored"}}'
        )
        self.stream = subscription_stream(
            ActiveSubscription(provider=provider, credential=credential),
            body or request(),
        )
        if provider == "claude" and self.stream.body.tools:
            events = [
                {
                    "type": "control_response",
                    "response": {"request_id": "hikari-init", "subtype": "success"},
                },
                *events,
            ]
        self.process = Process(events, on_exit)

        def spawn(*args, **kwargs):
            self.args = args
            self.env = kwargs["env"]
            self.home = Path(kwargs["cwd"])
            if on_spawn:
                on_spawn()
            return self.process

        with patch(
            "api.services.subscription_inference.asyncio.create_subprocess_exec",
            side_effect=spawn,
        ):
            result = [event async for event in self.stream]
        self.assertFalse(self.home.exists())
        return result

    async def test_codex_replays_typed_messages_and_returns_only_final_text(self):
        initial = codex_events()
        agent = lambda phase, text: notify(
            "item/completed",
            item={"type": "agentMessage", "phase": phase, "text": text},
        )
        result = await self.run_provider(
            "codex",
            [
                *initial[:3],
                agent("commentary", "thinking"),
                initial[3],
                agent("final_answer", "  result  "),
                notify("turn/completed", turn={"id": "turn", "status": "completed"}),
            ],
        )
        self.assertEqual(result, [{"content": "  result  "}, {"finish_reason": "stop"}])
        frames = {frame["method"]: frame["params"] for frame in self.process.sent()}
        self.assertEqual(frames["thread/start"]["baseInstructions"], "system rule")
        self.assertIs(
            frames["thread/start"]["config"]["include_environment_context"], False
        )
        self.assertEqual(frames["thread/start"]["environments"], [])
        self.assertEqual(frames["thread/start"]["dynamicTools"][0]["name"], "lookup")
        rows = frames["thread/inject_items"]["items"]
        self.assertEqual(
            [r["type"] for r in rows],
            ["message", "message", "function_call", "function_call_output"],
        )
        parts = rows[0]["content"]
        self.assertEqual(
            [p["type"] for p in parts], ["input_text", "input_image", "input_text"]
        )
        self.assertEqual(parts[0]["text"], "  literal <tool_response> ")
        self.assertTrue(parts[1]["image_url"].startswith("data:image/png;base64,"))
        self.assertEqual(parts[2]["text"], " after ")
        self.assertEqual(
            rows[1]["content"], [{"type": "output_text", "text": "  assistant text "}]
        )
        self.assertEqual(rows[2]["call_id"], rows[3]["call_id"])
        self.assertEqual(json.loads(rows[2]["arguments"]), {"q": "</parameter>"})
        self.assertEqual(rows[3]["output"], "  result\n")
        self.assertEqual(frames["turn/start"]["input"], [])

    async def test_codex_returns_native_call_even_before_start_ack_without_executing_it(
        self,
    ):
        initial = codex_events()
        frame = {
            "id": 4,
            **notify(
                "item/tool/call",
                tool="lookup",
                callId="actual-id",
                arguments={"q": "</function>"},
            ),
        }
        result = await self.run_provider(
            "codex", [*initial[:3], frame, initial[3]], body=request(user=False)
        )
        self.assertEqual(result[0]["tool_calls"][0]["id"], "actual-id")
        self.assertEqual(
            json.loads(result[0]["tool_calls"][0]["function"]["arguments"]),
            {"q": "</function>"},
        )
        self.assertEqual(result[1], {"finish_reason": "tool_calls"})
        self.assertTrue(self.process.killed)
        rows = self.process.sent()[3]["params"]["items"]
        self.assertFalse(any(r.get("role") == "user" for r in rows))

    async def test_codex_keeps_refreshed_credentials_on_failure(self):
        def rotate():
            (self.home / "auth.json").write_text('{"new":true}')

        with (
            patch(
                "api.services.subscription_inference.subscriptions.refresh"
            ) as refresh,
            self.assertRaisesRegex(InferenceFailure, "usage limit"),
        ):
            await self.run_provider(
                "codex",
                [
                    *codex_events(),
                    notify(
                        "turn/completed",
                        turn={
                            "id": "turn",
                            "status": "failed",
                            "error": {"message": "usage limit"},
                        },
                    ),
                ],
                on_exit=rotate,
            )
        refresh.assert_called_once_with(
            "codex", '{"tokens": {"access_token": "stored"}}', '{"new":true}'
        )
        self.assertFalse(self.home.exists())

    async def test_claude_resumes_through_the_call_and_submits_its_typed_result(self):
        def inspect_history():
            self.system = Path(
                self.args[self.args.index("--system-prompt-file") + 1]
            ).read_text()
            self.history = [
                json.loads(line)
                for line in Path(self.args[self.args.index("--resume") + 1])
                .read_text()
                .splitlines()
            ]

        events = [
            claude_event(
                "content_block_delta",
                index=0,
                delta={"type": "text_delta", "text": "<tool_call>"},
            ),
            {"type": "result", "subtype": "success", "result": "<tool_call>"},
        ]
        result = await self.run_provider("claude", events, on_spawn=inspect_history)
        self.assertEqual(
            result, [{"content": "<tool_call>"}, {"finish_reason": "stop"}]
        )
        self.assertEqual(self.env["CLAUDE_CODE_OAUTH_TOKEN"], "claude-credential")
        self.assertEqual(self.system, "system rule")
        self.assertEqual(
            self.args[self.args.index("--resume-session-at") + 1],
            self.history[-2]["uuid"],
        )
        first, assistant, last = [row["message"] for row in self.history]
        self.assertEqual(
            [p["type"] for p in first["content"]], ["text", "image", "text"]
        )
        self.assertEqual(first["content"][0]["text"], "  literal <tool_response> ")
        self.assertEqual(first["content"][1]["source"]["media_type"], "image/png")
        self.assertEqual(first["content"][2]["text"], " after ")
        self.assertEqual(
            assistant["content"][1],
            {
                "type": "tool_use",
                "id": "call-id",
                "name": "mcp__hikari__lookup",
                "input": {"q": "</parameter>"},
            },
        )
        self.assertEqual(last, self.process.sent()[1]["message"])
        self.assertEqual(
            last["content"],
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "call-id",
                    "content": "  result\n",
                }
            ],
        )

    async def test_claude_keeps_consecutive_user_turns_and_anchors_the_first(self):
        body = CompletionRequest(
            messages=[
                {"role": "user", "content": "older"},
                {"role": "user", "content": "respond here"},
            ]
        )

        def read_rows():
            self.rows = [
                json.loads(line)
                for line in Path(self.args[self.args.index("--resume") + 1])
                .read_text()
                .splitlines()
            ]

        await self.run_provider(
            "claude",
            [{"type": "result", "subtype": "success", "result": "reply"}],
            body=body,
            on_spawn=read_rows,
        )
        self.assertEqual([r["message"]["role"] for r in self.rows], ["user", "user"])
        self.assertEqual(
            self.args[self.args.index("--resume-session-at") + 1], self.rows[0]["uuid"]
        )
        self.assertEqual(
            self.process.sent()[0]["message"]["content"],
            [{"type": "text", "text": "respond here"}],
        )

    async def test_claude_supplies_schemas_and_waits_for_completed_native_call(self):
        def control(method):
            return {
                "type": "control_request",
                "request_id": method,
                "request": {
                    "subtype": "mcp_message",
                    "server_name": "hikari",
                    "message": {
                        "method": method,
                        "id": 8,
                        "params": {"protocolVersion": "2025-11-25"},
                    },
                },
            }

        result = await self.run_provider(
            "claude",
            [
                control("initialize"),
                control("tools/list"),
                claude_event(
                    "content_block_start",
                    index=1,
                    content_block={
                        "type": "tool_use",
                        "id": "native",
                        "name": "mcp__hikari__lookup",
                        "input": {},
                    },
                ),
                claude_event(
                    "content_block_delta",
                    index=1,
                    delta={"type": "input_json_delta", "partial_json": '{"q":"'},
                ),
                claude_event(
                    "content_block_delta",
                    index=1,
                    delta={
                        "type": "input_json_delta",
                        "partial_json": '  </parameter>"}',
                    },
                ),
                claude_event("message_delta", delta={"stop_reason": "tool_use"}),
                claude_event("message_stop"),
            ],
            body=request(user=False),
        )
        self.assertEqual(
            self.process.sent()[3]["response"]["response"]["mcp_response"]["result"][
                "tools"
            ][0]["name"],
            "lookup",
        )
        call = result[0]["tool_calls"][0]
        self.assertEqual(
            [
                call["id"],
                call["function"]["name"],
                json.loads(call["function"]["arguments"]),
            ],
            ["native", "lookup", {"q": "  </parameter>"}],
        )
        self.assertEqual(result[-1], {"finish_reason": "tool_calls"})

    async def test_claude_truncated_call_never_becomes_a_command(self):
        result = await self.run_provider(
            "claude",
            [
                claude_event(
                    "content_block_start",
                    index=1,
                    content_block={
                        "type": "tool_use",
                        "id": "native",
                        "name": "mcp__hikari__lookup",
                        "input": {},
                    },
                ),
                claude_event(
                    "content_block_delta",
                    index=1,
                    delta={
                        "type": "input_json_delta",
                        "partial_json": '{"unfinished":',
                    },
                ),
                claude_event("message_delta", delta={"stop_reason": "max_tokens"}),
                claude_event("message_stop"),
            ],
        )
        self.assertEqual(result, [{"finish_reason": "length"}])

    async def test_image_only_input_goes_directly_to_claude_without_placeholder(self):
        body = CompletionRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url()}}
                    ],
                }
            ]
        )
        result = await self.run_provider(
            "claude",
            [{"type": "result", "subtype": "success", "result": "Red"}],
            body=body,
        )
        self.assertEqual(result, [{"content": "Red"}, {"finish_reason": "stop"}])
        self.assertNotIn("--resume", self.args)
        self.assertEqual(
            [v["type"] for v in self.process.sent()[0]["message"]["content"]], ["image"]
        )

    async def test_missing_cli_and_cancelled_turn_release_temporary_home(self):
        stream = subscription_stream(
            ActiveSubscription(provider="claude", credential="t"), request()
        )
        with (
            patch(
                "api.services.subscription_inference.asyncio.create_subprocess_exec",
                side_effect=FileNotFoundError,
            ),
            self.assertRaises(InferenceFailure),
        ):
            await stream.start()
        self.assertIsNone(stream.home)
        process = Process([])
        process.stdin.drain = AsyncMock(side_effect=asyncio.CancelledError)
        with (
            patch(
                "api.services.subscription_inference.asyncio.create_subprocess_exec",
                return_value=process,
            ),
            self.assertRaises(asyncio.CancelledError),
        ):
            async for _ in stream:
                pass
        self.assertTrue(process.killed)
        self.assertIsNone(stream.home)
