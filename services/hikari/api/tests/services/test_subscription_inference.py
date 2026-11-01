import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from api.pydantic_models.subscriptions import ActiveSubscription
from api.routes.v1.chat.completions import CompletionRequest
from api.services import subscription_inference
from api.services.inference_errors import InferenceFailure
from api.services.subscription_inference import SubscriptionStream, render_prompt


class FakeStream:
    def __init__(self, lines):
        self.lines = [line.encode() + b"\n" for line in lines]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.lines:
            raise StopAsyncIteration
        return self.lines.pop(0)


class FakeStdin:
    def __init__(self):
        self.data = b""

    def write(self, data):
        self.data += data

    async def drain(self):
        pass

    def close(self):
        pass


class FakeStderr:
    def __init__(self, text=""):
        self.text = text

    async def read(self):
        return self.text.encode()


class FakeProcess:
    def __init__(self, lines, returncode=0, stderr="", on_exit=None):
        self.stdout = FakeStream(lines)
        self.stdin = FakeStdin()
        self.stderr = FakeStderr(stderr)
        self.exit_code = returncode
        self.returncode = None
        self.on_exit = on_exit
        self.killed = False

    async def wait(self):
        if self.returncode is None and self.on_exit:
            self.on_exit()
        self.returncode = self.exit_code
        return self.returncode

    def kill(self):
        self.killed = True


def request(**values):
    return CompletionRequest(
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "function": {
                            "name": "lookup",
                            "arguments": {"city": "Oslo", "days": 2},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "Rain", "tool_call_id": "call_0"},
        ],
        tools=[{"function": {"name": "lookup", "parameters": {"type": "object"}}}],
        **values,
    )


class PromptTests(unittest.TestCase):
    def test_tools_and_history_use_the_local_checkpoint_format(self):
        system, prompt = render_prompt(request())
        self.assertTrue(system.startswith(subscription_inference.TOOLS_HEADER))
        self.assertIn('"name": "lookup"', system)
        self.assertTrue(system.endswith("Be brief."))
        self.assertIn("<user>\nWeather?\n</user>", prompt)
        self.assertIn(
            "<function=lookup>\n<parameter=city>\nOslo\n</parameter>\n"
            "<parameter=days>\n2\n</parameter>\n</function>",
            prompt,
        )
        self.assertIn("<tool_response>\nRain\n</tool_response>", prompt)
        self.assertNotIn("Be brief.", prompt)


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def run_stream(self, subscription, process):
        with patch(
            "api.services.subscription_inference.asyncio.create_subprocess_exec",
            return_value=process,
        ) as spawn:
            self.spawn = spawn
            stream = SubscriptionStream(subscription, request())
            events = [event async for event in stream]
        return events, spawn

    async def test_claude_streams_text_deltas_with_its_token(self):
        delta = lambda text: json.dumps(
            {
                "type": "stream_event",
                "event": {"delta": {"type": "text_delta", "text": text}},
            }
        )
        process = FakeProcess(
            [
                delta("Hel"),
                "not json",
                delta("lo"),
                json.dumps({"type": "result", "is_error": False}),
            ]
        )
        events, spawn = await self.run_stream(
            ActiveSubscription(provider="claude", credential="sk-ant-oat01-x"), process
        )
        self.assertEqual(
            events, [{"content": "Hel"}, {"content": "lo"}, {"finish_reason": "stop"}]
        )
        command = spawn.call_args.args
        self.assertEqual(command[command.index("--model") + 1], "claude-haiku-5-5")
        self.assertEqual(command[command.index("--tools") + 1], "")
        self.assertEqual(
            spawn.call_args.kwargs["env"]["CLAUDE_CODE_OAUTH_TOKEN"], "sk-ant-oat01-x"
        )
        self.assertIn(b"<conversation>", process.stdin.data)

    async def test_claude_error_result_fails_the_turn(self):
        process = FakeProcess(
            [
                json.dumps(
                    {"type": "result", "is_error": True, "result": "Not logged in"}
                )
            ]
        )
        with self.assertRaisesRegex(InferenceFailure, "Not logged in"):
            await self.run_stream(
                ActiveSubscription(provider="claude", credential="t"), process
            )

    async def test_codex_returns_last_message_and_keeps_refreshed_auth(self):
        original = json.dumps({"tokens": {"access_token": "old"}})
        refreshed = json.dumps({"tokens": {"access_token": "new"}})
        home = lambda: Path(self.spawn.call_args.kwargs["env"]["CODEX_HOME"])
        process = FakeProcess(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "Thinking"},
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "Rainy"},
                    }
                ),
                json.dumps({"type": "turn.completed"}),
            ],
            on_exit=lambda: (home() / "auth.json").write_text(refreshed),
        )
        with patch(
            "api.services.subscription_inference.subscriptions.refresh"
        ) as refresh:
            events, spawn = await self.run_stream(
                ActiveSubscription(provider="codex", credential=original), process
            )
        self.assertEqual(events, [{"content": "Rainy"}, {"finish_reason": "stop"}])
        command = spawn.call_args.args
        self.assertEqual(command[command.index("--model") + 1], "gpt-6.1-terra")
        self.assertTrue(
            process.stdin.data.decode().startswith(subscription_inference.TOOLS_HEADER)
        )
        refresh.assert_called_once_with("codex", original, refreshed)
        self.assertFalse(home().exists())

    async def test_codex_failure_reports_its_message(self):
        process = FakeProcess(
            [json.dumps({"type": "turn.failed", "error": {"message": "usage limit"}})],
            returncode=1,
        )
        with self.assertRaisesRegex(InferenceFailure, "usage limit"):
            await self.run_stream(
                ActiveSubscription(provider="codex", credential='{"tokens": {}}'),
                process,
            )

    async def test_missing_cli_is_unavailable(self):
        with (
            patch(
                "api.services.subscription_inference.asyncio.create_subprocess_exec",
                side_effect=FileNotFoundError,
            ),
            self.assertRaises(InferenceFailure) as caught,
        ):
            await SubscriptionStream(
                ActiveSubscription(provider="claude", credential="t"), request()
            ).start()
        self.assertEqual(caught.exception.status_code, 503)

    async def test_cancelled_start_kills_process_and_removes_home(self):
        process = FakeProcess([])

        async def cancelled_drain():
            raise asyncio.CancelledError

        process.stdin.drain = cancelled_drain
        stream = SubscriptionStream(
            ActiveSubscription(provider="claude", credential="t"), request()
        )
        with (
            patch(
                "api.services.subscription_inference.asyncio.create_subprocess_exec",
                return_value=process,
            ) as spawn,
            self.assertRaises(asyncio.CancelledError),
        ):
            await stream.start()
        self.assertTrue(process.killed)
        self.assertFalse(Path(spawn.call_args.kwargs["cwd"]).exists())
