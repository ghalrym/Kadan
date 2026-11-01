"""Chat through the user's Claude or Codex subscription by driving its own CLI.

The CLIs run with their built-in tools disabled or sandboxed read-only. Tool
definitions and tool history are rendered into the prompt in the same
`<tool_call>` format the local checkpoint emits, so callers parse replies the same way.
"""

import asyncio
import json
import os
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path

import anyio
from api.pydantic_models.subscriptions import SUBSCRIPTION_MODELS, ActiveSubscription
from api.repositories import subscriptions
from api.services.inference_errors import InferenceFailure

OUTPUT_LINE_LIMIT = 16 * 1024 * 1024
ERROR_CHARACTERS = 2000

TOOLS_HEADER = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
TOOLS_FOOTER = (
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following "
    "format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
    "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
    "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an "
    "inner <function=...></function> block must be nested within "
    "<tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n- If there is no function "
    "call available, answer the question like normal with your current knowledge "
    "and do not tell the user about function calls\n</IMPORTANT>"
)
REPLY_INSTRUCTION = (
    "Continue the conversation below as the assistant. Reply with only the "
    "assistant's next message. Do not use any tools of your own; the only "
    "functions available are the ones listed in your instructions."
)


def render_tool_call(name: str, arguments: dict) -> str:
    parameters = "".join(
        f"<parameter={key}>\n"
        + (value if isinstance(value, str) else json.dumps(value))
        + "\n</parameter>\n"
        for key, value in arguments.items()
    )
    return f"<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call>"


def render_prompt(body) -> tuple[str, str]:
    """Split a completion request into system instructions and a transcript prompt."""
    system = [message.text for message in body.messages if message.role == "system"]
    if body.tools:
        system.insert(
            0,
            TOOLS_HEADER
            + "".join(
                "\n" + json.dumps(tool.model_dump(exclude_none=True))
                for tool in body.tools
            )
            + TOOLS_FOOTER,
        )
    turns = []
    for message in body.messages:
        if message.role == "system":
            continue
        if message.role == "tool":
            turns.append(
                f"<user>\n<tool_response>\n{message.text}\n</tool_response>\n</user>"
            )
            continue
        parts = [message.text.strip()] if message.text.strip() else []
        parts += [
            render_tool_call(call.function.name, call.function.arguments)
            for call in getattr(message, "tool_calls", None) or ()
        ]
        turns.append(
            f"<{message.role}>\n" + "\n\n".join(parts) + f"\n</{message.role}>"
        )
    prompt = (
        REPLY_INSTRUCTION
        + "\n\n<conversation>\n"
        + "\n".join(turns)
        + "\n</conversation>"
    )
    return "\n\n".join(system), prompt


class SubscriptionStream:
    """One CLI run; iterate for `content` events, then a terminal `finish_reason`."""

    def __init__(self, subscription: ActiveSubscription, body):
        self.subscription = subscription
        self.model = SUBSCRIPTION_MODELS[subscription.provider]
        self.system, self.prompt = render_prompt(body)
        if subscription.provider == "codex" and self.system:
            # Codex exec has no system prompt option; instructions lead the prompt.
            self.prompt = self.system + "\n\n" + self.prompt
        self.home: Path | None = None
        self.process: asyncio.subprocess.Process | None = None
        self.stderr: asyncio.Task | None = None

    def _command(self) -> tuple[list[str], dict[str, str]]:
        env = {**os.environ, "HOME": str(self.home)}
        if self.subscription.provider == "claude":
            env["CLAUDE_CODE_OAUTH_TOKEN"] = self.subscription.credential
            env.pop("ANTHROPIC_API_KEY", None)
            return [
                "claude",
                "-p",
                "--model",
                self.model,
                "--output-format",
                "stream-json",
                "--verbose",
                "--include-partial-messages",
                "--tools",
                "",
                "--strict-mcp-config",
                "--no-session-persistence",
                "--system-prompt",
                self.system or "You are a helpful assistant.",
            ], env
        env["CODEX_HOME"] = str(self.home)
        env.pop("OPENAI_API_KEY", None)
        (self.home / "auth.json").write_text(self.subscription.credential)
        return [
            "codex",
            "exec",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "-c",
            'approval_policy="never"',
            "--model",
            self.model,
            "--cd",
            str(self.home),
            "-",
        ], env

    async def start(self) -> None:
        self.home = Path(
            tempfile.mkdtemp(prefix=f"hikari-{self.subscription.provider}-")
        )
        command, env = self._command()
        try:
            self.process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.home,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=OUTPUT_LINE_LIMIT,
            )
        except FileNotFoundError as error:
            await self.aclose()
            raise InferenceFailure(
                f"{self.subscription.provider.title()} is not available in this Hikari installation.",
                503,
            ) from error
        self.stderr = asyncio.create_task(self.process.stderr.read())
        self.process.stdin.write(self.prompt.encode())
        await self.process.stdin.drain()
        self.process.stdin.close()

    async def _events(self):
        async for line in self.process.stdout:
            with suppress(ValueError):
                event = json.loads(line)
                if isinstance(event, dict):
                    yield event

    async def _claude(self):
        result = None
        async for event in self._events():
            if event.get("type") == "stream_event":
                delta = event.get("event", {}).get("delta", {})
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yield delta["text"]
            elif event.get("type") == "result":
                result = event
        if result is None or result.get("is_error"):
            message = (result or {}).get("result") or await self._error_text()
            raise InferenceFailure(
                f"Claude subscription request failed: {message}", 502
            )

    async def _codex(self):
        reply, failure, completed = None, None, False
        async for event in self._events():
            kind = event.get("type")
            item = event.get("item") or {}
            if kind == "item.completed" and item.get("type") == "agent_message":
                reply = item.get("text", "")
            elif kind == "turn.completed":
                completed = True
            elif kind == "turn.failed":
                failure = (event.get("error") or {}).get("message")
            elif kind == "error":
                failure = event.get("message")
        if not completed or reply is None:
            message = failure or await self._error_text()
            raise InferenceFailure(f"Codex subscription request failed: {message}", 502)
        if reply:
            yield reply

    async def _error_text(self) -> str:
        await self.process.wait()
        stderr = (await self.stderr).decode(errors="replace").strip()
        return (
            stderr[-ERROR_CHARACTERS:]
            or f"the CLI exited with status {self.process.returncode}."
        )

    async def __aiter__(self):
        if self.process is None:
            await self.start()
        try:
            source = (
                self._claude()
                if self.subscription.provider == "claude"
                else self._codex()
            )
            async for text in source:
                yield {"content": text}
            if await self.process.wait() != 0:
                raise InferenceFailure(
                    f"{self.subscription.provider.title()} subscription request failed: "
                    + await self._error_text(),
                    502,
                )
            yield {"finish_reason": "stop"}
        finally:
            with anyio.CancelScope(shield=True):
                await self.aclose()

    async def text(self) -> str:
        return "".join([event.get("content", "") async for event in self])

    async def aclose(self) -> None:
        if self.process is not None:
            if self.process.returncode is None:
                with suppress(ProcessLookupError):
                    self.process.kill()
            await self.process.wait()
        if self.stderr is not None and not self.stderr.done():
            self.stderr.cancel()
        home, self.home = self.home, None
        if home is None:
            return
        try:
            if self.subscription.provider == "codex":
                await self._keep_refreshed_codex_auth(home / "auth.json")
        finally:
            shutil.rmtree(home, ignore_errors=True)

    async def _keep_refreshed_codex_auth(self, path: Path) -> None:
        """Codex rotates its tokens in auth.json; save them so the next run signs in."""
        with suppress(OSError):
            credential = path.read_text().strip()
            if credential and credential != self.subscription.credential:
                with suppress(ValueError):
                    json.loads(credential)
                    await asyncio.to_thread(
                        subscriptions.refresh,
                        "codex",
                        self.subscription.credential,
                        credential,
                    )
