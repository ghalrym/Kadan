"""One structured chat request through the user's signed-in provider CLI."""

import asyncio
import json
import os
import shutil
import tempfile
from collections import deque
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import anyio
from api.pydantic_models.chat import ChatMessage, ChatTextPart
from api.pydantic_models.subscriptions import SUBSCRIPTION_MODELS, ActiveSubscription
from api.repositories import subscriptions
from api.services.inference_errors import InferenceFailure

OUTPUT_LINE_LIMIT = 16 * 1024 * 1024
ERROR_CHARACTERS = 2000
CLAUDE_TOOL_PREFIX = "mcp__hikari__"


def text_content(message: ChatMessage) -> str:
    return "".join(
        part.text
        for part in message.ordered_content()
        if isinstance(part, ChatTextPart)
    )


def tool_call(call_id, name, arguments):
    if not isinstance(call_id, str) or not call_id or not isinstance(arguments, dict):
        raise InferenceFailure("The provider returned an incomplete tool call.", 502)
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


class ProviderCLI:
    output_limit = OUTPUT_LINE_LIMIT

    def __init__(self, subscription: ActiveSubscription, body):
        self.subscription = subscription
        self.model = SUBSCRIPTION_MODELS[subscription.provider]
        self.body = body
        self.home: Path | None = None
        self.process: asyncio.subprocess.Process | None = None
        self.stderr: asyncio.Task | None = None

    async def start(self):
        self.home = Path(
            tempfile.mkdtemp(prefix=f"hikari-{self.subscription.provider}-")
        )
        try:
            command, env = self._command()
            self.process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.home,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=self.output_limit,
            )
            self.stderr = asyncio.create_task(self.process.stderr.read())
        except FileNotFoundError as error:
            await self.aclose()
            raise InferenceFailure(
                f"{self.subscription.provider.title()} is unavailable in this Hikari installation."
            ) from error
        except BaseException:
            with anyio.CancelScope(shield=True):
                await self.aclose()
            raise

    async def write(self, frame):
        self.process.stdin.write(
            (json.dumps(frame, ensure_ascii=False) + "\n").encode()
        )
        await self.process.stdin.drain()

    async def read_event(self):
        line = await self.process.stdout.readline()
        if not line:
            raise InferenceFailure(
                f"Provider connection ended. {await self._error_text()}", 502
            )
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise TypeError("expected object")
            return event
        except (ValueError, TypeError) as error:
            raise InferenceFailure(
                "The provider returned an unreadable response.", 502
            ) from error

    async def _error_text(self) -> str:
        await self.process.wait()
        stderr = (await self.stderr).decode(errors="replace").strip()
        return (
            stderr[-ERROR_CHARACTERS:]
            or f"The provider exited with status {self.process.returncode}."
        )

    async def __aiter__(self):
        try:
            if self.process is None:
                await self.start()
            async for event in self.generate_chat():
                yield event
        except (KeyError, TypeError, OSError, ValueError) as error:
            raise InferenceFailure(
                "The provider could not complete the chat request.", 502
            ) from error
        finally:
            with anyio.CancelScope(shield=True):
                await self.aclose()

    async def aclose(self) -> None:
        if self.process is not None:
            if self.process.returncode is None:
                with suppress(ProcessLookupError):
                    self.process.kill()
            await self.process.wait()
        if self.stderr is not None:
            if not self.stderr.done():
                self.stderr.cancel()
            await asyncio.gather(self.stderr, return_exceptions=True)
        home, self.home = self.home, None
        if home is None:
            return
        try:
            if self.subscription.provider == "codex":
                await self._keep_refreshed_codex_auth(home / "auth.json")
        finally:
            shutil.rmtree(home, ignore_errors=True)

    async def _keep_refreshed_codex_auth(self, path: Path) -> None:
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


class CodexAppServer(ProviderCLI):
    def __init__(self, subscription, body):
        super().__init__(subscription, body)
        self.sequence = 0
        self.pending_events = deque()

    async def send(self, method, params, request_id=None):
        frame = {"method": method, "params": params}
        if request_id is not None:
            frame["id"] = request_id
        await self.write(frame)

    async def event(self):
        return (
            self.pending_events.popleft()
            if self.pending_events
            else await self.read_event()
        )

    async def rpc(self, method, params):
        self.sequence += 1
        await self.send(method, params, self.sequence)
        while True:
            event = await self.read_event()
            if event.get("id") == self.sequence and "method" not in event:
                if "error" in event:
                    raise InferenceFailure(
                        str(event["error"].get("message") or event["error"])[:2000], 502
                    )
                return event["result"]
            # Events, including tool calls, can arrive before the RPC acknowledgment.
            self.reject_server_request(event)
            self.pending_events.append(event)

    def reject_server_request(self, event):
        if "id" in event and "method" in event and event["method"] != "item/tool/call":
            raise InferenceFailure(
                "ChatGPT requested an action outside this conversation.", 502
            )

    async def initialize(self):
        await self.rpc(
            "initialize",
            {
                "clientInfo": {"name": "hikari", "title": "Hikari", "version": "1.0"},
                "capabilities": {"experimentalApi": True},
            },
        )
        await self.send("initialized", {})

    def _command(self):
        env = {**os.environ, "HOME": str(self.home), "CODEX_HOME": str(self.home)}
        env.pop("OPENAI_API_KEY", None)
        (self.home / "auth.json").write_text(self.subscription.credential)
        return [
            "codex",
            "app-server",
            "--listen",
            "stdio://",
            "-c",
            "features.shell_tool=false",
            "-c",
            "features.multi_agent=false",
            "-c",
            'web_search="disabled"',
        ], env


def codex_items(messages):
    items = []
    for message in messages:
        if message.role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": text_content(message),
                }
            )
            continue
        parts = []
        for part in message.ordered_content():
            if isinstance(part, ChatTextPart):
                kind = "output_text" if message.role == "assistant" else "input_text"
                parts.append({"type": kind, "text": part.text})
            else:
                parts.append({"type": "input_image", "image_url": part.image_url.url})
        if parts:
            items.append({"type": "message", "role": message.role, "content": parts})
        for call in message.tool_calls or ():
            items.append(
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.function.name,
                    "arguments": json.dumps(call.function.arguments),
                }
            )
    return items


class CodexSubscription(CodexAppServer):
    async def generate_chat(self):
        await self.initialize()
        messages = self.body.messages
        leading = 0
        while leading < len(messages) and messages[leading].role == "system":
            leading += 1
        definitions = [tool.function for tool in self.body.tools or ()]
        names = {tool.name for tool in definitions}
        thread = await self.rpc(
            "thread/start",
            {
                "model": self.model,
                "cwd": str(self.home),
                "ephemeral": True,
                "approvalPolicy": "never",
                "sandbox": "read-only",
                "environments": [],
                "allowProviderModelFallback": False,
                # The CLI hosts a language provider here. Its own environment
                # description would otherwise become an extra user turn.
                "config": {
                    "include_environment_context": False,
                    "include_collaboration_mode_instructions": False,
                    "include_apps_instructions": False,
                    "include_permissions_instructions": False,
                    "project_doc_max_bytes": 0,
                    "tools.update_plan.enabled": False,
                    "tools.experimental_request_user_input.enabled": False,
                    "features.image_generation": False,
                    "features.plugins": False,
                    "features.apps": False,
                    "features.skill_search": False,
                    "features.hooks": False,
                    "features.code_mode": False,
                    "features.code_mode_only": False,
                    "features.sleep_tool": False,
                    "features.skip_host_skill_discovery": True,
                },
                "baseInstructions": "\n\n".join(
                    text_content(m) for m in messages[:leading]
                )
                or "You are a helpful assistant.",
                "dynamicTools": [
                    {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.parameters,
                        "deferLoading": False,
                    }
                    for tool in definitions
                ],
            },
        )
        thread_id = thread["thread"]["id"]
        items = codex_items(messages[leading:])
        if not items:
            raise InferenceFailure("This chat needs a message to answer.", 422)
        await self.rpc("thread/inject_items", {"threadId": thread_id, "items": items})
        # The complete input is already recorded, including the final tool result.
        # Starting an idle thread with empty input samples without adding a user turn.
        started = await self.rpc("turn/start", {"threadId": thread_id, "input": []})
        turn_id = started["turn"]["id"]
        reply = ""
        while True:
            event = await self.event()
            method, params = event.get("method"), event.get("params") or {}
            if params.get("threadId") != thread_id:
                continue
            if method == "item/tool/call" and params.get("turnId") == turn_id:
                if params["tool"] not in names or params.get("namespace"):
                    raise InferenceFailure(
                        "ChatGPT requested an unavailable tool.", 502
                    )
                yield {
                    "tool_calls": [
                        tool_call(params["callId"], params["tool"], params["arguments"])
                    ]
                }
                yield {"finish_reason": "tool_calls"}
                return
            if "id" in event:
                raise InferenceFailure(
                    "ChatGPT requested an action outside this conversation.", 502
                )
            if method == "error" and not params.get("willRetry"):
                raise InferenceFailure(
                    str(
                        params.get("error", {}).get("message")
                        or "ChatGPT request failed."
                    )[:2000],
                    502,
                )
            if method == "item/completed" and params.get("turnId") == turn_id:
                item = params["item"]
                if item["type"] == "agentMessage" and item.get("phase") in (
                    None,
                    "final_answer",
                ):
                    reply = item["text"]
            if method == "turn/completed" and params["turn"]["id"] == turn_id:
                turn = params["turn"]
                if turn["status"] != "completed":
                    raise InferenceFailure(
                        str((turn.get("error") or {}).get("message") or turn["status"])[
                            :2000
                        ],
                        502,
                    )
                if reply:
                    yield {"content": reply}
                yield {"finish_reason": "stop"}
                return


def claude_turn(message):
    if message.role == "tool":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": text_content(message),
                }
            ],
        }
    parts = []
    for part in message.ordered_content():
        if isinstance(part, ChatTextPart):
            parts.append({"type": "text", "text": part.text})
        else:
            mime, payload = part.image_url.url.removeprefix("data:").split(
                ";base64,", 1
            )
            parts.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": payload},
                }
            )
    for call in message.tool_calls or ():
        parts.append(
            {
                "type": "tool_use",
                "id": call.id,
                "name": CLAUDE_TOOL_PREFIX + call.function.name,
                "input": call.function.arguments,
            }
        )
    return {"role": message.role, "content": parts}


class ClaudeSubscription(ProviderCLI):
    def _command(self):
        env = {
            **os.environ,
            "HOME": str(self.home),
            "CLAUDE_CONFIG_DIR": str(self.home / ".claude"),
            "CLAUDE_CODE_OAUTH_TOKEN": self.subscription.credential,
            "ENABLE_TOOL_SEARCH": "false",
        }
        env.pop("ANTHROPIC_API_KEY", None)
        command = [
            "claude",
            "-p",
            "--model",
            self.model,
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--tools",
            "",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--disable-slash-commands",
            "--max-turns",
            "1",
            "--permission-mode",
            "dontAsk",
            "--no-session-persistence",
        ]
        self.tools = {
            tool.function.name: tool.function for tool in self.body.tools or ()
        }
        if self.tools:
            command.extend(
                ["--allowedTools", *(CLAUDE_TOOL_PREFIX + name for name in self.tools)]
            )
        system = []
        turns = []
        for message in self.body.messages:
            if message.role == "system":
                if turns:
                    raise InferenceFailure(
                        "Claude requires system instructions before the conversation.",
                        422,
                    )
                system.append(text_content(message))
            else:
                turns.append(claude_turn(message))
        if not turns or turns[-1]["role"] != "user":
            raise InferenceFailure(
                "Claude needs a user message or tool result to answer.", 422
            )
        self.last_turn = turns[-1]
        system_file = self.home / "system.txt"
        system_file.write_text("\n\n".join(system) or "You are a helpful assistant.")
        command.extend(["--system-prompt-file", str(system_file)])
        if len(turns) > 1:
            # Resume through the penultimate record. Including the final record in the
            # file lets Claude validate a completed call/result pair before truncation.
            session = str(uuid4())
            parent = None
            timestamp = datetime.now(UTC).isoformat()
            path = self.home / "history.jsonl"
            with path.open("w") as out:
                for index, turn in enumerate(turns):
                    row_id = str(uuid4())
                    role = turn["role"]
                    payload = turn
                    if role == "assistant":
                        payload = {
                            **turn,
                            "type": "message",
                            "id": "msg_" + uuid4().hex,
                            "model": self.model,
                            "stop_reason": "tool_use"
                            if any(
                                part["type"] == "tool_use" for part in turn["content"]
                            )
                            else "end_turn",
                        }
                    row = {
                        "uuid": row_id,
                        "parentUuid": parent,
                        "sessionId": session,
                        "type": role,
                        "message": payload,
                        "timestamp": timestamp,
                        "isSidechain": False,
                        "cwd": str(self.home),
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if index == len(turns) - 2:
                        anchor = row_id
                    parent = row_id
            command.extend(
                ["--resume", str(path), "--fork-session", "--resume-session-at", anchor]
            )
        return command, env

    async def serve_mcp(self, frame):
        request = frame["request"]
        if (
            request.get("subtype") != "mcp_message"
            or request.get("server_name") != "hikari"
        ):
            raise InferenceFailure("Claude requested an unsupported action.", 502)
        message = request["message"]
        method = message.get("method")
        if method == "initialize":
            result = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "hikari", "version": "1.0"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.parameters,
                    }
                    for tool in self.tools.values()
                ]
            }
        elif (
            method == "tools/call"
            and message.get("params", {}).get("name") in self.tools
        ):
            # The CLI may request execution while its inference stream is closing.
            # The outer Kadan run owns execution after the provider finishes.
            return
        elif method in ("ping", "notifications/initialized"):
            result = {}
        else:
            raise InferenceFailure(
                "Claude requested an unsupported subscription tool operation.", 502
            )
        await self.write(
            {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": frame["request_id"],
                    "response": {
                        "mcp_response": {
                            "id": message.get("id", 0),
                            "jsonrpc": "2.0",
                            "result": result,
                        }
                    },
                },
            }
        )

    async def generate_chat(self):
        if self.tools:
            await self.write(
                {
                    "type": "control_request",
                    "request_id": "hikari-init",
                    "request": {"subtype": "initialize", "sdkMcpServers": ["hikari"]},
                }
            )
            while True:
                event = await self.read_event()
                if event.get("type") == "control_request":
                    await self.serve_mcp(event)
                elif event.get("type") == "control_response":
                    reply = event["response"]
                    if reply.get("request_id") == "hikari-init":
                        if reply.get("subtype") != "success":
                            raise InferenceFailure(
                                str(
                                    reply.get("error")
                                    or "Claude could not initialize tools."
                                )[:2000],
                                502,
                            )
                        break
                elif event.get("type") == "result":
                    raise InferenceFailure(
                        str(
                            event.get("result") or "Claude ended during initialization."
                        )[:2000],
                        502,
                    )
        await self.write(
            {"type": "user", "message": self.last_turn, "parent_tool_use_id": None}
        )
        if not self.tools:
            self.process.stdin.close()
        calls = {}
        reason = None
        emitted_text = False
        while True:
            event = await self.read_event()
            if event.get("parent_tool_use_id") is not None:
                continue
            if event.get("type") == "control_request":
                await self.serve_mcp(event)
            if (
                event.get("type") == "control_response"
                and event["response"]["subtype"] == "error"
            ):
                raise InferenceFailure(
                    str(
                        event["response"].get("error")
                        or "Claude initialization failed."
                    )[:2000],
                    502,
                )
            if event.get("type") == "stream_event":
                block = event["event"]
                kind, delta = block["type"], block.get("delta") or {}
                if (
                    kind == "content_block_start"
                    and block["content_block"]["type"] == "tool_use"
                ):
                    tool = block["content_block"]
                    name = tool["name"].removeprefix(CLAUDE_TOOL_PREFIX)
                    if (
                        tool["name"] != CLAUDE_TOOL_PREFIX + name
                        or name not in self.tools
                    ):
                        raise InferenceFailure(
                            "Claude requested an unavailable tool.", 502
                        )
                    calls[block["index"]] = (tool["id"], name, tool["input"], [])
                elif (
                    kind == "content_block_delta"
                    and delta["type"] == "input_json_delta"
                ):
                    calls[block["index"]][3].append(delta["partial_json"])
                elif kind == "content_block_delta" and delta["type"] == "text_delta":
                    emitted_text = True
                    yield {"content": delta["text"]}
                elif kind == "message_delta":
                    reason = delta.get("stop_reason")
                elif kind == "message_stop" and reason == "max_tokens":
                    yield {"finish_reason": "length"}
                    return
                elif kind == "message_stop" and reason == "tool_use":
                    if not calls:
                        raise InferenceFailure(
                            "Claude returned an incomplete tool call.", 502
                        )
                    yield {
                        "tool_calls": [
                            tool_call(
                                call_id,
                                name,
                                json.loads("".join(chunks)) if chunks else initial,
                            )
                            for call_id, name, initial, chunks in calls.values()
                        ]
                    }
                    yield {"finish_reason": "tool_calls"}
                    return
            if event.get("type") == "result":
                if event.get("is_error") or event.get("subtype") != "success":
                    raise InferenceFailure(
                        str(
                            event.get("result")
                            or event.get("errors")
                            or "Claude request failed."
                        )[:2000],
                        502,
                    )
                if calls:
                    raise InferenceFailure(
                        "Claude ended without confirming its tool call.", 502
                    )
                if not emitted_text and event.get("result"):
                    yield {"content": event["result"]}
                yield {"finish_reason": "stop"}
                return


def subscription_stream(subscription: ActiveSubscription, body) -> ProviderCLI:
    if subscription.provider == "codex":
        return CodexSubscription(subscription, body)
    return ClaudeSubscription(subscription, body)
