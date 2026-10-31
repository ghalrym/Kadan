"""Codex sign-in and threads on the agent's computer, never on the API host."""

import base64
import json
from uuid import UUID, uuid4

from pydantic import ValidationError

from .config import ServiceError
from .pydantic_models.coding import (
    CodexFinished,
    CodexFinishedRuns,
    CodexOrigin,
    CodexSetup,
    CodexStarted,
    CodexStatus,
    CodexStopped,
    CodexTask,
    CodexThread,
)
from .sandbox_client import SandboxClient


class CodingClient:
    def __init__(self, computer: SandboxClient):
        self.computer = computer

    async def request(self, operation: str, **arguments) -> dict:
        session_id = f"kadan-codex-{uuid4()}"
        payload = base64.urlsafe_b64encode(
            json.dumps(
                {"operation": operation, **arguments}, ensure_ascii=False
            ).encode()
        ).decode()
        try:
            response = await self.computer.request(
                "POST",
                "/v1/bash/exec",
                seconds=35,
                json={
                    "session_id": session_id,
                    "command": f"python3 /opt/kadan/codex_threads.py {payload}",
                    "exec_dir": "/home/gem",
                    "timeout": 30,
                    "hard_timeout": 35,
                    "max_output_length": 2000000,
                },
            )
            data = response.json()["data"]
            if (
                not isinstance(data, dict)
                or data.get("status") != "completed"
                or data.get("exit_code") != 0
            ):
                raise ValueError("Unconfirmed Codex operation")
            result = json.loads(data["stdout"])
            if not isinstance(result, dict):
                raise ValueError("Invalid Codex response")
        except (ServiceError, ValueError, KeyError, TypeError) as error:
            raise ServiceError(
                "Kadan couldn't reach Codex on the agent's computer. Check that the "
                "computer is running, then refresh.",
                503,
                "codex_unavailable",
            ) from error
        finally:
            # Each call opens its own Bash session; close it so polling cannot fill
            # the sandbox's session table. Background runs are detached from it.
            try:
                await self.computer.request(
                    "POST", f"/v1/bash/sessions/{session_id}/close", seconds=8
                )
            except ServiceError:
                pass
        if "error" in result:
            raise ServiceError(result["error"], 409, "codex_rejected")
        return result

    async def read_model(self, model, operation: str, **arguments):
        result = await self.request(operation, **arguments)
        try:
            return model.model_validate(result)
        except ValidationError as error:
            raise ServiceError(
                "The computer returned an invalid Codex response.",
                502,
                "codex_invalid_response",
            ) from error

    async def status(self) -> CodexStatus:
        if not self.computer.configured:
            return CodexStatus(threads=[])
        return await self.read_model(CodexStatus, "threads")

    async def login_status(self) -> CodexSetup:
        if not self.computer.configured:
            return CodexSetup(
                state="unavailable",
                message="No computer is connected. Ask the person who runs Kadan to connect one.",
            )
        return await self.read_model(CodexSetup, "setup")

    async def login(self) -> CodexSetup:
        return await self.read_model(CodexSetup, "login")

    async def logout(self) -> CodexSetup:
        return await self.read_model(CodexSetup, "logout")

    async def thread(self, thread_id: UUID) -> CodexThread:
        return await self.read_model(CodexThread, "thread", thread_id=str(thread_id))

    async def start(
        self, task: CodexTask, request_id: UUID, origin: CodexOrigin
    ) -> CodexStarted:
        return await self.read_model(
            CodexStarted,
            "start",
            **task.model_dump(mode="json"),
            request_id=str(request_id),
            origin=origin.model_dump(mode="json"),
        )

    async def finished(self) -> list[CodexFinished]:
        return (await self.read_model(CodexFinishedRuns, "finished")).runs

    async def acknowledge(self, request_id: UUID) -> None:
        await self.request("acknowledge", request_id=str(request_id))

    async def stop(self, thread_id: UUID) -> CodexStopped:
        return await self.read_model(CodexStopped, "stop", thread_id=str(thread_id))
