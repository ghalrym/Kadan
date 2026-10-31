"""Control durable coding jobs on the existing computer, never on the API host."""

import base64
import json
from uuid import UUID, uuid4

from pydantic import ValidationError

from .config import ServiceError
from .pydantic_models.coding import (
    CodingJob,
    CodingStart,
    CodingStatus,
    CodexLoginStatus,
)
from .sandbox_client import SandboxClient


class CodingClient:
    def __init__(self, computer: SandboxClient):
        self.computer = computer

    async def request(
        self, operation: str, *, login_control: bool = False, **arguments
    ) -> dict:
        session_id = f"kadan-coding-control-{uuid4()}"
        payload = base64.urlsafe_b64encode(
            json.dumps(
                {"operation": operation, **arguments}, ensure_ascii=False
            ).encode()
        ).decode()
        try:
            response = await self.computer.request(
                "POST",
                "/v1/bash/exec",
                seconds=35 if login_control else 25,
                json={
                    "session_id": session_id,
                    "command": f"python3 /opt/kadan/{'codex_login.py' if login_control else 'coding_jobs.py'} {payload}",
                    "exec_dir": "/home/gem",
                    "timeout": 30 if login_control else 20,
                    "hard_timeout": 35 if login_control else 25,
                    "max_output_length": 256000,
                },
            )
            envelope = response.json()
            data = envelope["data"]
            if (
                not isinstance(data, dict)
                or data.get("status") != "completed"
                or data.get("exit_code") != 0
            ):
                raise ValueError("Unconfirmed control operation")
            result = json.loads(data["stdout"])
            if not isinstance(result, dict):
                raise ValueError("Invalid control response")
        except (ServiceError, ValueError, KeyError, TypeError) as error:
            raise ServiceError(
                "Kadan could not confirm the coding task's status. Refresh Coding "
                "before trying again; the task may still be running. Do not submit "
                "it again. If this keeps failing, ask the person who runs Kadan "
                "to check that its computer is running and up to date.",
                503,
                "coding_unconfirmed",
            ) from error
        finally:
            # Polling must not fill the sandbox's limited Bash session table and
            # evict unrelated commands. The detached worker owns a separate session.
            try:
                await self.computer.request(
                    "POST", f"/v1/bash/sessions/{session_id}/close", seconds=8
                )
            except ServiceError:
                pass
        if "error" in result:
            # Only the supervisor's fixed messages are exposed; shell stderr is never returned.
            raise ServiceError(result["error"], 409, "coding_rejected")
        return result

    async def status(self) -> CodingStatus:
        if not self.computer.configured:
            return CodingStatus(
                setup={
                    "state": "unavailable",
                    "message": "No computer is connected. Ask the person who runs Kadan to connect one, then refresh here.",
                },
                jobs=[],
            )
        return await self.read_model(CodingStatus, "status")

    async def job(self, job_id: UUID, after: int = 0) -> CodingJob:
        job = await self.read_model(
            CodingJob, "status", job_id=str(job_id), after=after
        )
        return self.match_job(job, job_id)

    async def start(self, arguments: CodingStart) -> CodingJob:
        job = await self.read_model(
            CodingJob, "start", **arguments.model_dump(mode="json")
        )
        return self.match_job(job, arguments.request_id)

    async def cancel(self, job_id: UUID) -> CodingJob:
        job = await self.read_model(CodingJob, "cancel", job_id=str(job_id))
        return self.match_job(job, job_id)

    @staticmethod
    def match_job(job: CodingJob, job_id: UUID) -> CodingJob:
        if job.id != job_id or job.request_id != job_id:
            raise ServiceError(
                "The computer returned another coding job's status.",
                502,
                "coding_invalid_response",
            )
        return job

    async def read_model(self, model, operation: str, **arguments):
        result = await self.request(operation, **arguments)
        try:
            return model.model_validate(result)
        except ValidationError as error:
            raise ServiceError(
                "The computer returned an invalid coding status. Refresh before retrying.",
                502,
                "coding_invalid_response",
            ) from error

    async def login_control(self, operation: str) -> CodexLoginStatus:
        result = await self.request(operation, login_control=True)
        try:
            return CodexLoginStatus.model_validate(result)
        except ValidationError as error:
            raise ServiceError(
                "The computer returned an invalid sign-in status.",
                502,
                "coding_invalid_response",
            ) from error
