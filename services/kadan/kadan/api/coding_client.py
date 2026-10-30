"""Control durable coding jobs on the existing computer, never on the API host."""

import base64
import json
from uuid import UUID, uuid4

from pydantic import ValidationError

from .config import ServiceError
from .pydantic_models.coding import CodingJob, CodingStart, CodingStatus
from .sandbox_client import SandboxClient


class CodingClient:
    def __init__(self, computer: SandboxClient):
        self.computer = computer

    async def request(self, operation: str, **arguments) -> dict:
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
                seconds=25,
                json={
                    "session_id": session_id,
                    "command": f"python3 /opt/kadan/coding_jobs.py {payload}",
                    "exec_dir": "/home/gem",
                    "timeout": 20,
                    "hard_timeout": 25,
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
                "Coding control did not confirm the operation. Refresh coding jobs "
                "before retrying. If submitting, inspect the job using the same request ID. "
                "Check that the computer uses the current sandbox image.",
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
                setup={"state": "unavailable", "message": "No computer is configured."},
                jobs=[],
            )
        return await self.read_model(CodingStatus, "status")

    async def job(self, job_id: UUID, after: int = 0) -> CodingJob:
        return await self.read_model(
            CodingJob, "status", job_id=str(job_id), after=after
        )

    async def start(self, arguments: CodingStart) -> CodingJob:
        return await self.read_model(
            CodingJob, "start", **arguments.model_dump(mode="json")
        )

    async def cancel(self, job_id: UUID) -> CodingJob:
        return await self.read_model(CodingJob, "cancel", job_id=str(job_id))

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
