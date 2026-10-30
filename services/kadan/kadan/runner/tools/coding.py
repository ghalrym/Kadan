from uuid import UUID

from pydantic import Field

from ...api.coding_client import CodingClient
from ...api.pydantic_models.coding import CodingStart
from ...api.sandbox_client import SandboxClient
from .base import Arguments, computer_tool


class CodingStatusArguments(Arguments):
    job_id: UUID | None = Field(
        default=None, description="Omit to inspect setup and recent jobs."
    )
    after: int = Field(default=0, ge=0, description="Last event cursor already read.")


@computer_tool("GET")
async def codex_status(args: CodingStatusArguments, computer: SandboxClient) -> dict:
    """Read Codex setup or saved job progress. A reconnect never starts work.

    For a lost submission, job_id is its request_id. Keep next_cursor for later reads.
    """
    client = CodingClient(computer)
    if args.job_id is None:
        status = await client.status()
        return {
            "setup": status.setup.model_dump(),
            "jobs": [
                {
                    "id": str(job.id),
                    "status": job.status,
                    "cwd": job.cwd[:200],
                    "session_id": job.session_id,
                }
                for job in status.jobs[:5]
            ],
            "recent_jobs": len(status.jobs),
            "jobs_truncated": len(status.jobs) > 5,
        }
    result = await client.job(args.job_id, args.after)
    return result.model_dump(mode="json")


@computer_tool("POST")
async def codex_start(args: CodingStart, computer: SandboxClient) -> dict:
    """Delegate coding to Codex in the shared computer; returns before completion.

    Inspect codex_status first. Supply the user's scoped task and absolute project
    directory. Choose one UUID request_id and reuse it after uncertain submission;
    the job ID is that UUID. One job runs at a time. To continue a saved Codex
    session, supply its prior resume_job_id with a new request_id and follow-up
    prompt. Never resubmit completed work as a new job to inspect its result.
    """
    return (await CodingClient(computer).start(args)).model_dump(mode="json")


class CodingCancelArguments(Arguments):
    job_id: UUID


@computer_tool("POST")
async def codex_cancel(args: CodingCancelArguments, computer: SandboxClient) -> dict:
    """Stop a coding job and its owned processes. Read status until cleanup is confirmed.

    Chat Stop is separate. Cancellation keeps file changes already made.
    """
    return (await CodingClient(computer).cancel(args.job_id)).model_dump(mode="json")


TOOLS = [codex_status, codex_start, codex_cancel]
