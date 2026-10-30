from typing import Protocol
from uuid import UUID

from pydantic import Field

from ...api.coding_client import CodingClient
from ...api.pydantic_models.coding import CodingTask
from ...api.sandbox_client import SandboxClient
from .base import Arguments, computer_tool, conversation_tool


class CodingStatusArguments(Arguments):
    job_id: UUID | None = Field(
        default=None, description="Omit to inspect setup and recent jobs."
    )
    after: int = Field(default=0, ge=0, description="Last event cursor already read.")


@computer_tool("GET")
async def codex_status(args: CodingStatusArguments, computer: SandboxClient) -> dict:
    """Read Codex setup or saved job progress. A reconnect never starts work.

    Use the recorded job ID. Keep next_cursor for later reads.
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


class CodingConversation(Protocol):
    async def delegate_coding(self, args: CodingTask) -> dict: ...


@conversation_tool
async def codex_start(args: CodingTask, conversation: CodingConversation) -> dict:
    """Delegate coding and receive its final result in this conversation run.

    Inspect codex_status first. Supply the user's scoped task and absolute project
    directory. Kadan waits without inference and reattaches after a runner restart.
    The runner owns the durable job ID. One job runs at a time. To continue a saved
    Codex session, supply its prior resume_job_id and a scoped follow-up prompt.
    After the result, finish the requested work and report the outcome with
    send_message. Do not resubmit failed or uncertain work automatically.
    """
    return await conversation.delegate_coding(args)


class CodingCancelArguments(Arguments):
    job_id: UUID


@computer_tool("POST")
async def codex_cancel(args: CodingCancelArguments, computer: SandboxClient) -> dict:
    """Stop a coding job and its owned processes. Read status until cleanup is confirmed.

    Chat Stop is separate. Cancellation keeps file changes already made.
    """
    return (await CodingClient(computer).cancel(args.job_id)).model_dump(mode="json")


TOOLS = [codex_status, codex_cancel]
DELEGATION_TOOLS = [codex_start]
