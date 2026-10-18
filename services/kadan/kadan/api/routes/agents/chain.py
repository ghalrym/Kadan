from fastapi import APIRouter, Request

from ...pydantic_models.resources import AgentKey, Resource

router = APIRouter()


@router.get("/api/agents/{key}/chain")
async def get_chain(key: AgentKey, request: Request) -> Resource:
    """Return the agent's one continuous conversation, creating it on first use."""
    return await request.app.state.services.chain(key)
