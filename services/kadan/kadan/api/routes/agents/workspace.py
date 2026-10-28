from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Request

from ...pydantic_models.resources import AgentKey, ResourceKind, WorkspaceDocument

router = APIRouter()
DOCUMENT_KINDS = (ResourceKind.SPACE, ResourceKind.DECK)


@router.get("/api/agents/{key}/workspace/documents")
async def list_documents(
    key: AgentKey,
    request: Request,
    q: Annotated[str, Query(max_length=500)] = "",
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[WorkspaceDocument]:
    resources = await request.app.state.services.list_resource_kinds(
        key, DOCUMENT_KINDS, q=q, limit=limit, offset=offset
    )
    return [
        WorkspaceDocument.model_validate(resource.model_dump())
        for resource in resources
    ]


@router.get("/api/agents/{key}/workspace/documents/{resource_id}")
async def get_document(
    key: AgentKey, resource_id: UUID, request: Request
) -> WorkspaceDocument:
    resource = await request.app.state.services.resource_of_kinds(
        key, resource_id, DOCUMENT_KINDS
    )
    return WorkspaceDocument.model_validate(resource.model_dump())
