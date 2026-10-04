from typing import Annotated
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from api.pydantic_models.requests import RequestRecord, RequestType, RequestStatus
from api.services.telemetry import telemetry

router = APIRouter(prefix='/v1/requests', tags=['Requests'])


class RequestFilters(BaseModel):
    """Validate optional type/status filters and a bounded search/page request."""
    model_config = ConfigDict(extra='forbid')
    type: RequestType | None = None
    status: RequestStatus | None = None
    search: str = Field(default='', max_length=128)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=100)


class RequestsResponse(BaseModel):
    """A page of retained observations with its filtered total and process epoch."""
    requests: list[RequestRecord]
    total: int
    retention_limit: int
    started_at: str


class RequestResponse(BaseModel):
    """Wrap a single retained HTTP observation for the detail endpoint."""
    request: RequestRecord


@router.get('', operation_id='listRequests')
def list_requests(filters: Annotated[RequestFilters, Query()]) -> RequestsResponse:
    """Filter a newest-first snapshot and return the requested page.

    Search matches IDs, endpoints, catalog IDs and safe summaries. Totals cover
    retained matches only; concurrent completions or eviction can move records
    between successive page requests. No database history is consulted."""
    matches = [record for record in telemetry.records()
               if (filters.type is None or record.type == filters.type)
               and (filters.status is None or record.status == filters.status)
               and filters.search.casefold() in ' '.join((record.id, record.endpoint, record.model or '',
                                                          record.prompt, record.output)).casefold()]
    return RequestsResponse(requests=matches[filters.offset:filters.offset + filters.limit],
                            total=len(matches), retention_limit=telemetry.capacity, started_at=telemetry.started_at)


@router.get('/{request_id}', operation_id='getRequest')
def get_request(request_id: str) -> RequestResponse:
    """Return a retained observation by ID, or 404 after eviction or restart.

    The lookup uses a snapshot, so it never holds the telemetry lock while
    serializing the response."""
    for record in telemetry.records():
        if record.id == request_id:
            return RequestResponse(request=record)
    raise HTTPException(status_code=404, detail='Request not found; records expire and reset on API restart')
