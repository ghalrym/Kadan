from uuid import UUID

import anyio
import httpx
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ....config import ServiceError
from ....pydantic_models.resources import AgentKey
from ....services import response_error

router = APIRouter()


class AssetStream(StreamingResponse):
    def __init__(self, response: httpx.Response):
        headers = {
            name: response.headers[name]
            for name in ("content-type", "content-length", "content-disposition")
            if name in response.headers
        }
        headers.update(
            {
                "Cache-Control": "private, no-store",
            }
        )
        super().__init__(response.aiter_raw(), headers=headers)
        self.response = response

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self.response.aclose()


@router.get("/api/agents/{key}/assets/{asset_id}/content")
async def asset_content(
    key: AgentKey, asset_id: UUID, request: Request, download: bool = False
):
    service = request.app.state.services
    upstream = service.client.build_request(
        "GET",
        service.settings.tsuchi_url + f"/api/agents/{key}/assets/{asset_id}/content",
        params={"download": str(download).lower()},
    )
    try:
        response = await service.client.send(upstream, stream=True)
    except httpx.RequestError as error:
        raise ServiceError(
            "Tsuchi is unavailable. Check that its local service is running."
        ) from error
    if not response.is_success:
        try:
            await response.aread()
            raise response_error(response, "Tsuchi")
        finally:
            await response.aclose()
    return AssetStream(response)
