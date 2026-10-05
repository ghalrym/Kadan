"""Upload conditioning media for native video jobs."""
from typing import Literal
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from api.services.video_inputs import video_inputs

router = APIRouter(prefix='/v1/videos/inputs', tags=['Videos'])


class VideoInputResponse(BaseModel):
    id: str


@router.post('', operation_id='uploadVideoInput')
async def upload_video_input(kind: Literal['image', 'audio', 'video'], request: Request) -> VideoInputResponse:
    try:
        return VideoInputResponse(id=await video_inputs.save(kind, request.stream(), request.headers.get('content-type', '')))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
