from typing import Annotated
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator
from api.pydantic_models.requests import RequestRecord, RequestType, RequestStatus

router = APIRouter(prefix="/v1/requests", tags=["Requests"])

MOCK_REQUESTS = [
    RequestRecord(
        id='req_8d91b20cfa01',
        time='14:28:51.204',
        type='LLM',
        model='Qwen3 32B Instruct',
        status=200,
        latency='1.24 s',
        ttft='182 ms',
        tokensPerSecond=84,
        endpoint='/v1/chat/completions',
        prompt='Summarize this incident report in three bullets.',
        output=('• Image generation failed for 52 minutes after GPU 0 ran out of '
                'memory.\n'
                '• Unthrottled image edit jobs caused the spike.\n'
                '• A per-client concurrency cap of 4 has been applied.'),
    ),
    RequestRecord(
        id='req_09cf137ab202',
        time='14:28:48.916',
        type='Image',
        model='FLUX.1 [dev]',
        status=200,
        latency='5.82 s',
        endpoint='/v1/images/generations',
        prompt='isometric server room, soft morning light',
        output='img_b202.png · 1024×1024',
    ),
    RequestRecord(
        id='req_a2e671c8b303',
        time='14:28:44.370',
        type='STT',
        model='Whisper Large v3 Turbo',
        status=200,
        latency='1.68 s',
        endpoint='/v1/audio/transcriptions',
        prompt='meeting_0412.wav',
        output=('Okay, quick update on the migration. The new GPU node is racked and '
                'passing burn-in. I would like to move batch transcription over on '
                'Thursday.'),
    ),
    RequestRecord(
        id='req_aa9dca32f404',
        time='14:28:41.062',
        type='Decision',
        model='jev-latest',
        status=200,
        latency='426 ms',
        endpoint='/v1/decisions',
        prompt=('Customer cannot reset their password. They tried the reset link twice '
                'and it expired both times.'),
        output='tier_1',
    ),
    RequestRecord(
        id='req_719cd304d505',
        time='14:28:37.519',
        type='TTS',
        model='F5-TTS',
        status=200,
        latency='892 ms',
        endpoint='/v1/audio/speech',
        prompt='Your order has shipped and will arrive Thursday.',
        output='tts_d505.wav',
    ),
    RequestRecord(
        id='req_604aa250f606',
        time='14:28:33.821',
        type='Video',
        model='Wan 2.2 T2V 14B',
        status=202,
        latency='108 ms',
        endpoint='/v1/videos/generations',
        prompt='drone shot over a foggy pine forest at sunrise',
        output='vid_f606 · queued · 8s · 720p',
    ),
    RequestRecord(
        id='req_002dc17b1707',
        time='14:28:29.094',
        type='LLM',
        model='Qwen3 32B Instruct',
        status=200,
        latency='2.31 s',
        ttft='243 ms',
        tokensPerSecond=76,
        endpoint='/v1/chat/completions',
        prompt='Write a SQL query that returns weekly active users.',
        output=("SELECT date_trunc('week', created_at) AS week,\n"
                '       count(DISTINCT user_id) AS wau\n'
                'FROM events\n'
                'GROUP BY 1\n'
                'ORDER BY 1 DESC;'),
    ),
    RequestRecord(
        id='req_3c916b21e808',
        time='14:28:24.652',
        type='Image',
        model='FLUX.1 Kontext',
        status=500,
        latency='38 ms',
        endpoint='/v1/images/edits',
        prompt='replace the sky with a stormy overcast, keep the building untouched',
        output='CUDA out of memory while allocating 2.1 GiB.',
    ),
    RequestRecord(
        id='req_8cc31529e909',
        time='14:28:21.183',
        type='LLM',
        model='Qwen3 32B Instruct',
        status=429,
        latency='12 ms',
        endpoint='/v1/chat/completions',
        prompt='Classify this support ticket by urgency.',
        output='Client batch-worker exceeded 60 requests/min.',
    ),
    RequestRecord(
        id='req_145af202a010',
        time='14:28:17.806',
        type='TTS',
        model='F5-TTS',
        status=200,
        latency='1.02 s',
        endpoint='/v1/audio/speech',
        prompt='Welcome back. You have three new messages.',
        output='tts_a010.wav',
    ),
    RequestRecord(
        id='req_ef35d193a111',
        time='14:28:13.440',
        type='LLM',
        model='Qwen3 32B Instruct',
        status=200,
        latency='1.82 s',
        ttft='206 ms',
        tokensPerSecond=92,
        endpoint='/v1/chat/completions',
        prompt='Translate the release notes into Spanish.',
        output=('Notas de la versión 0.0.1: se añadió el registro de solicitudes en '
                'vivo y la generación de video.'),
    ),
    RequestRecord(
        id='req_268bc601c212',
        time='14:28:09.254',
        type='STT',
        model='Whisper Large v3 Turbo',
        status=200,
        latency='2.04 s',
        endpoint='/v1/audio/transcriptions',
        prompt='standup.webm',
        output=('Yesterday I finished the request log. Today I am on video generation. '
                'No blockers.'),
    ),
    RequestRecord(
        id='req_149df823d313',
        time='14:28:05.941',
        type='Decision',
        model='jev-latest',
        status=200,
        latency='318 ms',
        endpoint='/v1/decisions',
        prompt=('Customer cannot reset their password. They tried the reset link twice '
                'and it expired both times.'),
        output='tier_1',
    ),
    RequestRecord(
        id='req_509bc373e414',
        time='14:28:01.627',
        type='Image',
        model='FLUX.1 [dev]',
        status=200,
        latency='6.14 s',
        endpoint='/v1/images/generations',
        prompt='product shot of a matte black speaker on concrete',
        output='img_e414.png · 1024×1024',
    ),
]


class RequestFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: RequestType | None = None
    status: RequestStatus | None = None
    search: str = ""
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=100)

    @field_validator("status", mode="before")
    @classmethod
    def parse_status_query(cls, value: object) -> object:
        if isinstance(value, str) and value.isdecimal():
            return int(value)
        return value


class RequestsResponse(BaseModel):
    requests: list[RequestRecord]
    total: int


class RequestResponse(BaseModel):
    request: RequestRecord


@router.get("")
def list_requests(filters: Annotated[RequestFilters, Query()]) -> RequestsResponse:
    matches = [record for record in MOCK_REQUESTS
               if (filters.type is None or record.type == filters.type)
               and (filters.status is None or record.status == filters.status)
               and filters.search.casefold() in " ".join((record.id, record.endpoint, record.model, record.prompt, record.output)).casefold()]
    return RequestsResponse(requests=matches[filters.offset:filters.offset + filters.limit], total=len(matches))


@router.get("/{request_id}")
def get_request(request_id: str) -> RequestResponse:
    for record in MOCK_REQUESTS:
        if record.id == request_id:
            return RequestResponse(request=record)
    raise HTTPException(status_code=404, detail="Request not found")
