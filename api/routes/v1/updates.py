"""Fixed-operation bridge to the optional host updater; no Docker/exec access."""
import os

from fastapi import APIRouter, Request, Response, HTTPException
import httpx
from pydantic import BaseModel, ConfigDict, Field

from api.services.release import identity

router = APIRouter(prefix='/v1/updates', tags=['Updates'])


class UpdateStatus(BaseModel):
    current: str
    available: str | None = None
    configured: bool = False
    authorized: bool = False
    phase: str = 'disabled'
    message: str = 'Updates require one-time server setup.'
    error: str | None = None
    can_cancel: bool = False


class PairRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    code: str = Field(min_length=32, max_length=128)


class InstallRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    commit: str = Field(pattern=r'^[0-9a-f]{40}$')


def bridge(request, response, method, operation, body=None):
    socket = os.environ.get('KADAN_UPDATER_SOCKET')
    if not socket:
        if method == 'GET':
            return UpdateStatus(current=identity()['commit'])
        raise HTTPException(503, 'Updates require one-time server setup.')
    headers = {name: request.headers.get(name, '') for name in
               ('origin', 'cookie', 'x-kadan-update', 'sec-fetch-site')}
    if method == 'POST':
        headers['content-type'] = 'application/json'
    try:
        with httpx.Client(transport=httpx.HTTPTransport(uds=socket), timeout=5, trust_env=False) as client:
            result = client.request(method, 'http://localhost/' + operation, headers=headers, json=body)
        payload = result.json()
    except (httpx.HTTPError, ValueError, OSError) as exc:
        raise HTTPException(503, 'Host updater is unavailable; check its service on the server.') from exc
    if result.status_code >= 400:
        raise HTTPException(result.status_code, payload.get('detail', 'Update request failed'))
    if 'set-cookie' in result.headers:
        response.headers['set-cookie'] = result.headers['set-cookie']
    response.headers['cache-control'] = 'no-store'
    return UpdateStatus(**payload)


@router.get('', response_model=UpdateStatus)
def get_updates(request: Request, response: Response):
    return bridge(request, response, 'GET', 'status')


@router.post('/pair', response_model=UpdateStatus)
def pair_updates(body: PairRequest, request: Request, response: Response):
    return bridge(request, response, 'POST', 'pair', body.model_dump())


@router.post('/install', response_model=UpdateStatus, status_code=202)
def install_update(body: InstallRequest, request: Request, response: Response):
    return bridge(request, response, 'POST', 'install', body.model_dump())


@router.post('/cancel', response_model=UpdateStatus)
def cancel_update(request: Request, response: Response):
    return bridge(request, response, 'POST', 'cancel')
