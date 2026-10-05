"""Host-helper-only drain and readiness, never exposed by the frontend proxy."""
import asyncio
import hmac
import os
from pathlib import Path
import time

from alembic.config import Config
from alembic.script import ScriptDirectory
from fastapi import APIRouter, Header, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import text

from api.database import get_engine
from api.services.decisions import decision_manager
from api.services.maintenance import admission, restoring
from api.services.model_downloads import model_manager
from api.services.release import identity
from api.services.runtime import runtime_manager


def authorize(x_kadan_host: str = Header(default='')):
    path = os.environ.get('KADAN_HOST_KEY_FILE')
    expected = Path(path).read_text().strip() if path else ''
    if not expected or not hmac.compare_digest(expected, x_kadan_host):
        raise HTTPException(403, 'Host helper authorization required')


router = APIRouter(prefix='/internal/updates', dependencies=[Depends(authorize)], include_in_schema=False)
task = None
state = 'idle'
error = None
cancel = False


async def drain_work(timeout=120):
    global state, error
    deadline = time.monotonic() + timeout
    try:
        while admission.count():
            if cancel:
                state = 'cancelled'
                return
            if time.monotonic() >= deadline:
                raise TimeoutError('Active work did not finish; nothing was restarted.')
            await asyncio.sleep(.05)
        if cancel:
            state = 'cancelled'
            return
        state = 'cleaning'
        # This task is never cancelled by an HTTP timeout. A stuck cleanup keeps
        # admission closed and is reported to the helper; it must not stop Docker.
        await decision_manager.close()
        await runtime_manager.close()
        await asyncio.to_thread(model_manager.close)
        state = 'drained'
    except TimeoutError:
        state, error = 'busy', 'Active work did not finish; nothing was restarted.'
    except Exception as exc:
        state, error = 'error', f'Drain failed ({type(exc).__name__}); no restart is safe.'


@router.post('/drain', status_code=202)
async def drain():
    global task, state, error, cancel
    if task is None or task.done():
        admission.seal()
        state, error, cancel = 'waiting', None, False
        task = asyncio.create_task(drain_work())
    return status()


@router.get('/drain')
def status():
    return {'state': state, 'active': admission.count(), 'error': error}


@router.post('/resume')
async def resume():
    global cancel, state
    cancel = True
    if task is not None and not task.done():
        raise HTTPException(409, 'Drain cleanup is still running; wait before resuming.')
    if state == 'error':
        raise HTTPException(409, 'Cleanup failed; inspect the server before resuming.')
    marker = os.environ.get('KADAN_MAINTENANCE_FILE')
    if marker and Path(marker).exists():
        raise HTTPException(409, 'Host maintenance marker is still present')
    admission.resume()
    state = 'idle'
    await runtime_manager.start()
    return {'state': 'resumed'}


@router.post('/prepare')
async def prepare():
    """Restore the saved LLM while public admission remains sealed."""
    token = restoring.set(True)
    try:
        await runtime_manager.start()
    finally:
        restoring.reset(token)
    return runtime_manager.status()


@router.get('/ready')
def ready():
    """Certify build + DB schema; expose model readiness separately and honestly."""
    try:
        config = Config(str(Path(__file__).parents[2] / 'alembic.ini'))
        expected = ScriptDirectory.from_config(config).get_heads()
        with get_engine().connect() as connection:
            actual = list(connection.execute(text('SELECT version_num FROM alembic_version')).scalars())
        if sorted(actual) != sorted(expected):
            raise RuntimeError('Migration revision mismatch')
    except Exception as exc:
        raise HTTPException(503, 'Database is unavailable or migrations do not match') from exc
    return {**identity(), 'ready': True, 'model': runtime_manager.status()['state']}
