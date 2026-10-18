"""Worker machines connect here; each carries one worker per inference type."""

from fastapi import APIRouter, WebSocket

from api.services.inference import inference
from api.services.worker_hub import authorized

router = APIRouter(tags=["Workers"])


@router.websocket("/v1/workers")
async def workers(websocket: WebSocket):
    """Registers a worker machine that sends `Authorization: Bearer <HIKARI_WORKER_TOKEN>`."""
    if not authorized(websocket):
        # Closing before accept answers the upgrade with 403.
        await websocket.close(code=1008)
        return
    await websocket.accept()
    await inference.worker.serve(websocket)
