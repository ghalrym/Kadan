import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from api.routes.v1 import workers


class WorkerRouteTests(unittest.IsolatedAsyncioTestCase):
    def socket(self, token):
        return SimpleNamespace(
            headers={"authorization": f"Bearer {token}"},
            accept=AsyncMock(),
            close=AsyncMock(),
        )

    async def test_wrong_token_is_closed_before_accepting(self):
        socket = self.socket("wrong")
        with (
            patch.dict(os.environ, {"HIKARI_WORKER_TOKEN": "secret"}),
            patch.object(workers.inference.worker, "serve", new=AsyncMock()) as serve,
        ):
            await workers.workers(socket)
        socket.close.assert_awaited_once_with(code=1008)
        socket.accept.assert_not_awaited()
        serve.assert_not_awaited()

    async def test_matching_token_is_served_by_the_hub(self):
        socket = self.socket("secret")
        with (
            patch.dict(os.environ, {"HIKARI_WORKER_TOKEN": "secret"}),
            patch.object(workers.inference.worker, "serve", new=AsyncMock()) as serve,
        ):
            await workers.workers(socket)
        socket.accept.assert_awaited_once()
        serve.assert_awaited_once_with(socket)
