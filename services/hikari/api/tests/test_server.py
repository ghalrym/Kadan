import unittest
from unittest.mock import AsyncMock, Mock, patch

from api.server import app, lifespan


class LifespanTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_starts_fifo_and_shutdown_closes_owners(self):
        manager = Mock(start=AsyncMock(), close=AsyncMock())
        with patch("api.server.inference", manager):
            async with lifespan(app):
                manager.start.assert_awaited_once()
                manager.close.assert_not_awaited()
            manager.close.assert_awaited_once()
