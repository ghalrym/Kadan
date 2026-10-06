import unittest
from unittest.mock import AsyncMock, Mock, patch

from api.server import app, lifespan


class LifespanTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_restores_model_and_shutdown_closes_owners(self):
        runtime = Mock(start=AsyncMock(), close=AsyncMock())
        downloads = Mock()
        manager = Mock(start=AsyncMock(), close=AsyncMock())
        with patch('api.server.runtime_manager', runtime), patch('api.server.model_manager', downloads), patch('api.server.memory_manager', manager):
            async with lifespan(app):
                runtime.start.assert_awaited_once()
                runtime.close.assert_not_awaited()
            manager.close.assert_awaited_once()
            downloads.close.assert_called_once()
