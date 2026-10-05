import unittest
from unittest.mock import AsyncMock, Mock, patch

from api.server import app, lifespan


class LifespanTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_restores_model_and_shutdown_closes_owners(self):
        runtime = Mock(start=AsyncMock(), close=AsyncMock())
        downloads = Mock()
        with patch('api.server.runtime_manager', runtime), patch('api.server.model_manager', downloads):
            async with lifespan(app):
                runtime.start.assert_awaited_once()
                runtime.close.assert_not_awaited()
            runtime.close.assert_awaited_once()
            downloads.close.assert_called_once()
