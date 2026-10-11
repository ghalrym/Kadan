import unittest
from unittest.mock import AsyncMock, Mock, patch

from api.server import app, lifespan


class LifespanTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_starts_fifo_and_shutdown_closes_owners(self):
        downloads = Mock()
        videos = Mock()
        manager = Mock(start=AsyncMock(), close=AsyncMock())
        with patch('api.server.model_manager', downloads), patch('api.server.video_jobs', videos), patch('api.server.memory_manager', manager):
            async with lifespan(app):
                manager.start.assert_awaited_once()
                manager.close.assert_not_awaited()
                downloads.close.assert_not_called()
                videos.close.assert_not_called()
                manager.llm.load.assert_not_called()
            manager.close.assert_awaited_once()
            downloads.close.assert_called_once()
            videos.close.assert_called_once()
