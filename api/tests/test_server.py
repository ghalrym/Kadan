import asyncio
import threading
import unittest
from unittest.mock import AsyncMock, Mock, patch

from api.server import app, lifespan


class LifespanTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_restores_model_and_shutdown_closes_owners(self):
        runtime = Mock(start=AsyncMock(), close=AsyncMock())
        downloads = Mock()
        speech = Mock()
        with patch('api.server.runtime_manager', runtime), patch('api.server.model_manager', downloads), patch('api.server.speech_runtime', speech):
            async with lifespan(app):
                runtime.start.assert_awaited_once()
                runtime.close.assert_not_awaited()
            runtime.close.assert_awaited_once()
            downloads.close.assert_called_once()
            speech.unload.assert_called_once()

    async def test_repeated_shutdown_cancellation_waits_for_speech_cleanup(self):
        started, release = threading.Event(), threading.Event()
        runtime = Mock(start=AsyncMock(), close=AsyncMock())
        downloads = Mock()
        def unload():
            started.set()
            release.wait(5)
        speech = Mock(unload=Mock(side_effect=unload))
        with patch('api.server.runtime_manager', runtime), patch('api.server.model_manager', downloads), patch('api.server.speech_runtime', speech):
            async def run():
                async with lifespan(app):
                    pass
            task = asyncio.create_task(run())
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                runtime.close.assert_not_awaited()
                downloads.close.assert_not_called()
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            runtime.close.assert_awaited_once()
            downloads.close.assert_called_once()
