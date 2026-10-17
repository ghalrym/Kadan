import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from api.pydantic_models.inference_task import ModelContextTask
from api.routes.v1 import models


class ContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_change_only_persists_and_never_unloads(self):
        snapshot = SimpleNamespace(models=[], selected_model_id="small")
        with (
            patch.object(models.inference, "models_status", return_value=snapshot),
            patch.object(models.inference, "submit", new=AsyncMock()) as submit,
            patch.object(models.inference, "unload", new=AsyncMock()) as unload,
        ):
            await models.configure_context(
                "small", models.ContextRequest(context_limit=32768)
            )
        submit.assert_awaited_once_with(
            ModelContextTask(model_id="small", context_limit=32768),
            feature="model_context",
            operation="configure",
        )
        unload.assert_not_awaited()
