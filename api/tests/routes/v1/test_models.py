from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from api.pydantic_models.inference_task import ModelContextTask
from api.routes.v1 import models


class SelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_muse_selection_is_submitted_to_existing_queue(self):
        model_id = "muse-spark-open-q3-xl"
        snapshot = SimpleNamespace(
            models=[], selected_model_id=model_id, selected_image=None
        )
        body = models.SelectionRequest(model_id=model_id)
        with (
            patch.object(models.inference, "models_status", return_value=snapshot),
            patch.object(models.inference, "submit", new=AsyncMock()) as submit,
        ):
            result = await models.select_model(body)
        submit.assert_awaited_once_with(
            body, feature="model_selection", operation="configure"
        )
        self.assertEqual(result.selected_model_id, model_id)

    def test_unknown_language_model_is_rejected(self):
        with self.assertRaises(ValidationError):
            models.SelectionRequest(model_id="muse-spark-open-unknown")


class ContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_change_only_persists_and_never_unloads(self):
        snapshot = SimpleNamespace(
            models=[], selected_model_id="small", selected_image=None
        )
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
