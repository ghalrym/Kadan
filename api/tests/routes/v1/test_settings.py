import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from pydantic import ValidationError

from api.pydantic_models.inference_task import ImageSelectionTask, WhisperFormattingTask
from api.routes.v1 import settings
from api.services.inference_errors import InferenceFailure


def snapshot(selected_image="qwen-image-2.1-turbo"):
    return SimpleNamespace(
        selected_model_id="small",
        whisper_formatting=True,
        selected_image=selected_image,
        models=[
            SimpleNamespace(id="small", kind="llm"),
            SimpleNamespace(id="qwen-image-2.1", kind="image"),
            SimpleNamespace(id="qwen-image-2.1-turbo", kind="image"),
        ],
    )


class SettingsTests(unittest.IsolatedAsyncioTestCase):
    def test_image_selection_comes_from_worker(self):
        with patch.object(settings.inference, "models_status", return_value=snapshot()):
            result = settings.get_settings()
        image = next(model for model in result.models if model.type == "Image")
        self.assertEqual(image.selected, "qwen-image-2.1-turbo")
        self.assertEqual(image.options, ["qwen-image-2.1", "qwen-image-2.1-turbo"])

    async def test_image_selection_is_submitted_to_worker(self):
        body = settings.SettingsRequest(models={"Image": "qwen-image-2.1"})
        with (
            patch.object(
                settings.inference,
                "models_status",
                return_value=snapshot("qwen-image-2.1"),
            ),
            patch.object(settings.inference, "submit", new=AsyncMock()) as submit,
        ):
            await settings.update_settings(body)
        submit.assert_awaited_once_with(
            ImageSelectionTask(model_id="qwen-image-2.1"),
            feature="image_selection",
            operation="configure",
        )

    def test_unknown_image_model_is_rejected(self):
        with self.assertRaises(ValidationError):
            settings.SettingsRequest(models={"Image": "small"})
        with self.assertRaises(ValidationError):
            settings.SettingsRequest(models={"Video": "anything"})

    async def test_gpt_oss_language_models_are_selectable(self):
        for model_id in ("gpt-oss-20b", "medium"):
            body = settings.SettingsRequest(models={"LLM": model_id})
            with (
                patch.object(
                    settings.inference, "models_status", return_value=snapshot()
                ),
                patch.object(
                    settings.inference, "select_model", new=AsyncMock()
                ) as select,
            ):
                await settings.update_settings(body)
            select.assert_awaited_once_with(model_id)
        with self.assertRaises(ValidationError):
            settings.SettingsRequest(models={"LLM": "gpt-oss-7b"})

    async def test_image_selection_leaves_language_model_resident(self):
        body = settings.SettingsRequest(models={"Image": "qwen-image-2.1"})
        with (
            patch.object(settings.inference, "models_status", return_value=snapshot()),
            patch.object(settings.inference, "submit", new=AsyncMock()),
            patch.object(settings.inference, "unload", new=AsyncMock()) as unload,
        ):
            await settings.update_settings(body)
        unload.assert_not_awaited()

    async def test_muse_selection_uses_existing_settings_and_catalog(self):
        model_id = "muse-spark-open-q3-xl"
        status = snapshot()
        status.selected_model_id = model_id
        status.models.append(SimpleNamespace(id=model_id, kind="llm"))
        body = settings.SettingsRequest(models={"LLM": model_id})
        with (
            patch.object(settings.inference, "models_status", return_value=status),
            patch.object(settings.inference, "select_model", new=AsyncMock()) as select,
        ):
            result = await settings.update_settings(body)
        select.assert_awaited_once_with(model_id)
        language_model = next(model for model in result.models if model.type == "LLM")
        self.assertEqual(language_model.selected, model_id)
        self.assertEqual(language_model.options, ["small", model_id])

    async def test_formatting_saves_both_states_without_selecting_a_model(self):
        for enabled in (False, True):
            status = snapshot()
            status.whisper_formatting = enabled
            with (
                patch.object(settings.inference, "models_status", return_value=status),
                patch.object(settings.inference, "submit", new=AsyncMock()) as submit,
                patch.object(
                    settings.inference, "select_model", new=AsyncMock()
                ) as select,
            ):
                result = await settings.update_settings(
                    settings.SettingsRequest(models={}, whisper_formatting=enabled)
                )
            submit.assert_awaited_once_with(
                WhisperFormattingTask(enabled=enabled),
                feature="whisper_formatting",
                operation="configure",
            )
            select.assert_not_awaited()
            self.assertEqual(result.whisper_formatting, enabled)

    async def test_omitted_formatting_leaves_saved_preference_unchanged(self):
        status = snapshot()
        status.whisper_formatting = False
        with (
            patch.object(settings.inference, "models_status", return_value=status),
            patch.object(settings.inference, "submit", new=AsyncMock()) as submit,
        ):
            result = await settings.update_settings(settings.SettingsRequest(models={}))
        submit.assert_not_awaited()
        self.assertFalse(result.whisper_formatting)

    async def test_formatting_save_failure_is_reported(self):
        with (
            patch.object(
                settings.inference,
                "submit",
                new=AsyncMock(side_effect=InferenceFailure("Cannot save setting", 503)),
            ),
            self.assertRaises(HTTPException) as caught,
        ):
            await settings.update_settings(
                settings.SettingsRequest(models={}, whisper_formatting=False)
            )
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(caught.exception.detail, "Cannot save setting")
