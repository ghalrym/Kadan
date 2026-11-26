import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from pydantic import ValidationError

from api.pydantic_models.inference_task import LlmSelectionTask, WhisperFormattingTask
from api.pydantic_models.model_store import ModelStoreStatus
from api.routes.v1 import settings
from api.services.inference_errors import InferenceFailure


def checkpoint(model_id, kind, status="not_downloaded"):
    return {
        "id": model_id,
        "repo_id": "test/model",
        "revision": "pinned",
        "license": "MIT",
        "estimated_bytes": 20,
        "kind": kind,
        "display_name": model_id,
        "license_url": None,
        "license_notice": None,
        "inference_available": True,
        "status": status,
        "downloaded_bytes": 5 if status == "downloading" else 0,
        "total_bytes": 20 if status == "downloading" else 0,
        "error": None,
        "context_limit": None,
        "architecture_context_limit": None,
    }


def snapshot():
    return ModelStoreStatus.model_validate(
        {
            "models": [
                checkpoint("inkling-2-mini-iq4-nl", "llm", "downloading"),
                checkpoint("superwhisper-s1-mini", "formatting", "complete"),
                checkpoint("qwen-image-2.1-turbo", "image"),
                checkpoint("qwen-tts-1.7b-custom", "speech"),
                checkpoint("whisper-medium", "transcription"),
            ],
            "whisper_formatting": True,
        }
    )


class SettingsTests(unittest.IsolatedAsyncioTestCase):
    def test_fixed_assignments_report_progress_without_selection(self):
        with patch.object(settings.inference, "models_status", return_value=snapshot()):
            result = settings.get_settings()
        self.assertEqual(result.defaults.llm, "inkling-2-mini-iq4-nl")
        self.assertEqual(result.defaults.speech, "qwen-tts-1.7b-custom")
        self.assertEqual(result.defaults.formatting, "superwhisper-s1-mini")
        self.assertEqual(result.models[0].downloaded_bytes, 5)
        self.assertEqual(result.video_defaults.duration, 5)
        self.assertEqual(result.video_defaults.resolution, "480p")
        self.assertNotIn("tiers", result.model_dump())

    def test_unknown_tier_and_model_selections_are_rejected(self):
        with self.assertRaises(ValidationError):
            settings.SettingsRequest(tier="go")
        for models in (
            {"LLM": "inkling-2-mini-iq4-nl"},
            {"Image": "qwen-image-2.1-turbo"},
        ):
            with self.assertRaises(ValidationError):
                settings.SettingsRequest(models=models)

    async def test_formatting_saves_both_states(self):
        for enabled in (False, True):
            status = snapshot()
            status.whisper_formatting = enabled
            with (
                patch.object(settings.inference, "models_status", return_value=status),
                patch.object(settings.inference, "submit", new=AsyncMock()) as submit,
            ):
                result = await settings.update_settings(
                    settings.SettingsRequest(whisper_formatting=enabled)
                )
            submit.assert_awaited_once_with(
                WhisperFormattingTask(enabled=enabled),
                feature="whisper_formatting",
                operation="configure",
            )
            self.assertEqual(result.whisper_formatting, enabled)

    async def test_omitted_settings_are_unchanged(self):
        status = snapshot()
        status.whisper_formatting = False
        with (
            patch.object(settings.inference, "models_status", return_value=status),
            patch.object(settings.inference, "submit", new=AsyncMock()) as submit,
        ):
            result = await settings.update_settings(settings.SettingsRequest())
        submit.assert_not_awaited()
        self.assertFalse(result.whisper_formatting)

    async def test_save_failure_is_reported(self):
        with (
            patch.object(
                settings.inference,
                "submit",
                new=AsyncMock(side_effect=InferenceFailure("Cannot save setting", 503)),
            ),
            self.assertRaises(HTTPException) as caught,
        ):
            await settings.update_settings(
                settings.SettingsRequest(whisper_formatting=False)
            )
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(caught.exception.detail, "Cannot save setting")

    async def test_llm_model_variant_saves_and_rejects_unknown_ids(self):
        with self.assertRaises(ValidationError):
            settings.SettingsRequest(llm_model="qwen-7b")
        for model_id in (
            "inkling-2-mini-iq4-nl",
            "inkling-2-mini-q8-0",
            "inkling-2-mini-abliterated-iq4-nl",
            "inkling-2-mini-abliterated-q8-0",
        ):
            status = snapshot()
            status.selected_model_id = model_id
            status.defaults.llm = model_id
            with (
                patch.object(settings.inference, "models_status", return_value=status),
                patch.object(settings.inference, "submit", new=AsyncMock()) as submit,
            ):
                result = await settings.update_settings(
                    settings.SettingsRequest(llm_model=model_id)
                )
            submit.assert_awaited_once_with(
                LlmSelectionTask(model_id=model_id),
                feature="llm_selection",
                operation="configure",
            )
            self.assertEqual(result.defaults.llm, model_id)
