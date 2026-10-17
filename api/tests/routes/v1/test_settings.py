import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from pydantic import ValidationError

from api.pydantic_models.inference_task import ImageSelectionTask, WhisperFormattingTask
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


def snapshot(selected_image="qwen-image-2.1-turbo", tier=None):
    return ModelStoreStatus.model_validate(
        {
            "models": [
                checkpoint("qwen3.8-27b-q4-k-m", "llm", "downloading"),
                checkpoint("superwhisper-s1-mini", "formatting", "complete"),
                checkpoint("qwen-image-2.1", "image"),
                checkpoint("qwen-image-2.1-turbo", "image"),
            ],
            "tiers": [
                {
                    "id": "go",
                    "name": "Go",
                    "vram_gb": 8,
                    "ram_gb": 32,
                    "models": {
                        "llm": "qwen3.8-27b-q4-k-m",
                        "image": None,
                        "video": None,
                        "speech": None,
                        "transcription": None,
                        "formatting": "superwhisper-s1-mini",
                    },
                }
            ],
            "selected_tier": tier,
            "selected_model_id": "qwen3.8-27b-q4-k-m" if tier else None,
            "whisper_formatting": True,
            "selected_image": selected_image,
        }
    )


class SettingsTests(unittest.IsolatedAsyncioTestCase):
    def test_tiers_report_assigned_checkpoints_and_progress(self):
        with patch.object(settings.inference, "models_status", return_value=snapshot()):
            result = settings.get_settings()
        self.assertIsNone(result.tier)
        tier = result.tiers[0]
        self.assertEqual((tier.id, tier.vram_gb, tier.ram_gb), ("go", 8, 32))
        models = {entry.modality: entry.model for entry in tier.models}
        self.assertEqual(
            list(models),
            ["llm", "image", "video", "speech", "transcription", "formatting"],
        )
        self.assertEqual(
            (models["llm"].id, models["llm"].status, models["llm"].downloaded_bytes),
            ("qwen3.8-27b-q4-k-m", "downloading", 5),
        )
        self.assertEqual(models["formatting"].status, "complete")
        self.assertIsNone(models["image"])

    async def test_tier_selection_saves_and_queues_downloads(self):
        body = settings.SettingsRequest(tier="go", license_acknowledged=True)
        with (
            patch.object(
                settings.inference, "models_status", return_value=snapshot(tier="go")
            ),
            patch.object(settings.inference, "select_tier", new=AsyncMock()) as select,
            patch.object(
                settings.inference, "download_tier", new=AsyncMock()
            ) as download,
        ):
            result = await settings.update_settings(body)
        select.assert_awaited_once_with("go")
        download.assert_awaited_once_with("go", True)
        self.assertEqual(result.tier, "go")
        language = next(model for model in result.models if model.type == "LLM")
        self.assertEqual(language.selected, "qwen3.8-27b-q4-k-m")

    def test_unknown_tier_and_language_selection_are_rejected(self):
        with self.assertRaises(ValidationError):
            settings.SettingsRequest(tier="max")
        with self.assertRaises(ValidationError):
            settings.SettingsRequest(models={"LLM": "qwen3.8-27b-q4-k-m"})

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
            patch.object(settings.inference, "unload", new=AsyncMock()) as unload,
        ):
            await settings.update_settings(body)
        submit.assert_awaited_once_with(
            ImageSelectionTask(model_id="qwen-image-2.1"),
            feature="image_selection",
            operation="configure",
        )
        unload.assert_not_awaited()

    def test_unknown_image_model_is_rejected(self):
        with self.assertRaises(ValidationError):
            settings.SettingsRequest(models={"Image": "qwen3.8-27b-q4-k-m"})
        with self.assertRaises(ValidationError):
            settings.SettingsRequest(models={"Video": "anything"})

    async def test_formatting_saves_both_states_without_selecting_a_tier(self):
        for enabled in (False, True):
            status = snapshot()
            status.whisper_formatting = enabled
            with (
                patch.object(settings.inference, "models_status", return_value=status),
                patch.object(settings.inference, "submit", new=AsyncMock()) as submit,
                patch.object(
                    settings.inference, "select_tier", new=AsyncMock()
                ) as select,
            ):
                result = await settings.update_settings(
                    settings.SettingsRequest(whisper_formatting=enabled)
                )
            submit.assert_awaited_once_with(
                WhisperFormattingTask(enabled=enabled),
                feature="whisper_formatting",
                operation="configure",
            )
            select.assert_not_awaited()
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
