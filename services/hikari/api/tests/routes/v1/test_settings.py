import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from pydantic import ValidationError

from api.pydantic_models.inference_task import WhisperFormattingTask
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


def snapshot(tier=None):
    return ModelStoreStatus.model_validate(
        {
            "models": [
                checkpoint("qwen3.8-27b-q4-k-m", "llm", "downloading"),
                checkpoint("superwhisper-s1-mini", "formatting", "complete"),
                checkpoint("qwen-image-2.1-turbo", "image"),
                checkpoint("qwen-tts-1.7b-custom", "speech"),
                checkpoint("whisper-medium", "transcription"),
            ],
            "tiers": [
                {
                    "id": "go",
                    "name": "Go",
                    "vram_gb": 8,
                    "ram_gb": 32,
                    "video_defaults": {"duration": 5, "resolution": "480p"},
                    "models": {
                        "llm": "qwen3.8-27b-q4-k-m",
                        "image": "qwen-image-2.1-turbo",
                        "video": None,
                        "speech": "qwen-tts-1.7b-custom",
                        "transcription": "whisper-medium",
                        "formatting": "superwhisper-s1-mini",
                    },
                }
            ],
            "selected_tier": tier,
            "selected_model_id": "qwen3.8-27b-q4-k-m" if tier else None,
            "whisper_formatting": True,
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
        self.assertEqual(models["image"].id, "qwen-image-2.1-turbo")
        self.assertEqual(models["speech"].id, "qwen-tts-1.7b-custom")
        self.assertEqual(models["transcription"].id, "whisper-medium")
        self.assertIsNone(models["video"])
        self.assertEqual(tier.video_defaults.duration, 5)
        self.assertEqual(tier.video_defaults.resolution, "480p")

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

    def test_unknown_tier_and_model_selections_are_rejected(self):
        with self.assertRaises(ValidationError):
            settings.SettingsRequest(tier="max")
        # The tier decides every model; there is no per-model selection.
        for models in (
            {"LLM": "qwen3.8-27b-q4-k-m"},
            {"Image": "qwen-image-2.1-turbo"},
        ):
            with self.assertRaises(ValidationError):
                settings.SettingsRequest(models=models)

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
