import unittest
from dataclasses import replace
from unittest.mock import patch

from fastapi import HTTPException

from pydantic import ValidationError

from api.routes.v1.settings import SettingsRequest, get_settings
from api.services.model_catalog import CATALOG


class SettingsValidationTests(unittest.TestCase):
    def test_settings_reject_model_from_another_category(self):
        with self.assertRaises(ValidationError):
            SettingsRequest(models={"LLM": "F5-TTS"})


class LiveSettingsTests(unittest.TestCase):
    def test_get_has_real_selection_and_no_media_defaults(self):
        from unittest.mock import patch
        from api.routes.v1.settings import get_settings
        with patch('api.routes.v1.settings.model_manager') as manager:
            manager.status.return_value = {'selected_model_id': None}
            result = get_settings()
        self.assertIsNone(result.models[0].selected)
        self.assertEqual(result.models[0].options, ['small', 'medium', 'large'])
        self.assertIsNone(result.whisper_formatting)

    def test_all_non_llm_kinds_are_excluded(self):
        catalog = {'small': CATALOG['small']}
        for kind in ('video', 'speech', 'transcription', 'formatting', 'image'):
            catalog[kind] = replace(CATALOG['small'], id=kind, kind=kind)
        with patch('api.routes.v1.settings.CATALOG', catalog), patch(
            'api.routes.v1.settings.model_manager'
        ) as manager:
            manager.status.return_value = {'selected_model_id': 'small'}
            result = get_settings()
        self.assertEqual(result.models[0].options, ['small'])
        self.assertEqual(result.models[0].selected, 'small')

    def test_non_llm_or_unknown_stored_selection_is_unavailable(self):
        for selected in ('h3-fl2va', 'unknown'):
            with self.subTest(selected=selected), patch(
                'api.routes.v1.settings.model_manager'
            ) as manager:
                manager.status.return_value = {'selected_model_id': selected}
                with self.assertRaises(HTTPException) as raised:
                    get_settings()
                self.assertEqual(raised.exception.status_code, 503)

    def test_update_uses_model_store(self):
        from unittest.mock import patch
        from api.routes.v1.settings import update_settings
        with patch('api.routes.v1.settings.model_manager') as manager:
            manager.status.return_value = {'selected_model_id': 'small'}
            result = update_settings(SettingsRequest(models={'LLM': 'small'}))
            manager.select.assert_called_once_with('small')
        self.assertEqual(result.models[0].selected, 'small')

    def test_formatting_does_not_mutate_selection(self):
        from unittest.mock import patch
        from fastapi import HTTPException
        from api.routes.v1.settings import update_settings
        with patch('api.routes.v1.settings.model_manager') as manager:
            with self.assertRaises(HTTPException) as raised:
                update_settings(SettingsRequest(models={'LLM': 'small'}, whisper_formatting=True))
            self.assertEqual(raised.exception.status_code, 503)
            manager.select.assert_not_called()

    def test_chat_history_contains_no_demo_messages(self):
        from api.routes.v1.chat.messages import list_messages
        self.assertEqual(list_messages().messages, [])
