import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from api.pydantic_models.inference_task import ChatTask, InferenceTask, TranscriptionTask
from api.pydantic_models.model_store import ModelAssignments, ModelStoreStatus
from api.routes.v1.chat.completions import CompletionRequest
from api.routes.v1.settings import SettingsRequest, get_settings
from api.services import transcription
from api.services.inference_errors import InferenceFailure
from api.services.worker_hub import WorkerHub, target_worker


def checkpoint(model_id, kind):
    return {
        'id': model_id, 'repo_id': 'unit/model', 'revision': 'pinned', 'license': 'unit',
        'estimated_bytes': 10, 'kind': kind, 'display_name': model_id, 'license_url': None,
        'license_notice': None, 'inference_available': True, 'status': 'not_downloaded',
        'downloaded_bytes': 0, 'total_bytes': 10, 'error': None, 'context_limit': None,
        'architecture_context_limit': None,
    }


class FixedStackTests(unittest.TestCase):
    def test_default_chat_selects_structured_lfm_without_feature_flags(self):
        body = CompletionRequest(messages=[{'role': 'user', 'content': 'Hello'}], tools=[])
        task = InferenceTask(operation='completion', input={'feature': 'llm', **body.model_dump(exclude_unset=True)})
        self.assertEqual(task.input.model, 'lfm3-1.7b-bf16')
        self.assertEqual(target_worker(task), 'lfm3_llm')
        self.assertEqual(task.model_dump()['input']['model'], 'lfm3-1.7b-bf16')

    def test_removed_language_model_is_rejected_instead_of_substituted(self):
        for model in ('qwen3.6-35b-a3b-q4-k-m', '', 'other'):
            with self.subTest(model=model), self.assertRaises(ValidationError):
                ChatTask(model=model, messages=[{'role': 'user', 'text': 'Hello'}])

    def test_tier_settings_are_no_longer_accepted(self):
        with self.assertRaises(ValidationError):
            SettingsRequest(tier='go', license_acknowledged=True)
        with self.assertRaises(ValidationError):
            InferenceTask(operation='configure', input={'feature': 'tier_selection', 'tier': 'go'})

    def test_fixed_assignments_do_not_fabricate_connected_checkpoints(self):
        hub = WorkerHub()
        native = ModelStoreStatus(models=[checkpoint('qwen-image-2.1-turbo', 'image'), checkpoint('qwen-tts-1.7b-custom', 'speech')])
        hub.primary = 'image'
        hub.workers = {'image': SimpleNamespace(models=native, kind='image')}
        hub._refresh_models()
        self.assertEqual([row.id for row in hub.models.models], ['qwen-image-2.1-turbo'])
        self.assertEqual(hub.models.selected_model_id, 'lfm3-1.7b-bf16')
        self.assertEqual(hub.models.defaults.vision, 'lfm3-vl-2b-bf16')
        with patch('api.routes.v1.settings.inference.models_status', return_value=hub.models):
            result = get_settings()
        self.assertEqual(len(result.models), 1)
        self.assertNotIn('tiers', result.model_dump())
        self.assertEqual((result.video_defaults.duration, result.video_defaults.resolution), (5, '480p'))

    def test_old_language_download_cannot_reach_any_worker(self):
        hub = WorkerHub()
        task = InferenceTask(operation='download', input={'feature': 'models', 'model_id': 'qwen3.6-35b-a3b-q4-k-m'})
        with self.assertRaisesRegex(InferenceFailure, 'fixed stack'):
            hub._store_targets(task)


class LfmFormattingTests(unittest.IsolatedAsyncioTestCase):
    def raw(self):
        return {'text': 'hello there', 'raw_text': 'hello there', 'language': 'en',
                'model': 'medium', 'formatting_status': 'requested', 'formatting_model': None}

    async def test_formatter_uses_fixed_lfm_queue_and_preserves_whisper_raw(self):
        raw = self.raw()
        with patch.object(transcription.inference, 'submit', new=AsyncMock(side_effect=[raw,
                {'text': 'Hello there.', 'model': 'lfm3-1.7b-bf16', 'finish_reason': 'stop'}])) as submit:
            result = await transcription.transcribe(TranscriptionTask(audio='unit'))
        task = submit.await_args_list[1].args[0]
        self.assertEqual(task.model, 'lfm3-1.7b-bf16')
        self.assertEqual(task.messages[-1].text, raw['raw_text'])
        self.assertEqual(submit.await_args_list[1].kwargs, {'feature': 'llm', 'operation': 'completion'})
        self.assertEqual(result['raw_text'], raw['raw_text'])
        self.assertEqual(result['model'], 'medium')
        self.assertEqual(result['formatting_model'], 'lfm3-1.7b-bf16')

    async def test_incomplete_or_command_output_retains_raw_with_warning(self):
        for text, finish in [('', 'stop'), ('Partial', 'length'), ('<tool_call>bad', 'stop')]:
            raw = self.raw()
            with self.subTest(text=text), patch.object(transcription.inference, 'submit',
                    new=AsyncMock(side_effect=[raw, {'text': text, 'model': 'lfm3-1.7b-bf16', 'finish_reason': finish}])):
                result = await transcription.transcribe(TranscriptionTask(audio='unit'))
            self.assertEqual(result['text'], raw['raw_text'])
            self.assertEqual(result['formatting_status'], 'failed')
            self.assertTrue(result['formatting_error'])
