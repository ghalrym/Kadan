import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class QwenDispatchTests(unittest.TestCase):
    def test_all_official_modes_use_complete_waveform_methods(self):
        audio = SimpleNamespace(ndim=1)
        modules = {'numpy': Mock(), 'soundfile': SimpleNamespace(read=lambda *a, **k: (audio, 24000)),
                   'torch': Mock(), 'qwen_tts': Mock()}
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location('qwen_test_worker', Path(__file__).parents[2] / 'workers/qwen_tts.py')
            worker = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(worker)
        model = Mock()
        for size in ('0.6b', '1.7b'):
            worker.generate(model, {'script': 'hello', 'language': 'English', 'voice': {'mode': 'custom', 'speaker': 'Ryan'}})
            model.generate_custom_voice.assert_called_with(text='hello', language='English', non_streaming_mode=True, speaker='Ryan', instruct='')
            worker.generate(model, {'script': 'hello', 'language': 'English', 'voice': {'mode': 'clone', 'sample': 'UklGRg==', 'transcript': 'reference'}})
            model.generate_voice_clone.assert_called_with(text='hello', language='English', non_streaming_mode=True, ref_audio=(audio, 24000), ref_text='reference', x_vector_only_mode=False)
        worker.generate(model, {'script': 'hello', 'language': 'Auto', 'voice': {'mode': 'describe', 'description': 'warm'}})
        model.generate_voice_design.assert_called_with(text='hello', language='Auto', non_streaming_mode=True, instruct='warm')
