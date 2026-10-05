"""Official S2V call contract tested without models or native worker dependencies."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch
import wave

from api.inference.wan import WanProvider
from api.inference.video import VideoSpec
from api.services.model_catalog import CATALOG, allowed_asset, validate_assets
from api.services.wan_s2v_catalog import MODEL_ID, REQUIRED_FILES, REVISION


class S2VWorkerTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = Mock()
        self.config = SimpleNamespace(sample_shift=3, sample_steps=40, sample_guide_scale=4.5)
        self.factory = Mock(return_value=self.pipeline)
        self.writer = MagicMock()
        self.image = MagicMock()
        self.fit = MagicMock()
        self.modules = {
            'imageio': SimpleNamespace(get_writer=Mock(return_value=self.writer)),
            'imageio_ffmpeg': SimpleNamespace(get_ffmpeg_exe=lambda: '/known/ffmpeg'),
            'PIL': SimpleNamespace(Image=SimpleNamespace(open=Mock(return_value=self.image)), ImageOps=SimpleNamespace(fit=self.fit)),
            'torch': SimpleNamespace(inference_mode=lambda: lambda fn: fn),
            'torch.nn': SimpleNamespace(functional=SimpleNamespace(interpolate=MagicMock())),
            'torch.nn.functional': SimpleNamespace(interpolate=MagicMock()),
            'wan': SimpleNamespace(WanS2V=self.factory),
            'wan.configs': SimpleNamespace(WAN_CONFIGS={'s2v-14B': self.config}),
        }
        self.modules["torch"].nn = self.modules["torch.nn"]
        self.modules["torch.nn"].functional = self.modules["torch.nn.functional"]
        with patch.dict(sys.modules, self.modules):
            path = Path(__file__).parents[2] / 'inference/workers/wan_s2v.py'
            spec = importlib.util.spec_from_file_location('s2v_test_worker', path)
            self.worker = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.worker)
        self.payload = dict(spec=dict(prompt='Singing', negative_prompt='', duration=1, fps=16, seed=42),
                            checkpoint='/complete/s2v', width=854, height=480, output='/output.mp4')

    def test_native_image_and_audio_conditioning_and_sampler(self):
        self.worker.generate_frames(self.payload, '/uploaded/image.png', '/uploaded/audio.wav')
        self.factory.assert_called_once_with(config=self.config, checkpoint_dir='/complete/s2v',
            device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True, convert_model_dtype=True)
        self.pipeline.generate.assert_called_once_with(input_prompt='Singing',
            ref_image_path='/uploaded/image.png', audio_path='/uploaded/audio.wav', enable_tts=False,
            tts_prompt_audio=None, tts_prompt_text=None, tts_text=None, num_repeat=1, pose_video=None,
            max_area=854 * 480, infer_frames=16, shift=3, sample_solver='unipc', sampling_steps=40,
            guide_scale=4.5, n_prompt='', seed=42, offload_model=True, init_first_frame=False)

    def test_audio_is_trimmed_and_mux_failure_propagates(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / 'reference.wav'
            with wave.open(str(audio), 'wb') as output:
                output.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                output.writeframes(b'\0\0' * 32000)
            self.payload['spec'].update(image_path='/image.png', audio_path=str(audio))
            video = MagicMock(shape=(3, 16, 480, 854))
            def generate(payload, image_path, audio_path):
                with wave.open(str(audio_path)) as trimmed:
                    self.assertEqual(trimmed.getnframes(), 16000)
                return video
            with patch.object(self.worker, 'generate_frames', side_effect=generate), patch.object(
                    self.worker.subprocess, 'run', side_effect=RuntimeError('mux failed')) as mux:
                with self.assertRaisesRegex(RuntimeError, 'mux failed'):
                    self.worker.render(self.payload)
                self.assertTrue(mux.call_args.kwargs['check'])
                self.assertIn('1:a:0', mux.call_args.args[0])
            self.assertEqual(self.writer.__enter__.return_value.append_data.call_count, 16)


class S2VValidationTests(unittest.TestCase):
    def test_inputs_and_native_frame_rate_are_required_before_queueing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / 'speech.wav'
            with wave.open(str(audio), 'wb') as output:
                output.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                output.writeframes(b'\0\0' * 16000)
            manager = Mock()
            manager.get_checkpoint.return_value = (SimpleNamespace(kind='video'), root)
            provider = WanProvider('wan22-s2v-14b', 's2v-14B', manager=manager, python=sys.executable)
            for settings in ({}, {'fps': 16}, {'fps': 16, 'image_path': '/reference.png'}):
                with self.assertRaises(ValueError):
                    provider.validate(VideoSpec('hello', **settings))
            provider.validate(VideoSpec('hello', fps=16, duration=1, image_path='/reference.png', audio_path=str(audio)))
            with self.assertRaisesRegex(ValueError, 'at least as long'):
                provider.validate(VideoSpec('hello', fps=16, duration=2, image_path='/reference.png', audio_path=str(audio)))
            with self.assertRaisesRegex(ValueError, 'Pose video'):
                provider.validate(VideoSpec('hello', fps=16, image_path='/reference.png', audio_path=str(audio), video_path='/pose.mp4'))

    def test_catalog_includes_audio_conditioning_weights_without_duplicates(self):
        entry = CATALOG[MODEL_ID]
        self.assertEqual(entry.revision, REVISION)
        self.assertTrue(all(allowed_asset(name, entry) for name in REQUIRED_FILES))
        validate_assets(entry, set(REQUIRED_FILES))
        with self.assertRaises(ValueError):
            validate_assets(entry, set(REQUIRED_FILES) - {'wav2vec2-large-xlsr-53-english/model.safetensors'})
        for name in ('wav2vec2-large-xlsr-53-english/pytorch_model.bin', 'wav2vec2-large-xlsr-53-english/flax_model.msgpack', 'eval.py'):
            self.assertFalse(allowed_asset(name, entry))
