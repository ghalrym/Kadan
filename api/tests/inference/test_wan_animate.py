"""Animate preprocessing/inference contracts without weights or GPU execution."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from api.inference.resources import ResourceCancelled
from api.inference.video import VideoSpec
from api.inference.wan import WanProvider
from api.inference.wan_animate import WanAnimateProvider


def worker_fixture():
    config = SimpleNamespace(frame_num=77, sample_shift=5.0, sample_steps=20, sample_guide_scale=1.0)
    pipeline = MagicMock()
    output = MagicMock()
    output.shape = (3, 30, 720, 1280)
    pipeline.generate.return_value = output
    wan = SimpleNamespace(__file__='/native/wan/__init__.py', WanAnimate=MagicMock(return_value=pipeline))
    imageio = MagicMock()
    fixtures = {'wan': wan, 'wan.configs': SimpleNamespace(WAN_CONFIGS={'animate-14B': config}),
        'torch': SimpleNamespace(inference_mode=lambda: lambda function: function),
        'imageio': imageio, 'imageio_ffmpeg': SimpleNamespace(get_ffmpeg_exe=lambda: '/runtime/ffmpeg'),
        'decord': SimpleNamespace(VideoReader=lambda path: [None] * 30),
        'PIL': SimpleNamespace(Image=MagicMock(), ImageOps=MagicMock())}
    location = Path(__file__).resolve().parents[2] / 'inference/workers/wan_animate.py'
    definition = importlib.util.spec_from_file_location('animate_worker_fixture', location)
    worker = importlib.util.module_from_spec(definition)
    with patch.dict(sys.modules, fixtures):
        definition.loader.exec_module(worker)
    return worker, wan, pipeline, imageio


class WanAnimateTests(unittest.TestCase):
    def test_validation_requires_both_real_inputs(self):
        provider = WanAnimateProvider(python='/usr/bin/python3')
        with self.assertRaisesRegex(ValueError, 'image and driving video'):
            provider.validate(VideoSpec('A dancer', fps=30))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input'
            source.write_bytes(b'fixture')
            valid = VideoSpec('A dancer', fps=30, image_path=str(source), video_path=str(source))
            with patch.object(provider, '_checkpoint'):
                provider.validate(valid)
                with self.assertRaisesRegex(ValueError, '30 fps'):
                    provider.validate(VideoSpec('A dancer', fps=16, image_path=str(source), video_path=str(source)))

    def test_parent_owns_scratch_until_cancel_cleanup(self):
        provider = WanAnimateProvider()
        directories = []
        def fail(*args):
            directory = Path(provider.scratch_directory)
            self.assertTrue(directory.is_dir())
            (directory / 'source.png').write_bytes(b'fixture')
            directories.append(directory)
            raise ResourceCancelled('cancel')
        with patch.object(WanProvider, 'generate', side_effect=fail):
            with self.assertRaises(ResourceCancelled):
                provider.generate(VideoSpec('A dancer'), Path('/tmp/out.mp4'), None)
        self.assertFalse(directories[0].exists())

    def test_preprocessing_selects_real_mode_and_local_auxiliary_models(self):
        for mode in ('animate', 'replace'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                worker, wan, _, _ = worker_fixture()
                root = Path(directory)
                wan.__file__ = str(root / 'wan/__init__.py')
                script = root / 'wan/modules/animate/preprocess/preprocess_data.py'
                script.parent.mkdir(parents=True)
                script.write_text('# fixture')
                spec = VideoSpec('A dancer', duration=1, fps=30, image_path='/inputs/image.png',
                                 video_path='/inputs/video.mp4', animation_mode=mode)
                payload = dict(spec=vars(spec), checkpoint='/models/local', width=1280, height=720)
                calls = []
                def execute(command, **kwargs):
                    calls.append(command)
                    self.assertTrue(kwargs['check'])
                    if '--save_path' in command:
                        target = Path(command[command.index('--save_path') + 1])
                        target.mkdir()
                        for name in ('src_ref.png', 'src_pose.mp4', 'src_face.mp4', 'src_bg.mp4', 'src_mask.mp4'):
                            (target / name).write_bytes(b'fixture')
                with patch.object(worker.subprocess, 'run', side_effect=execute):
                    worker.prepare_inputs(payload, root)
                self.assertEqual(len(calls), 2)
                self.assertIn('/models/local/process_checkpoint', calls[1])
                self.assertIn('--replace_flag' if mode == 'replace' else '--retarget_flag', calls[1])
                self.assertNotIn('--use_flux', calls[1])

    def test_native_modes_sampler_and_output(self):
        for mode in ('animate', 'replace'):
            with self.subTest(mode=mode):
                worker, wan, pipeline, imageio = worker_fixture()
                payload = dict(spec=vars(VideoSpec('A dancer', duration=1, fps=30, animation_mode=mode)),
                    checkpoint='/models/local', width=1280, height=720,
                    scratch_directory='/scratch/local', output='/tmp/out.mp4')
                with patch.object(worker, 'prepare_inputs', return_value=Path('/scratch/processed')):
                    worker.run(payload)
                self.assertEqual(wan.WanAnimate.call_args.kwargs['use_relighting_lora'], mode == 'replace')
                options = pipeline.generate.call_args.kwargs
                self.assertEqual(options['src_root_path'], '/scratch/processed')
                self.assertEqual(options['replace_flag'], mode == 'replace')
                self.assertEqual((options['sampling_steps'], options['shift'], options['guide_scale']), (20, 5.0, 1.0))
                self.assertEqual(options['clip_len'], 77)
                self.assertEqual(imageio.get_writer.return_value.__enter__.return_value.append_data.call_count, 30)

    def test_failed_preprocessing_never_loads_generation_model(self):
        worker, wan, _, _ = worker_fixture()
        with patch.object(worker, 'prepare_inputs', side_effect=RuntimeError('missing mask')):
            with self.assertRaisesRegex(RuntimeError, 'missing mask'):
                worker.run(dict(spec=vars(VideoSpec('A dancer')), scratch_directory='/scratch'))
        wan.WanAnimate.assert_not_called()
