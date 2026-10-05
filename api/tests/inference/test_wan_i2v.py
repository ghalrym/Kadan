import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, MagicMock, patch

import torch

from api.inference.video import VideoSpec
from api.inference.wan import WanProvider


class WanI2VTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.image = Path(self.temp.name) / 'image.png'
        self.image.write_bytes(b'fixture image; decoding is in the worker')
        self.store = Mock()
        self.store.get_checkpoint.return_value = (types.SimpleNamespace(kind='video'), Path(self.temp.name))
        self.provider = WanProvider('wan22-i2v-a14b', 'i2v-A14B', manager=self.store, python=sys.executable)

    def test_requires_image_and_native_controls_before_queueing(self):
        with self.assertRaisesRegex(ValueError, 'requires an uploaded image'):
            self.provider.validate(VideoSpec(prompt='test', fps=16))
        for overrides in ({'fps': 24}, {'resolution': '1080p'}, {'aspect': '1:1'}, {'audio_path': '/audio'}):
            values = dict(prompt='test', fps=16, image_path=str(self.image))
            values.update(overrides)
            with self.assertRaises(ValueError): self.provider.validate(VideoSpec(**values))
        self.provider.validate(VideoSpec(prompt='test', fps=16, image_path=str(self.image)))
        self.store.get_checkpoint.assert_called_once_with('wan22-i2v-a14b')

    def test_worker_passes_image_to_wan_i2v_with_exact_sampler(self):
        config = types.SimpleNamespace(sample_shift=5.0, sample_steps=40, sample_guide_scale=(3.5, 3.5))
        native = Mock()
        native.generate.return_value = torch.zeros((3, 17, 16, 32))
        factory = Mock(return_value=native)
        wan = types.ModuleType('wan')
        wan.WanI2V, wan.WanTI2V, wan.WanT2V = factory, Mock(), Mock()
        configs = types.ModuleType('wan.configs')
        configs.WAN_CONFIGS = {'i2v-A14B': config}
        image_api, image_ops = MagicMock(), MagicMock()
        decoded = image_api.open.return_value.__enter__.return_value
        fitted = image_ops.fit.return_value
        pil = types.ModuleType('PIL')
        pil.Image, pil.ImageOps = image_api, image_ops
        imageio = MagicMock()
        payload = {'task': 'i2v-A14B', 'checkpoint': '/completed/checkpoint', 'output': '/output.mp4',
                   'width': 32, 'height': 16,
                   'spec': dict(prompt='motion', negative_prompt='blur', seed=9, duration=1,
                                fps=16, resolution='480p', image_path=str(self.image))}
        modules = {'wan': wan, 'wan.configs': configs, 'PIL': pil, 'imageio': imageio}
        with patch.dict(sys.modules, modules), patch('sys.stdin', io.StringIO(json.dumps(payload))):
            namespace = runpy.run_path(str(Path(__file__).parents[2] / 'inference/workers/wan_worker.py'))
            namespace['main']()
        factory.assert_called_once()
        wan.WanT2V.assert_not_called()
        image_api.open.assert_called_once_with(str(self.image))
        image_ops.exif_transpose.assert_called_once_with(decoded)
        args = native.generate.call_args.kwargs
        self.assertIs(args['img'], fitted)
        self.assertNotIn('size', args)
        self.assertEqual(args['max_area'], 512)
        self.assertEqual(args['frame_num'], 17)
        self.assertEqual((args['shift'], args['sampling_steps'], args['guide_scale'], args['sample_solver']),
                         (3.0, 40, (3.5, 3.5), 'unipc'))
        self.assertEqual(imageio.get_writer.return_value.__enter__.return_value.append_data.call_count, 16)
