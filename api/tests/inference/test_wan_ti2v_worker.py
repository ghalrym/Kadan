"""Exercise text and image native call signatures with CPU tensor fixtures."""
import io
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch


class TI2VWorkerTests(unittest.TestCase):
    def test_native_sampler_and_image_conditioning(self):
        config = SimpleNamespace(sample_shift=5.0, sample_steps=50, sample_guide_scale=5.0)
        pipeline = Mock()
        pipeline.generate.return_value = torch.zeros((3, 25, 32, 32))
        factory = Mock(return_value=pipeline)
        writer = Mock()
        writer.__enter__ = Mock(return_value=writer)
        writer.__exit__ = Mock(return_value=False)
        source_image = MagicMock()
        pillow = SimpleNamespace(Image=SimpleNamespace(open=Mock(return_value=source_image)),
            ImageOps=SimpleNamespace(fit=lambda image, size: SimpleNamespace(size=size)))
        modules = {'PIL': pillow, 'wan': SimpleNamespace(WanTI2V=factory),
                   'wan.configs': SimpleNamespace(WAN_CONFIGS={'ti2v-5B': config}),
                   'imageio': SimpleNamespace(get_writer=Mock(return_value=writer))}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'reference.png'
            source.write_bytes(b'image fixture')
            payload = dict(task='ti2v-5B', checkpoint='local-checkpoint', output='out.mp4', width=32, height=16,
                spec=dict(prompt='forest', negative_prompt='blur', duration=1, fps=24, seed=19, image_path=str(source)))
            with patch.dict('sys.modules', modules), patch('sys.stdin', io.StringIO(json.dumps(payload))):
                worker = runpy.run_path(str(Path(__file__).parents[2] / 'inference/workers/wan_worker.py'))
                worker['main']()
        kwargs = pipeline.generate.call_args.kwargs
        self.assertEqual(kwargs['img'].size, (32, 32))
        self.assertEqual((kwargs['shift'], kwargs['sampling_steps'], kwargs['guide_scale']), (5, 50, 5))
        self.assertEqual((kwargs['frame_num'], kwargs['seed'], kwargs['n_prompt']), (25, 19, 'blur'))
        self.assertTrue(kwargs['offload_model'])
        self.assertEqual(writer.append_data.call_count, 24)
        self.assertEqual(writer.append_data.call_args.args[0].shape, (16, 32, 3))
