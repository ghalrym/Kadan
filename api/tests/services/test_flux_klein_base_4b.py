"""Native FLUX.2 klein calls and persisted selection without real model weights."""
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from api.inference import flux_klein, native_image
from api.inference.resources import ResourceManager
from api.services.images import ImageManager
from api.services.model_catalog import CATALOG, allowed_asset, validate_assets
from api.services.model_downloads import ModelManager


class KleinBase4BTests(unittest.TestCase):
    def test_distilled_pipeline_receives_generation_and_edit_recipe(self):
        recipe = flux_klein.RECIPES['flux-klein-base-4b']
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'weights.safetensors').write_bytes(b'fixture')
            source = Image.new('RGB', (3, 3))
            pipeline = Mock(return_value=SimpleNamespace(images=[source]))
            factory = Mock(return_value=pipeline)
            torch = SimpleNamespace(float32='fp32', bfloat16='bf16', Generator=Mock())
            modules = lambda: (torch, SimpleNamespace(Flux2KleinPipeline=SimpleNamespace(from_pretrained=factory)))
            resources = ResourceManager(100 * 1024**3, {})
            for image in (None, source):
                native_image.generate(path, resources, 'Tree', '1:1', [7], threading.Event(),
                    device='cpu', image=image, modules=modules, recipe=recipe)
                kwargs = pipeline.call_args.kwargs
                self.assertEqual(kwargs['num_inference_steps'], 50)
                self.assertEqual(kwargs['guidance_scale'], 4.0)
                self.assertEqual((kwargs['width'], kwargs['height']), (1024, 1024))
                self.assertEqual(kwargs.get('image'), image)
                self.assertTrue(factory.call_args.kwargs['local_files_only'])
                self.assertFalse(resources.snapshot()['reservations'])

