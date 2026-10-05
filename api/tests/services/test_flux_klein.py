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


class KleinTests(unittest.TestCase):
    def test_distilled_pipeline_receives_generation_and_edit_recipe(self):
        recipe = flux_klein.RECIPES['flux-klein-4b']
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
                self.assertEqual(kwargs['num_inference_steps'], 4)
                self.assertEqual(kwargs['guidance_scale'], 1.0)
                self.assertEqual((kwargs['width'], kwargs['height']), (1024, 1024))
                self.assertEqual(kwargs.get('image'), image)
                self.assertTrue(factory.call_args.kwargs['local_files_only'])
                self.assertFalse(resources.snapshot()['reservations'])

    def test_selection_survives_restart_and_never_changes_llm(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = ModelManager(Path(folder))
            with patch.object(manager, '_checkpoint_complete', side_effect=lambda entry: entry.kind == 'image'):
                manager.select_image_model('flux-klein-4b')
                self.assertEqual(manager.status()['selected_image_model_id'], 'flux-klein-4b')
                self.assertIsNone(manager.status()['selected_model_id'])
                with self.assertRaises(ValueError): manager.select_image_model('small')
                with self.assertRaises(ValueError): manager.select_image_model('h3-fl2va')
            restored = ModelManager(Path(folder))
            with patch.object(restored, '_checkpoint_complete', return_value=True):
                self.assertEqual(restored.selected_image_model_id(), 'flux-klein-4b')
            self.assertIsNone(restored.selected_image_model_id())
            (Path(folder) / 'image-selection.json').write_text('{broken')
            self.assertIsNone(restored.selected_image_model_id())

    def test_saved_selection_and_explicit_override_route_correct_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            downloads = Mock()
            downloads.selected_image_model_id.return_value = 'flux-klein-4b'
            downloads.get_checkpoint.side_effect = lambda model: (CATALOG[model], Path(folder))
            backend = Mock(return_value=[Image.new('RGB', (2, 2))])
            manager = ImageManager(folder, downloads, Mock(), backend)
            result = manager.generate('Tree', '1:1', 1, 7, threading.Event())
            downloads.get_checkpoint.assert_called_with('flux-klein-4b')
            self.assertIn('flux-klein-4b', result.meta)
            manager.generate('Tree', '1:1', 1, 7, threading.Event(), model='qwen-image-2.1')
            downloads.get_checkpoint.assert_called_with('qwen-image-2.1')

    def test_only_complete_component_bundle_is_accepted(self):
        entry = CATALOG['flux-klein-4b']
        files = set(entry.required_files) | {f'{component}/model.safetensors' for component in entry.weight_paths}
        validate_assets(entry, files)
        with self.assertRaises(ValueError): validate_assets(entry, files - {'text_encoder/model.safetensors'})
        self.assertFalse(allowed_asset('flux-2-klein-4b.safetensors', entry))
        self.assertTrue(allowed_asset('LICENSE.md', entry))
        self.assertTrue(allowed_asset('transformer/diffusion_pytorch_model.safetensors', entry))
