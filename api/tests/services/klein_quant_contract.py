"""Synthetic complete download-to-native contract shared by checkpoint registrations."""
from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import unquote

from PIL import Image
import torch
from safetensors.torch import save

from api.inference import native_image
from api.inference.flux_klein import RECIPES
from api.inference.resources import ResourceManager
from api.services.images import ImageManager
from api.services.model_catalog import CATALOG, allowed_asset
from api.services.model_downloads import ModelManager


def check_checkpoint(test, model_id, *, revision, auxiliary_revision, steps, guidance, pipeline_name, quantization):
    entry, recipe = CATALOG[model_id], RECIPES[model_id]
    test.assertEqual(entry.revision, revision)
    test.assertEqual(recipe.revision, revision)
    test.assertEqual(recipe.quantization, quantization)
    test.assertEqual(recipe.pipeline, pipeline_name)
    test.assertEqual(entry.auxiliary_sources[0].revision, auxiliary_revision)
    test.assertEqual(recipe.steps, steps)
    test.assertEqual(recipe.guidance_scale, guidance)
    test.assertFalse(allowed_asset('transformer/diffusion_pytorch_model.safetensors', entry))
    test.assertFalse(allowed_asset('transformer/diffusion_pytorch_model.safetensors.index.json', entry))
    test.assertFalse(allowed_asset('custom.py', entry))
    test.assertEqual(entry.estimated_bytes, 0)
    test.assertEqual(len(entry.auxiliary_sources), 1)
    auxiliary = entry.auxiliary_sources[0]
    test.assertEqual(set(auxiliary.component_paths), {'tokenizer', 'scheduler', 'text_encoder', 'vae'})
    if quantization == 'fp8':
        state = {'img_in.weight': torch.ones(2, 16).to(torch.float8_e4m3fn)}
    else:
        state = {'img_in.weight': torch.full((2, 8), 0x24, dtype=torch.uint8),
                 'img_in.comfy_quant': torch.tensor(list(b'{"format":"nvfp4"}'), dtype=torch.uint8),
                 'img_in.weight_scale': torch.ones(512).to(torch.float8_e4m3fn),
                 'img_in.weight_scale_2': torch.tensor(1.)}
    primary = {name: b'fixture license' for name in entry.source_files}
    primary[recipe.weight_filename] = save(state)
    aux = {name: b'{}' for name in entry.required_files if name not in primary}
    for component in ('text_encoder', 'vae'):
        aux[f'{component}/model.safetensors'] = b'synthetic companion weights'
    for name in aux:
        if name.endswith('.index.json'):
            aux[name] = json.dumps({'weight_map': {'fixture': 'model.safetensors'}}).encode()
    # Upstream offers a duplicate BF16 transformer and unrelated executable; neither may transfer.
    aux['transformer/diffusion_pytorch_model.safetensors'] = b'forbidden duplicate'
    aux['custom.py'] = b'forbidden executable'
    sources = {(entry.repo_id, entry.revision): primary, (auxiliary.repo_id, auxiliary.revision): aux}
    transferred = []
    def network(url, **_kwargs):
        if '/api/models/' in url:
            repo, pin = url.split('/api/models/', 1)[1].split('/revision/')
            pin = pin.split('?')[0]
            payload = sources[repo, pin]
            return BytesIO(json.dumps({'sha': pin, 'siblings': [
                {'rfilename': name, 'lfs': {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}}
                for name, data in payload.items()]}).encode())
        repo, tail = url.split('huggingface.co/', 1)[1].split('/resolve/')
        pin, name = tail.split('/', 1)
        name = unquote(name)
        transferred.append(name)
        return BytesIO(sources[repo, pin][name])
    with tempfile.TemporaryDirectory() as folder:
        downloads = ModelManager(Path(folder) / 'models')
        with patch('api.services.model_downloads.urlopen', side_effect=network), patch(
            'api.services.model_downloads.open_gated_checkpoint', side_effect=network):
            downloads.start(model_id, license_acknowledged=True)
            downloads._thread.join(5)
        test.assertFalse(downloads._thread.is_alive())
        resolved, path = downloads.get_checkpoint(model_id)
        test.assertEqual(resolved, entry)
        test.assertEqual(set(transferred), set(primary) | (set(aux) - {'custom.py', 'transformer/diffusion_pytorch_model.safetensors'}))
        test.assertFalse((path / 'transformer/diffusion_pytorch_model.safetensors').exists())
        downloads.select_image_model(model_id)
        restored = ModelManager(Path(folder) / 'models')
        test.assertEqual(restored.selected_image_model_id(), model_id)
        test.assertIsNone(restored.status()['selected_model_id'])
        resources = ResourceManager(100 * 1024 ** 3, {})
        transformer = object()
        factory = Mock(return_value=transformer)
        pipeline = Mock(return_value=SimpleNamespace(images=[Image.new('RGB', (2, 2))]))
        pipeline_loader = Mock(return_value=pipeline)
        modules = lambda: (torch, SimpleNamespace(**{pipeline_name: SimpleNamespace(from_pretrained=pipeline_loader),
            'Flux2Transformer2DModel': SimpleNamespace(from_single_file=factory)}))
        generate = native_image.generate
        def native(*args, **kwargs):
            return generate(*args, **kwargs, modules=modules)
        manager = ImageManager(folder, restored, SimpleNamespace(ensure_resources=lambda: resources))
        with patch.dict('os.environ', {'KADAN_IMAGE_DEVICE': 'cpu'}), patch('api.services.images.native_image.generate', side_effect=native):
            result = manager.generate('Tree', '1:1', 1, 7, threading.Event())
        test.assertIn(model_id, result.meta)
        test.assertTrue(manager.file(result.id, 0).is_file())
        test.assertIs(pipeline_loader.call_args.kwargs['transformer'], transformer)
        test.assertTrue(pipeline_loader.call_args.kwargs['local_files_only'])
        test.assertEqual(pipeline_loader.call_args.args[0], str(path))
        test.assertEqual(factory.call_args.kwargs['config'], str(path))
        test.assertTrue(factory.call_args.kwargs['local_files_only'])
        test.assertEqual(factory.call_args.kwargs['subfolder'], 'transformer')
        test.assertEqual(factory.call_args.args[0]['img_in.weight'].dtype, torch.float32)
        test.assertEqual(pipeline.call_args.kwargs['num_inference_steps'], steps)
        test.assertEqual(pipeline.call_args.kwargs.get('guidance_scale'), guidance)
        if guidance is None:
            test.assertNotIn('guidance_scale', pipeline.call_args.kwargs)
        test.assertFalse(resources.snapshot()['reservations'])
