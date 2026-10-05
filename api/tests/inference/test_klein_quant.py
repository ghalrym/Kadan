import gc
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import weakref

import torch
from safetensors.torch import save_file

from api.inference.klein_quant import decode_state_dict, dense_size, swizzled_scales
from api.inference.native_image import ImageRecipe, generate
from api.inference.resources import ResourceManager, ResourceExhausted


def descriptor(fmt, **extra):
    return torch.tensor(list(json.dumps(dict(format=fmt, **extra)).encode()), dtype=torch.uint8)


class QuantizedKleinTests(unittest.TestCase):
    def test_scaled_fp8_and_column_smoothing_match_linear_algebra(self):
        weight = torch.tensor([[1, 2, -1, .5], [2, 1, 0, -2]], dtype=torch.float8_e4m3fn)
        state = {'layer.weight': weight, 'layer.comfy_quant': descriptor('float8_e4m3fn'),
                 'layer.weight_scale': torch.tensor([2., 3.]),
                 'layer.pre_quant_scale': torch.tensor([1., .5, 2., 4.]),
                 'layer.input_scale': torch.tensor(1.), 'layer.bias': torch.tensor([1., -1.])}
        dense = decode_state_dict(state, 'fp8', torch.float32)
        expected = weight.float() * torch.tensor([[2.], [3.]]) * state['layer.pre_quant_scale']
        torch.testing.assert_close(dense['layer.weight'], expected)
        torch.testing.assert_close(torch.nn.functional.linear(torch.ones(1, 4), dense['layer.weight'], dense['layer.bias']),
                                   torch.tensor([[5., -17.5]]))
        self.assertEqual(set(dense), {'layer.weight', 'layer.bias'})

    def test_swizzle_address_fixture_crosses_row_and_column_tiles(self):
        # Hand-derived cuBLAS addresses for rows0,31,32,127,128 and blockcolumns0,3,4.
        stored = torch.zeros(256 * 8, dtype=torch.uint8)
        expected = {(0, 0): (0, 1.), (31, 3): (499, 2.), (32, 0): (4, 3.),
                    (127, 4): (1020, 4.), (128, 0): (1024, 5.)}
        for address, value in expected.values():
            stored[address] = torch.tensor(value, dtype=torch.float8_e4m3fn).view(torch.uint8)
        scales = swizzled_scales(stored, 129, 80, 0, 129).float()
        for (row, column), (_, value) in expected.items():
            self.assertEqual(scales[row, column].item(), value)

    def test_high_nibble_first_nvfp4_and_global_scale(self):
        # 0x24 stores +1 then +2 in this format, repeating across one16-value block.
        state = {'layer.weight': torch.full((1, 8), 0x24, dtype=torch.uint8),
                 'layer.comfy_quant': descriptor('nvfp4'),
                 'layer.weight_scale': torch.zeros(512, dtype=torch.float8_e4m3fn),
                 'layer.weight_scale_2': torch.tensor(.5),
                 'layer.pre_quant_scale': torch.arange(1, 17).float()}
        state['layer.weight_scale'].view(torch.uint8)[0] = torch.tensor(2., dtype=torch.float8_e4m3fn).view(torch.uint8)
        dense = decode_state_dict(state, 'nvfp4', torch.float32)['layer.weight']
        torch.testing.assert_close(dense, torch.tensor([[1., 2.] * 8]) * torch.arange(1, 17))

    def test_unknown_formats_and_invalid_shapes_fail(self):
        states = [
            {'x.weight': torch.ones(2, 4), 'x.comfy_quant': descriptor('int8')},
            {'x.weight': torch.ones(2, 4).to(torch.float8_e4m3fn), 'x.weight_scale': torch.ones(3)},
            {'x.weight': torch.zeros(2, 8, dtype=torch.uint8)},
            {'x.weight': torch.ones(2, 4), 'x.comfy_quant': descriptor('float8_e4m3fn', rotation=True)},
            {'x.weight': torch.ones(2, 4).to(torch.float8_e4m3fn), 'x.pre_quant_scale': torch.ones(3)},
        ]
        for state in states:
            with self.assertRaises(ValueError):
                decode_state_dict(state, 'fp8')
        with self.assertRaises(ValueError):
            swizzled_scales(torch.zeros(1), 2, 16, 0, 2)

    def test_dense_header_budget_and_local_single_file_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'transformer').mkdir()
            (path / 'transformer/config.json').write_text('{}')
            save_file({'img_in.weight': torch.ones(2, 4).to(torch.float8_e4m3fn)}, str(path / 'quant.safetensors'))
            self.assertEqual(dense_size(path / 'quant.safetensors', 4), 32)
            self.assertEqual(dense_size(path / 'quant.safetensors', 2), 16)
            recipe = ImageRecipe('fixture', 'pin', 'Flux2KleinPipeline', {'1:1': (16, 16)}, 4, 1., 'fp8', 'quant.safetensors')
            transformer = SimpleNamespace()
            factory = Mock(return_value=transformer)
            pipeline = Mock(return_value=SimpleNamespace(images=[object()]))
            pipeline_type = SimpleNamespace(from_pretrained=Mock(return_value=pipeline))
            modules = lambda: (torch, SimpleNamespace(Flux2KleinPipeline=pipeline_type,
                Flux2Transformer2DModel=SimpleNamespace(from_single_file=factory)))
            resources = ResourceManager(20 * 1024 ** 3, {})
            generate(path, resources, 'test', '1:1', [1], threading.Event(), 'cpu', modules=modules, recipe=recipe)
            self.assertIs(pipeline_type.from_pretrained.call_args.kwargs['transformer'], transformer)
            self.assertTrue(factory.call_args.kwargs['local_files_only'])
            self.assertEqual(factory.call_args.kwargs['subfolder'], 'transformer')
            self.assertEqual(factory.call_args.args[0]['img_in.weight'].dtype, torch.float32)
            self.assertFalse(resources.snapshot()['reservations'])
            factory.reset_mock()
            with self.assertRaises(ResourceExhausted):
                generate(path, ResourceManager(1, {}), 'x', '1:1', [1], threading.Event(), 'cpu', modules=modules, recipe=recipe)
            factory.assert_not_called()

    def test_failed_loader_allocations_die_before_reservation_release(self):
        references = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'transformer').mkdir()
            (path / 'transformer/config.json').write_text('{}')
            save_file({'img_in.weight': torch.ones(2, 4).to(torch.float8_e4m3fn)}, str(path / 'quant.safetensors'))
            recipe = ImageRecipe('fixture', 'pin', 'Flux2KleinPipeline', {'1:1': (16, 16)}, 4, 1., 'fp8', 'quant.safetensors')
            def fail(state, **kwargs):
                tensor = state['img_in.weight']
                references.append(weakref.ref(tensor))
                raise RuntimeError('fixture construction failure')
            resources = ResourceManager(20 * 1024 ** 3, {})
            modules = lambda: (torch, SimpleNamespace(Flux2KleinPipeline=Mock(),
                Flux2Transformer2DModel=SimpleNamespace(from_single_file=fail)))
            with self.assertRaises(RuntimeError):
                generate(path, resources, 'test', '1:1', [1], threading.Event(), 'cpu', modules=modules, recipe=recipe)
            gc.collect()
            self.assertIsNone(references[0]())
            self.assertFalse(resources.snapshot()['reservations'])
