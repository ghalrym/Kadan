import unittest

import torch

from api.inference.quantization import (
    dequantize_mxfp4, dequantize_nvfp4, mxfp4_linear, nvfp4_linear, fp8_linear,
)


class QuantizationTests(unittest.TestCase):
    def test_all_nibbles_signs_and_order(self):
        codes = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2, dtype=torch.uint8).reshape(1, 1, 16)
        actual = dequantize_mxfp4(codes, torch.tensor([[127]], dtype=torch.uint8))
        values = [0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.]
        torch.testing.assert_close(actual, torch.tensor([values * 2]))

    def test_mxfp4_block_scales_and_reference_linear(self):
        blocks = torch.full((2, 2, 16), 0x22, dtype=torch.uint8)
        scales = torch.tensor([[126, 128], [127, 129]], dtype=torch.uint8)
        expected = torch.tensor([[.5] * 32 + [2.] * 32, [1.] * 32 + [4.] * 32])
        torch.testing.assert_close(dequantize_mxfp4(blocks, scales), expected)
        inputs = torch.arange(64, dtype=torch.float32).reshape(1, 64)
        bias = torch.tensor([1., -1.])
        # Force one row per tile; exercise actual linear with bias.
        torch.testing.assert_close(mxfp4_linear(inputs, blocks, scales, bias, scratch_bytes=64*64), inputs @ expected.T + bias)
        with self.assertRaises(MemoryError):
            mxfp4_linear(inputs, blocks, scales, scratch_bytes=1)

    def test_nvfp4_modelopt_e4m3_and_global(self):
        weights = torch.full((2, 16), 0xA2, dtype=torch.uint8)
        scales = torch.tensor([[1., 2.], [.5, 4.]], dtype=torch.float32).to(torch.float8_e4m3fn)
        globals_ = torch.tensor([2., .5])
        expected = torch.tensor([[2., -2.] * 8 + [4., -4.] * 8, [.25, -.25] * 8 + [2., -2.] * 8])
        torch.testing.assert_close(dequantize_nvfp4(weights, scales, globals_), expected)
        torch.testing.assert_close(dequantize_nvfp4(weights, scales.view(torch.uint8), globals_), expected)
        inputs = torch.arange(32, dtype=torch.float32)
        torch.testing.assert_close(nvfp4_linear(inputs, weights, scales, globals_, scratch_bytes=32*64), inputs @ expected.T)

    def test_compressed_tensors_int32_reciprocal_global(self):
        weights = torch.full((1, 2), 0x22222222, dtype=torch.int32)
        scales = torch.ones((1, 1)).to(torch.float8_e4m3fn)
        expected = torch.full((1, 16), .25)
        torch.testing.assert_close(dequantize_nvfp4(weights, scales, torch.tensor(4.), layout='compressed-tensors'), expected)

    def test_fp8_linear_row_scales(self):
        weight = torch.tensor([[1., 2.], [3., 4.]]).to(torch.float8_e4m3fn)
        output = fp8_linear(torch.tensor([[1., 2.]]), weight, torch.tensor([2., .5]), scratch_bytes=128)
        torch.testing.assert_close(output, torch.tensor([[10., 5.5]]))

    def test_reject_wrong_layout_and_preserve_nan_scale(self):
        with self.assertRaises(ValueError):
            dequantize_mxfp4(torch.zeros((1, 8), dtype=torch.uint8), torch.zeros(1, dtype=torch.uint8))
        with self.assertRaises(ValueError):
            dequantize_nvfp4(torch.zeros((1, 8), dtype=torch.uint8), torch.ones((1, 1)), torch.ones(1))
        self.assertTrue(torch.isnan(dequantize_mxfp4(torch.ones((1, 1, 16), dtype=torch.uint8), torch.tensor([[255]], dtype=torch.uint8))).all())
