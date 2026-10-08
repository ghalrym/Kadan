"""CPU contracts only: does not certify real-weight or CUDA numerical parity."""
import unittest
from unittest.mock import patch

import torch
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock

import metrics
from replay import eager, module, inputs


class HarnessTests(unittest.TestCase):
    def test_fixed_threshold_metrics_and_reference_at_worst(self):
        reference=torch.tensor([0.,1.,2.,-4.])
        actual=reference+torch.tensor([.021,.03125,0.,0.])
        result=metrics.compare(actual,reference)
        self.assertEqual(result['violations'],1)
        self.assertEqual(result['worst_flat_index'],1)
        self.assertEqual(result['reference_magnitude_at_worst'],1.)
        self.assertAlmostEqual(result['max_abs_error'],.03125)
        self.assertGreater(result['max_normalized_tolerance_ratio'],1.)
        self.assertGreater(result['relative_l2'],0.)

    def test_comparison_streaming_and_nonfinite_rejection(self):
        a=torch.arange(21).float();b=a.clone();b[9]+=1
        whole=metrics.compare(a,b)
        with patch.object(metrics,'CHUNK',3):
            chunked=metrics.compare(a,b)
        for key in whole:
            if isinstance(whole[key],float):self.assertAlmostEqual(whole[key],chunked[key])
            else:self.assertEqual(whole[key],chunked[key])
        failed=metrics.compare(torch.tensor([float('nan')]),torch.zeros(1))
        self.assertEqual(failed['nonfinite'],1);self.assertEqual(failed['violations'],1)

    def test_trained_state_loading_and_short_input_global_positions(self):
        torch.manual_seed(12)
        original=QwenImage21TransformerBlock(32,4,8).eval()
        copied=module(QwenImage21TransformerBlock,dict(dim=32,num_attention_heads=4,attention_head_dim=8),
            original.state_dict(),'cpu',torch.float32)
        data=dict(hidden=torch.randn(1,10,32),modulation=torch.randn(2,128),
            rotary=torch.polar(torch.ones(10,4),torch.randn(10,4)),
            prefix=(torch.randn(1,3,4,8),torch.randn(1,3,4,8)),key_valid=torch.ones(1,13,dtype=torch.bool))
        args=inputs(data,'cpu',torch.float32,rows=4)
        torch.testing.assert_close(args[2],data['rotary'][:4],rtol=0,atol=0)
        self.assertEqual(args[-1].shape,(1,7))
        self.assertIsNone(inputs(data,'cpu',torch.float32,include_hidden=False)[0])
        with torch.no_grad():
            torch.testing.assert_close(eager(original,*args),eager(copied,*args),rtol=0,atol=0)

    def test_chain_keeps_output_between_blocks_instead_of_resetting(self):
        torch.manual_seed(9)
        blocks=[QwenImage21TransformerBlock(32,4,8).eval() for _ in range(4)]
        value=torch.randn(1,4,32);first=value.clone()
        mod=torch.randn(2,128);rope=torch.ones(4,4,dtype=torch.complex64)
        prefix=(torch.randn(1,3,4,8),torch.randn(1,3,4,8))
        with torch.no_grad():
            for block in blocks:value=eager(block,value,mod,rope,prefix,None)
            reset=eager(blocks[-1],first,mod,rope,prefix,None)
        self.assertFalse(torch.equal(value,reset))


if __name__=='__main__':unittest.main()
