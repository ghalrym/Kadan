import unittest

import torch

from controlled_projection import ControlledProjection


class ControlledProjectionTests(unittest.TestCase):
    def test_partial_tail_and_row_ownership(self):
        x=(torch.arange(2*3*519).reshape(2,3,519)%7-3).float()
        w=(torch.arange(5*519).reshape(5,519)%5-2).float()
        operation=ControlledProjection(w)
        expected=(x.double()@w.double().t()).float()
        self.assertTrue(torch.equal(operation(x),expected))
        self.assertTrue(torch.equal(torch.cat([operation(x[:,:1]),operation(x[:,1:])],dim=1),expected))
        self.assertEqual(operation.packed_bytes,w.numel()*4)

    def test_bf16_is_not_silently_promoted(self):
        with self.assertRaises(ValueError):ControlledProjection(torch.ones(2,3,dtype=torch.bfloat16))
        op=ControlledProjection(torch.ones(2,3))
        with self.assertRaises(ValueError):op(torch.ones(1,3,dtype=torch.bfloat16))


if __name__=='__main__':unittest.main()
