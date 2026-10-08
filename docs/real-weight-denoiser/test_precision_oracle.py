import unittest

import torch

from precision_oracle import assess, attention64, rotate_pairs, short_block


class PrecisionOracleTests(unittest.TestCase):
    def test_zero_gates_preserve_hidden_through_complete_equations(self):
        width=4
        state={name+'.weight':torch.eye(width,dtype=torch.float64) for name in
            ('attn.to_q','attn.to_k','attn.to_v','attn.to_out.0','img_mlp.gate_layer','img_mlp.proj','img_mlp.out')}
        state.update({'attn.norm_q.weight':torch.ones(2),'attn.norm_k.weight':torch.ones(2)})
        hidden=torch.arange(12,dtype=torch.float64).reshape(1,3,4)
        data=dict(state=state,config=dict(eps=1e-6,num_attention_heads=2,attention_head_dim=2),
            hidden=hidden,modulation=torch.zeros(2,16),rotary=torch.ones(3,1,dtype=torch.complex64),
            prefix=(torch.zeros(1,1,2,2),torch.zeros(1,1,2,2)),key_valid=None)
        output,_=short_block(data,rows=3)
        self.assertTrue(torch.equal(output,hidden))

    def test_rope_is_pair_rotation_without_float32_roundtrip(self):
        value=torch.tensor([[[[1.+2**-40,2.]]]],dtype=torch.float64)
        result=rotate_pairs(value,torch.tensor([[1j]],dtype=torch.complex64))
        self.assertEqual(result[0,0,0,0],-2.)
        self.assertEqual(result[0,0,0,1],1.+2**-40)

    def test_attention_mask_and_head_ownership(self):
        query=torch.zeros(1,1,2,2,dtype=torch.float64)
        key=torch.zeros(1,2,2,2,dtype=torch.float64)
        value=torch.tensor([[[[2.,4.],[6.,8.]],[[100.,200.],[300.,400.]]]],dtype=torch.float64)
        result=attention64(query,key,value,torch.tensor([[True,False]]))
        self.assertTrue(torch.equal(result,value[:,:1]))
        self.assertFalse(torch.equal(result,attention64(query,key,value,None)))
        self.assertFalse(torch.equal(result,attention64(query,key,value.flip(2),torch.tensor([[True,False]]))))

    def test_virtual_match_cannot_hide_oracle_or_eager_failure(self):
        eager=torch.zeros(1,dtype=torch.float64)
        virtual=torch.ones(1,dtype=torch.float64)
        result=assess(eager,virtual,eager,virtual.clone())
        self.assertEqual(result['distributed_vs_virtual']['violations'],0)
        self.assertEqual(result['virtual_vs_fp64']['violations'],1)
        self.assertEqual(result['original_eager_compatibility']['violations'],1)
        self.assertEqual(result['eager_vs_fp64']['violations'],0)


if __name__=='__main__':unittest.main()
