import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from precision_oracle import short_block
from trace_binding import sha256, verify_binding


class OracleProtocolTests(unittest.TestCase):
    def test_nonuniform_nonzero_full_equations_against_scalar_reference(self):
        # One head, two target rows, one prefix: independent Python scalar math.
        weights={
            'attn.to_q':[[.3,-.7],[.9,.2]],'attn.to_k':[[.8,.1],[-.4,.6]],
            'attn.to_v':[[-.2,.5],[.7,-.3]],'attn.to_out.0':[[.4,.8],[-.6,.3]],
            'img_mlp.gate_layer':[[.7,-.2],[.1,.9]],'img_mlp.proj':[[-.5,.4],[.6,.2]],
            'img_mlp.out':[[.3,-.8],[.5,.7]]}
        hidden=[[.2,-.9],[1.3,.4]];scale1=[.1,-.2];gate1=[.3,-.4]
        scale2=[-.15,.25];gate2=[.55,.2];eps=1e-4
        def linear(x,name):return [sum(a*b for a,b in zip(x,row)) for row in weights[name]]
        def norm(x,rms=False):
            center=0. if rms else sum(x)/len(x)
            z=[v-center for v in x];den=(sum(v*v for v in z)/len(z)+eps)**.5
            return [v/den for v in z]
        def mod(x,scale):return [v*(1+s) for v,s in zip(norm(x),scale)]
        q=[];k=[];v=[]
        coefficients=[(.6,.8),(-.8,.6)]
        # Coefficients are promoted from captured complex64, not ideal decimals.
        rope=torch.tensor([complex(*p) for p in coefficients],dtype=torch.complex64)[:,None]
        for index,row in enumerate(hidden):
            x=mod(row,scale1)
            def projected(name,gain):
                z=[a*b for a,b in zip(norm(linear(x,name),True),gain)]
                c=float(rope[index,0].real);s=float(rope[index,0].imag)
                return [z[0]*c-z[1]*s,z[0]*s+z[1]*c]
            q.append(projected('attn.to_q',[1.1,.9]));k.append(projected('attn.to_k',[.8,1.2]))
            v.append(linear(x,'attn.to_v'))
        keys=[[.3,-.1]]+k;values=[[-.2,.6]]+v;expected=[]
        for row,query in zip(hidden,q):
            logits=[sum(a*b for a,b in zip(query,key))/math.sqrt(2) for key in keys]
            probs=[math.exp(x-max(logits)) for x in logits];total=sum(probs)
            attended=[sum(p/total*x[j] for p,x in zip(probs,values)) for j in range(2)]
            projected=linear(attended,'attn.to_out.0')
            residual=[x+math.tanh(g)*y for x,g,y in zip(row,gate1,projected)]
            x=mod(residual,scale2);g=linear(x,'img_mlp.gate_layer');p=linear(x,'img_mlp.proj')
            down=linear([a/(1+math.exp(-a))*b for a,b in zip(g,p)],'img_mlp.out')
            expected.append([x+math.tanh(g)*y for x,g,y in zip(residual,gate2,down)])
        state={name+'.weight':torch.tensor(w,dtype=torch.float64) for name,w in weights.items()}
        state.update({'attn.norm_q.weight':torch.tensor([1.1,.9],dtype=torch.float64),'attn.norm_k.weight':torch.tensor([.8,1.2],dtype=torch.float64)})
        data=dict(state=state,config=dict(eps=eps,num_attention_heads=1,attention_head_dim=2),
            hidden=torch.tensor([hidden],dtype=torch.float64),
            modulation=torch.tensor([scale1+gate1+scale2+gate2,[.9]*8],dtype=torch.float64),rotary=rope,
            prefix=(torch.tensor([[[[.3,-.1]]]],dtype=torch.float64),torch.tensor([[[[-.2,.6]]]],dtype=torch.float64)),key_valid=None)
        result,_=short_block(data,rows=2)
        torch.testing.assert_close(result,torch.tensor([expected],dtype=torch.float64),atol=2e-14,rtol=2e-14)

    def test_binding_rejects_modified_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'block.pt').write_bytes(b'capture')
            manifest={'blocks':[{}]*7+[{'file':'block.pt','sha256':sha256(root/'block.pt')}]}
            (root/'manifest.json').write_text(json.dumps(manifest))
            (root/'settings.json').write_text('{}')
            binding=dict(protocol='block7-traces-v1',capture_manifest_sha256=sha256(root/'manifest.json'),
                block_sha256=sha256(root/'block.pt'),producer_source_commit='1cc8c05e84d17cfa7e9f730510db01842063b546',backends={})
            for backend in ('default','math'):
                (root/f'{backend}-intermediates.pt').write_bytes(b'tensor')
                (root/f'{backend}-report.json').write_text(json.dumps(dict(backend=backend,operators=[backend])))
                binding['backends'][backend]=dict(backend=backend,operators=[backend],files={name:sha256(root/name)
                    for name in [f'{backend}-intermediates.pt',f'{backend}-report.json','settings.json']})
            (root/'trace-binding.json').write_text(json.dumps(binding))
            verify_binding(root,root)
            (root/'default-intermediates.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Trace hash mismatch'):verify_binding(root,root)


if __name__=='__main__':unittest.main()
