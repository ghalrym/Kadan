"""Independent installed Diffusers CPU block reference, with explicit BF16 rounding."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import torch
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock, QwenImage21Rope


def save(path,model,bfloat):
    header={};data=bytearray()
    for name,value in model.state_dict().items():
        value=value.detach().to(torch.bfloat16 if bfloat else torch.float32).contiguous();raw=value.view(torch.uint8).numpy().tobytes();start=len(data);data.extend(raw)
        header['transformer_blocks.0.'+name]=dict(dtype='BF16' if bfloat else 'F32',shape=list(value.shape),data_offsets=[start,len(data)])
    encoded=json.dumps(header).encode();path.write_bytes(struct.pack('<Q',len(encoded))+encoded+data)


def main(binary):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);rows=[]
    for seed in (31,109):
        torch.manual_seed(seed);model=QwenImage21TransformerBlock(dim=16,num_attention_heads=2,attention_head_dim=8,mlp_ratio=3).float().eval()
        with torch.no_grad():
            for name,value in model.named_parameters():
                value.normal_(0,.15)
                if 'norm_' in name:value.add_(1)
        original={k:v.clone() for k,v in model.state_dict().items()}
        for bf16 in (False,True):
            model.load_state_dict({k:v.to(torch.bfloat16).float() if bf16 else v for k,v in original.items()})
            with tempfile.TemporaryDirectory() as directory,torch.no_grad():
                root=Path(directory);save(root/'weights.safetensors',model,bf16)
                for text,height,width in ((1,2,2),(3,2,3),(4,3,2)):
                    tokens=text+height*width;x=torch.randn(1,tokens,16);mod=torch.randn(2,64)*.3;mask=torch.arange(tokens)>=text
                    rope=QwenImage21Rope(theta=10000,axes_dim=[2,2,4])(img_shapes=[(1,height,width)],image_pad_mask=mask,device='cpu')
                    expected=model(x,mod,rotary_emb=rope,target_token_mask=mask,segments=[(0,text,True)])
                    (root/'config').write_text(f'16 2 8 48 2 2 4 {text} {height} {width}')
                    (root/'input').write_bytes(x.numpy().tobytes());(root/'mod').write_bytes(mod.numpy().tobytes());out=root/'out'
                    reply=json.loads(subprocess.check_output([binary,str(root),'weights.safetensors',str(root/'config'),str(root/'input'),str(root/'mod'),str(out)]));actual=torch.frombuffer(bytearray(out.read_bytes()),dtype=torch.float32).reshape_as(expected);out.unlink()
                    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5);assert reply['resident_bytes']==0 and not reply['full_image_generation']
                    # No future text/image leakage into the first causal text row.
                    changed=x.clone();changed[:,1:]+=torch.randn_like(changed[:,1:])*2;(root/'input').write_bytes(changed.numpy().tobytes())
                    subprocess.check_output([binary,str(root),'weights.safetensors',str(root/'config'),str(root/'input'),str(root/'mod'),str(out)])
                    altered=torch.frombuffer(bytearray(out.read_bytes()),dtype=torch.float32).reshape_as(expected);out.unlink();torch.testing.assert_close(actual[:,0],altered[:,0],atol=0,rtol=0)
                    rows.append(dict(seed=seed,dtype='BF16' if bf16 else 'F32',text=text,height=height,width=width,max_absolute_error=float((actual-expected).abs().max()),causal_prefix_exact=True))
    print(json.dumps(dict(case_count=len(rows),cases=rows,gpu_execution=False,real_checkpoint_validated=False),indent=2))
if __name__=='__main__':main(sys.argv[1])
