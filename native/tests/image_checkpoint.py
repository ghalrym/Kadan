"""Explicit installed-checkpoint block reference; no full image inference."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
from safetensors import safe_open
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock, QwenImage21Rope


def main(binary,root):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261010)
    with torch.device('meta'):
        model=QwenImage21TransformerBlock(dim=4096,num_attention_heads=32,attention_head_dim=128,mlp_ratio=3)
    weights={};hashes={};shard='diffusion_pytorch_model-00001-of-00002.safetensors'
    with safe_open(root/shard,framework='pt',device='cpu') as source:
        for name in model.state_dict():
            value=source.get_tensor('transformer_blocks.0.'+name)
            hashes[name]=hashlib.sha256(value.view(torch.uint8).numpy()).hexdigest();weights[name]=value.float()
    model.load_state_dict(weights,assign=True);model.eval();del weights
    x=torch.randn(1,6,4096)*.1;mod=torch.randn(2,16384)*.1;mask=torch.arange(6)>=2
    rope=QwenImage21Rope(theta=10000,axes_dim=[16,56,56])(img_shapes=[(1,2,2)],image_pad_mask=mask,device='cpu')
    with torch.no_grad():expected=model(x,mod,rotary_emb=rope,target_token_mask=mask,segments=[(0,2,True)])
    with tempfile.TemporaryDirectory() as directory:
        temp=Path(directory);(temp/'config').write_text('4096 32 128 12288 16 56 56 2 2 2');(temp/'input').write_bytes(x.numpy().tobytes());(temp/'mod').write_bytes(mod.numpy().tobytes())
        reply=json.loads(subprocess.check_output([str(binary),str(root),shard,str(temp/'config'),str(temp/'input'),str(temp/'mod'),str(temp/'output')]))
        actual=torch.frombuffer(bytearray((temp/'output').read_bytes()),dtype=torch.float32).reshape_as(expected)
        torch.testing.assert_close(actual,expected,atol=2e-4,rtol=2e-4)
    assert reply['resident_bytes']==0 and not reply['full_image_generation']
    print(json.dumps(dict(real_checkpoint_validated=True,block=0,tokens=6,state=4096,maximum_absolute_error=float((actual-expected).abs().max()),rms_error=float(((actual-expected)**2).mean().sqrt()),tensor_sha256=hashes,native_result=reply,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),gpu_execution=False,full_image_generation=False),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
