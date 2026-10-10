"""Complete tiny Qwen Image denoiser reference across split shards and timesteps."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import torch
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel


def save(root,model,bfloat):
    for shard in range(2):
        header={};data=bytearray()
        for i,(name,value) in enumerate(model.state_dict().items()):
            if i%2!=shard:continue
            value=value.detach().to(torch.bfloat16 if bfloat else torch.float32).contiguous();raw=value.view(torch.uint8).numpy().tobytes();start=len(data);data.extend(raw)
            header[name]=dict(dtype='BF16' if bfloat else 'F32',shape=list(value.shape),data_offsets=[start,len(data)])
        encoded=json.dumps(header).encode();(root/f'{shard}.safetensors').write_bytes(struct.pack('<Q',len(encoded))+encoded+data)


def main(binary):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);rows=[]
    for seed in (53,131):
        torch.manual_seed(seed);model=QwenImage21Transformer2DModel(in_channels=4,out_channels=4,num_layers=2,attention_head_dim=8,num_attention_heads=2,context_in_dim=16,mlp_ratio=3,axes_dims_rope=(2,2,4),causal_condition=True).float().eval()
        with torch.no_grad():
            for name,value in model.named_parameters():
                value.normal_(0,.12)
                if '.attn.norm_' in name:value.add_(1)
        original={k:v.clone() for k,v in model.state_dict().items()}
        for bf16 in (False,True):
            model.load_state_dict({k:v.to(torch.bfloat16).float() if bf16 else v for k,v in original.items()})
            with tempfile.TemporaryDirectory() as directory,torch.no_grad():
                root=Path(directory);save(root,model,bf16)
                for text,height,width in ((1,2,2),(3,2,4),(4,4,2)):
                    latent=torch.randn(1,height*width,4);condition=torch.randn(1,text,16);mask=torch.cat((torch.zeros(1,text,dtype=torch.bool),torch.ones(1,height*width//4,dtype=torch.bool)),dim=1)
                    (root/'latent').write_bytes(latent.numpy().tobytes());(root/'condition').write_bytes(condition.numpy().tobytes())
                    for timestep in (0.,.37,1.):
                        expected=model(hidden_states=latent,encoder_hidden_states=condition,timestep=torch.tensor([timestep]),img_shapes=[[(1,height,width)]],img_mask=mask,return_dict=False)[0][:,-height*width:]
                        (root/'config').write_text(f'16 2 8 48 2 2 4 2 16 4 {text} {height} {width} {timestep}');out=root/'out'
                        reply=json.loads(subprocess.check_output([binary,str(root),str(root/'config'),str(root/'latent'),str(root/'condition'),str(out),'0.safetensors','1.safetensors']))
                        actual=torch.frombuffer(bytearray(out.read_bytes()),dtype=torch.float32).reshape_as(expected);out.unlink()
                        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5);assert reply['resident_bytes']==0 and reply['full_denoiser'] and not reply['full_image_generation']
                        rows.append(dict(seed=seed,dtype='BF16' if bf16 else 'F32',text=text,height=height,width=width,timestep=timestep,max_absolute_error=float((actual-expected).abs().max())))
    print(json.dumps(dict(case_count=len(rows),cases=rows,gpu_execution=False,full_image_generation=False),indent=2))
if __name__=='__main__':main(sys.argv[1])
