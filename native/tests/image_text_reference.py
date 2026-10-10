"""Qwen3-VL text-only conditioner parity at the pre-final-norm feature boundary."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel


def save(root,model,bfloat):
    for shard in range(2):
        header={};data=bytearray()
        for i,(name,value) in enumerate(model.state_dict().items()):
            if i%2!=shard:continue
            value=value.detach().to(torch.bfloat16 if bfloat else torch.float32).contiguous();raw=value.view(torch.uint8).numpy().tobytes();start=len(data);data.extend(raw)
            header['model.language_model.'+name]=dict(dtype='BF16' if bfloat else 'F32',shape=list(value.shape),data_offsets=[start,len(data)])
        encoded=json.dumps(header).encode();(root/f'{shard}.safetensors').write_bytes(struct.pack('<Q',len(encoded))+encoded+data)


def main(binary):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);rows=[]
    for seed in (37,103):
        torch.manual_seed(seed);cfg=Qwen3VLTextConfig(vocab_size=32,hidden_size=16,intermediate_size=48,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,rope_theta=5000000,rope_scaling={'rope_type':'default','mrope_section':[1,1,2],'mrope_interleaved':True});cfg._attn_implementation='eager';model=Qwen3VLTextModel(cfg).float().eval()
        with torch.no_grad():
            for name,value in model.named_parameters():
                value.normal_(0,.12)
                if 'norm.weight' in name:value.add_(1)
        handle=model.norm.register_forward_hook(lambda module,args,output:args[0])
        original={k:v.clone() for k,v in model.state_dict().items()}
        for bf16 in (False,True):
            model.load_state_dict({k:v.to(torch.bfloat16).float() if bf16 else v for k,v in original.items()})
            with tempfile.TemporaryDirectory() as directory,torch.no_grad():
                root=Path(directory);save(root,model,bf16);(root/'config').write_text('32 16 2 2 1 8 48')
                for count in (1,3,7):
                    ids=torch.randint(0,32,(1,count));expected=model(input_ids=ids,use_cache=False).last_hidden_state;(root/'ids').write_bytes(ids.numpy().astype('<u4').tobytes());out=root/'out'
                    result=json.loads(subprocess.check_output([binary,str(root),str(root/'config'),str(root/'ids'),str(out),'0.safetensors','1.safetensors']))
                    actual=torch.frombuffer(bytearray(out.read_bytes()),dtype=torch.float32).reshape_as(expected);out.unlink();torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5);assert result['resident_bytes']==0 and result['pre_final_norm']
                    changed=ids.clone();changed[:,1:]=(changed[:,1:]+7)%32;(root/'ids').write_bytes(changed.numpy().astype('<u4').tobytes());subprocess.check_output([binary,str(root),str(root/'config'),str(root/'ids'),str(out),'0.safetensors','1.safetensors']);altered=torch.frombuffer(bytearray(out.read_bytes()),dtype=torch.float32).reshape_as(expected);out.unlink();torch.testing.assert_close(actual[:,0],altered[:,0],atol=0,rtol=0)
                    rows.append(dict(seed=seed,dtype='BF16' if bf16 else 'F32',tokens=count,max_absolute_error=float((actual-expected).abs().max()),causal_prefix_exact=True))
        handle.remove()
    print(json.dumps(dict(case_count=len(rows),cases=rows,gpu_execution=False,pre_final_norm=True,full_image_generation=False),indent=2))
if __name__=='__main__':main(sys.argv[1])
