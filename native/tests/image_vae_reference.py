import json,subprocess,sys,tempfile
from pathlib import Path
import torch
from safetensors.torch import save_file
from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21

def main(binary):
 torch.set_num_threads(1);torch.set_num_interop_threads(1);cases=[]
 for seed in (47,109):
  torch.manual_seed(seed);model=AutoencoderKLQwenImage21(base_dim=4,decoder_base_dim=4,z_dim=2,dim_mult=[1,2,4,8,8],num_res_blocks=1,temperal_downsample=[False,True,True,True],in_channels=4,out_channels=4,is_residual=True,latents_mean=[0,0],latents_std=[1,1]).float().eval()
  with torch.no_grad():
   for name,value in model.named_parameters():
    value.normal_(0,.03)
    if name.endswith('.gamma'):value.add_(1)
  original={k:v.clone() for k,v in model.state_dict().items() if k.startswith(('decoder.','post_quant_conv.'))}
  for bf16 in (False,True):
   state={k:v.to(torch.bfloat16) if bf16 else v for k,v in original.items()};model.load_state_dict({k:v.float() for k,v in state.items()},strict=False)
   with tempfile.TemporaryDirectory() as directory,torch.no_grad():
    root=Path(directory);save_file(state,str(root/'model.safetensors'))
    for height,width in ((1,1),(2,2),(2,3)):
     latent=torch.randn(1,2,1,height,width)*.1;expected=model.decode(latent).sample
     (root/'config').write_text(f'4 2 4 1 {height} {width}');(root/'input').write_bytes(latent.numpy().tobytes());out=root/'output'
     result=json.loads(subprocess.check_output([binary,str(root),'model.safetensors',str(root/'config'),str(root/'input'),str(out)]));actual=torch.frombuffer(bytearray(out.read_bytes()),dtype=torch.float32).reshape_as(expected);out.unlink();torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5);assert result['resident_bytes']==0
     cases.append(dict(seed=seed,dtype='BF16' if bf16 else 'F32',height=height,width=width,maximum_absolute_error=float((actual-expected).abs().max())))
 print(json.dumps(dict(case_count=len(cases),cases=cases,gpu_execution=False,full_image_generation=False),indent=2))
if __name__=='__main__':main(sys.argv[1])
