"""Independent matrix-defined CPU F32/F64 oracle for the complete H3 denoiser.
This verifies the explicit native CPU contract, not production BF16/GPU parity.
"""
import argparse,hashlib,json,math,struct
from pathlib import Path
import torch
from safetensors import safe_open
PINNED_DIT='98afe42039e029b55001687df630c1a2ddf3b9193241e25f589af7c796fe885e'
SOURCE=Path('/usr/local/lib/python3.12/site-packages/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py')

def run(checkpoint,adapter,input_path,output):
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==PINNED_DIT
    torch.set_num_threads(2)
    with input_path.open('rb') as f:
        assert f.readline()==b'KADAN_H3_DENOISER_INPUT_V1\n';nt,nv,na=map(int,f.readline().split());n=nt+nv+na;raw=f.read()
    features=nt*5120+nv*96+na*32;count=features+n*4
    data=torch.frombuffer(bytearray(raw[:count*4]),dtype=torch.float32).clone();tags=torch.frombuffer(bytearray(raw[count*4:]),dtype=torch.int32).long()
    text=data[:nt*5120].reshape(nt,5120);video=data[nt*5120:nt*5120+nv*96].reshape(nv,96);audio=data[nt*5120+nv*96:features].reshape(na,32)
    positions=data[features:features+n*3].reshape(n,3);times=data[features+n*3:];unique,inverse=torch.unique(times,return_inverse=True)
    h4=torch.tensor([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]],dtype=torch.float64)/2
    rotations={};h=torch.ones((1,1),dtype=torch.float64)
    for size in (4,16,64,256):
        h=torch.kron(h,h4)
        if size in (64,256):rotations[size]=h
    with safe_open(checkpoint,framework='pt',device='cpu') as base,safe_open(adapter,framework='pt',device='cpu') as lora:
        def dense_projection(source,p,x):
            weight=source.get_slice(p+'.weight');out,inside=weight.get_shape();result=torch.empty((len(x),out),dtype=torch.float32)
            for start in range(0,out,128):result[:,start:start+128]=(x.double()@weight[start:start+128].double().T).float()
            return result
        def linear(p,x,bias=False,adapted=False):
            if p+'.comfy_quant' in base.keys():
                marker=json.loads(bytes(base.get_tensor(p+'.comfy_quant').tolist()));assert marker['format']=='int8_tensorwise' and marker['convrot'] is True
                group=marker['convrot_groupsize'];rotated=(x.double().reshape(len(x),-1,group)@rotations[group]).reshape_as(x).float();scale=rotated.abs().amax(dim=1).clamp_min(1e-10)/127
                codes=torch.round(rotated/scale[:,None]).clamp(-127,127);weight=base.get_slice(p+'.weight');out,inside=weight.get_shape();assert inside==x.shape[1]
                result=torch.empty((len(x),out));wscale=base.get_tensor(p+'.weight_scale').reshape(-1)
                for start in range(0,out,128):
                    dots=codes.double()@weight[start:start+128].double().T
                    result[:,start:start+128]=(dots.float()*scale[:,None])*wscale[start:start+128]
            else:result=dense_projection(base,p,x)
            if bias:result+=base.get_tensor(p+'.bias').float()
            if adapted:
                prefix='diffusion_model.'+p;rank=lora.get_slice(prefix+'.lora_A.weight').get_shape()[0]
                tmp=dense_projection(lora,prefix+'.lora_A',x);delta=dense_projection(lora,prefix+'.lora_B',tmp)
                result+=delta*(lora.get_tensor(prefix+'.alpha').item()/rank)
            assert torch.isfinite(result).all(),p
            return result
        def norm(p,x):
            weight=base.get_tensor(p+'.weight').float();return (x.double()*torch.rsqrt(x.double().square().mean(dim=-1,keepdim=True)+1e-5)*weight.double()).float()
        freq=base.get_tensor('rope.inv_freq').double();angle=(positions.double().unsqueeze(-1)*freq).flatten(1)
        def attn(p,x,rotate):
            packed=linear(p+'.qkv_proj',x,adapted=True).reshape(len(x),56,3,128)
            q,k,v=packed.unbind(2);q=norm(p+'.q_norm',q);k=norm(p+'.k_norm',k)
            if rotate:
                def rope(value):
                    a,b=value[...,:48].double(),value[...,48:96].double();c=angle.cos()[:,None,:];s=angle.sin()[:,None,:]
                    return torch.cat(((a*c-b*s).float(),(b*c+a*s).float(),value[...,96:]),dim=-1)
                q,k=rope(q),rope(k)
            scores=torch.einsum('thc,shc->hts',q.double(),k.double())/math.sqrt(128)
            attended=torch.einsum('hts,shc->thc',scores.softmax(-1),v.double()).float().reshape(len(x),7168)
            return linear(p+'.out_proj',attended,adapted=True)
        def mlp(p,x):
            gate,value=linear(p+'.fc1',x,adapted=True).chunk(2,dim=-1)
            activated=(gate.double()/(1+torch.exp(-gate.double()))).float()*value
            return linear(p+'.fc2',activated,adapted=True)
        text=linear('condition_proj',text,bias=True)
        for layer in range(2):
            p=f'token_refiner.blocks.{layer}'
            text=text+attn(p+'.attn',norm(p+'.norm1',text),False)
            text=text+mlp(p+'.mlp',norm(p+'.norm2',text));print('reference refiner',layer+1,flush=True)
        text=norm('token_refiner.final_norm',text)
        pieces=[text]
        if nv:pieces.append(linear('video_patch_proj',video,bias=True))
        if na:pieces.append(linear('audio_patch_proj',audio,bias=True))
        current=torch.cat(pieces)
        angle=unique.double()[:,None]*torch.exp(-math.log(10000)*torch.arange(128,dtype=torch.float64)/128)[None,:]
        t_freq=torch.cat((angle.cos(),angle.sin()),dim=1).float()
        t_hidden=linear('time_embedder.proj_in',t_freq,bias=True)
        t_hidden=(t_hidden.double()/(1+torch.exp(-t_hidden.double()))).float()
        t_embed=linear('time_embedder.proj_out',t_hidden,bias=True)
        t_embed=(t_embed.double()/(1+torch.exp(-t_embed.double()))).float()
        # Restore positional phases after constructing the time embedding.
        angle=(positions.double().unsqueeze(-1)*freq).flatten(1)
        combined=inverse*3+tags
        for layer in range(50):
            p=f'blocks.{layer}';params=linear(p+'.adaln_proj.linear',t_embed,bias=True).reshape(len(unique)*3,6,5376)[combined]
            shift,scale,gate,shift2,scale2,gate2=params.unbind(1)
            current=current+gate*attn(p+'.attn',norm(p+'.norm1',current)*(1+scale)+shift,True)
            current=current+gate2*mlp(p+'.mlp',norm(p+'.norm2',current)*(1+scale2)+shift2)
            assert torch.isfinite(current).all();print('reference block',layer+1,flush=True)
        params=linear('final_layer.adaln_proj.linear',t_embed,bias=True).reshape(len(unique),2,5376)[inverse];shift,scale=params.unbind(1)
        current=norm('final_layer.norm',current)*(1+scale)+shift
        result=[]
        if nv:result.append(linear('final_layer.video_out',current[nt:nt+nv],bias=True).flatten())
        if na:result.append(linear('final_layer.audio_out',current[nt+nv:],bias=True).flatten())
        expected=torch.cat(result);expected.numpy().tofile(output/'expected.f32')
    actual_path=output/'velocity.tensor'
    with actual_path.open('rb')as f:
        assert f.readline()==b'KADAN_H3_VELOCITY_F32_V1\n';assert tuple(map(int,f.readline().split()))==(nv,na);actual=torch.frombuffer(bytearray(f.read()),dtype=torch.float32)
    error=(actual-expected).abs();ok=bool(torch.allclose(actual,expected,atol=2e-4,rtol=2e-4))
    report={'values':len(expected),'max_abs':error.max().item(),'mean_abs':error.mean().item(),'passed':ok,'source_sha256':PINNED_DIT,'full_video_generation':False,'production_bf16_parity':False,'all_denoiser_blocks':50,'token_refiner_blocks':2,'turbo_lora':True}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));assert ok,report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('adapter');p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.checkpoint,a.adapter,a.input,a.output)
