"""CPU-only independent tensor-operation oracle for the native H3 decoder.
Usage: script BINARY [actual_audio_vae.safetensors]. No download or GPU use.
"""
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import torch
import torch.nn.functional as F
from h3_audio_fixture import shapes,write

def load(path):
    with Path(path).open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));out={}
        for name,s in h.items():
            if not name.startswith(('latents_','dec_in_proj.','decoder.')):continue
            assert s['dtype']=='F32'
            start,end=s['data_offsets'];f.seek(8+n+start)
            out[name]=torch.frombuffer(bytearray(f.read(end-start)),dtype=torch.float32).reshape(s['shape']).clone()
        return out

def filter12():
    # Reference window design for the fixture only. Production loads stored taps.
    A=2.285*5*math.pi*1.2+7.95
    beta=.1102*(A-8.7) if A>50 else .5842*(A-21)**.4+.07886*(A-21) if A>=21 else 0
    window=torch.kaiser_window(12,beta=beta,periodic=False)
    t=torch.arange(-6,6)+.5;v=.5*window*torch.sinc(.5*t)
    return (v/v.sum()).reshape(1,1,12)

def decode(w,packed,initial):
    # [stereo,time,32] -> independent mono batch [stereo,32,time].
    x=packed.permute(0,2,1);x=x*w['latents_std'][None,:,None]+w['latents_mean'][None,:,None]
    def conv(p,x,d=1):
        a=w[p+'.weight'];return F.conv1d(x,a,w.get(p+'.bias'),padding=(a.shape[-1]-1)*d//2,dilation=d)
    def act(p,x):
        c=x.shape[1];v=F.conv_transpose1d(F.pad(x,(5,5),mode='replicate'),w[p+'.upsample.filter'].expand(c,1,12),stride=2,groups=c)*2
        v=v[:,:,15:-15];a=w[p+'.act.alpha'].exp()[None,:,None];b=w[p+'.act.beta'].exp()[None,:,None]
        v=v+torch.sin(a*v).square()/(b+1e-9)
        return F.conv1d(F.pad(v,(5,6),mode='replicate'),w[p+'.downsample.lowpass.filter'].expand(c,1,12),stride=2,groups=c)
    def block(p,x):
        for j,d in enumerate((1,3,5)):
            v=conv(f'{p}.convs1.{j}',act(f'{p}.activations.{2*j}',x),d)
            v=conv(f'{p}.convs2.{j}',act(f'{p}.activations.{2*j+1}',v));x=x+v
        return x
    x=conv('dec_in_proj',x);x=conv('decoder.conv_pre',x)
    for i in range(7):
        p=f'decoder.ups.{i}.0';rate=5 if i<2 else 2;a=w[p+'.weight']
        x=F.conv_transpose1d(x,a,w[p+'.bias'],stride=rate,padding=(a.shape[-1]-rate)//2)
        branches=[block(f'decoder.resblocks.{i*3+j}',x) for j in range(3)]
        x=(branches[0]+branches[1]+branches[2])/3
    x=conv('decoder.conv_post',act('decoder.activation_post',x)).clamp(-1,1)
    return x[:,0,:].T.contiguous()

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(173)
    rows=[]
    with tempfile.TemporaryDirectory() as folder:
        folder=Path(folder)
        if len(sys.argv)>2:
            checkpoint=Path(sys.argv[2]);w=load(checkpoint);mode='full';initial=1024;cases=(1,2)
        else:
            mode='small';initial=128;cases=(1,2,3);w={}
            for name,shape in shapes().items():
                if name.endswith('.filter'):v=filter12()
                elif name=='latents_std':v=torch.linspace(.7,1.7,32)
                elif name=='latents_mean':v=torch.linspace(-.2,.2,32)
                elif '.act.' in name:v=torch.randn(shape)*.1
                elif name.endswith('.bias'):v=torch.randn(shape)*.03
                else:v=torch.randn(shape)*(.7/math.sqrt(math.prod(shape[1:])))
                w[name]=v
            checkpoint=folder/'weights.safetensors';write(checkpoint,{n:(tuple(v.shape),v.numpy().tobytes()) for n,v in w.items()})
        for frames in cases:
            packed=torch.randn(2,frames,32)*.3;packed[1]+=.7
            inp=folder/'input';out=folder/'output';inp.write_bytes(packed.numpy().tobytes())
            started=time.monotonic();subprocess.run([sys.argv[1],str(checkpoint.parent),str(inp),str(out),checkpoint.name,mode],check=True)
            actual=torch.frombuffer(bytearray(out.read_bytes()),dtype=torch.float32).reshape(-1,2)
            with torch.no_grad():expected=decode(w,packed,initial)
            error=(actual-expected).abs();assert actual.shape==(frames*800,2);assert torch.isfinite(actual).all()
            assert bool((error<=2e-5+2e-4*expected.abs()).all()),float(error.max())
            assert not torch.equal(expected[:,0],expected[:,1]),'fixture must exercise independent stereo'
            rows.append({'frames':frames,'samples_per_channel':frames*800,'max_abs_error':float(error.max()),'seconds':time.monotonic()-started,'reference_stereo_max_difference':float((expected[:,0]-expected[:,1]).abs().max())})
    print(json.dumps({'mode':mode,'sample_rate':32000,'gpu_execution':False,'cases':rows},indent=2))

if __name__=='__main__':main()
