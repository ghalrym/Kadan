"""Independent CPU F32/ConvRot INT8 whole-model oracle for H3's 50-layer tap.
Not production BF16/GPU parity. Usage: BINARY CHECKPOINT NEW_OUTPUT_DIRECTORY.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F


def run(binary,checkpoint,output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    before=checkpoint.stat();stream=checkpoint.open('rb')
    length=struct.unpack('<Q',stream.read(8))[0];assert length<1024*1024
    encoded=stream.read(length);header=json.loads(encoded);hashes={}
    assert json.loads(header['__metadata__']['minimax_h3_te'])==dict(num_hidden_layers=50,output='unnormalized_hidden_after_layer_50')
    def tensor(name):
        item=header[name];start,end=item['data_offsets'];stream.seek(8+length+start);data=stream.read(end-start)
        assert len(data)==end-start;hashes[name]=hashlib.sha256(data).hexdigest()
        dtype={'BF16':'<u2','F32':'<f4','I8':'i1','U8':'u1'}[item['dtype']]
        array=np.frombuffer(data,dtype=dtype).copy().reshape(item['shape'])
        if item['dtype']=='BF16':array=(array.astype(np.uint32)<<16).view(np.float32)
        return torch.from_numpy(array)
    h4=torch.tensor([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]],dtype=torch.float64)
    rotation=h4
    for _ in range(3):rotation=torch.kron(rotation,h4)
    rotation=rotation/16
    def linear(prefix,x):
        marker=bytes(tensor(prefix+'.comfy_quant').tolist());assert json.loads(marker)==dict(format='int8_tensorwise',convrot=True,convrot_groupsize=256)
        weight=tensor(prefix+'.weight');scale=tensor(prefix+'.weight_scale').T
        # Matrix-defined regular Hadamard reference, independent of C++ butterfly.
        rotated=(x.double().reshape(-1,256)@rotation).reshape(x.shape).float()
        activation_scale=rotated.abs().amax(dim=-1,keepdim=True).clamp(min=1e-10)/127
        codes=torch.round(rotated/activation_scale).clamp(-127,127).to(torch.int32)
        product=codes@weight.to(torch.int32).T
        return (product.float()*activation_scale)*scale
    ids=[0,1];values=[]
    info=header['model.embed_tokens.weight'];assert info['dtype']=='BF16' and info['shape']==[151936,5120]
    for token in ids:
        stream.seek(8+length+info['data_offsets'][0]+token*5120*2);data=stream.read(5120*2)
        values.append((np.frombuffer(data,dtype='<u2').astype(np.uint32)<<16).view(np.float32))
    hidden=torch.from_numpy(np.stack(values));positions=torch.arange(len(ids),dtype=torch.float64)
    angles=positions[:,None]/(5000000.0**(torch.arange(64,dtype=torch.float64)/64))
    def rope(x):
        a=x[...,:64].double();b=x[...,64:].double();cos=angles[:,None,:].cos();sin=angles[:,None,:].sin()
        return torch.cat((a*cos-b*sin,b*cos+a*sin),dim=-1).float()
    def rms(x,weight):
        return F.rms_norm(x.double(),(x.shape[-1],),weight.double(),1e-6).float()
    for layer in range(50):
        p=f'model.layers.{layer}.'
        n=rms(hidden,tensor(p+'input_layernorm.weight'))
        q=linear(p+'self_attn.q_proj',n).reshape(len(ids),64,128)
        k=linear(p+'self_attn.k_proj',n).reshape(len(ids),8,128)
        v=linear(p+'self_attn.v_proj',n).reshape(len(ids),8,128)
        q=rope(rms(q,tensor(p+'self_attn.q_norm.weight'))).transpose(0,1).unsqueeze(0)
        k=rope(rms(k,tensor(p+'self_attn.k_norm.weight'))).transpose(0,1).unsqueeze(0)
        attended=F.scaled_dot_product_attention(q.double(),k.double(),v.transpose(0,1).unsqueeze(0).double(),is_causal=True,enable_gqa=True).float()
        attended=attended[0].transpose(0,1).reshape(len(ids),8192)
        hidden=hidden+linear(p+'self_attn.o_proj',attended)
        n=rms(hidden,tensor(p+'post_attention_layernorm.weight'))
        hidden=hidden+linear(p+'mlp.down_proj',F.silu(linear(p+'mlp.gate_proj',n).double()).float()*linear(p+'mlp.up_proj',n))
        print('reference_layer='+str(layer+1),flush=True)
    output.mkdir(parents=True,exist_ok=False);(output/'ids.u32').write_bytes(struct.pack('<2I',*ids))
    result=subprocess.run([str(binary),str(checkpoint.parent),checkpoint.name,str(output/'ids.u32'),str(output/'features.tensor')],check=True,capture_output=True,text=True)
    with (output/'features.tensor').open('rb') as f:
        assert f.readline()==b'KADAN_H3_CONDITIONING_F32_V1\n'
        assert list(map(int,f.readline().split()))==[2,5120]
        assert f.readline()==b'F32LE\n';data=f.read()
    actual=np.frombuffer(data,dtype='<f4').reshape(2,5120);expected=hidden.numpy()
    (output/'expected.f32').write_bytes(expected.tobytes())
    absolute=np.abs(actual-expected);cosine=float((actual.flatten()@expected.flatten())/(np.linalg.norm(actual)*np.linalg.norm(expected)))
    report=dict(token_ids=ids,layers=50,values=actual.size,max_absolute_error=float(absolute.max()),mean_absolute_error=float(absolute.mean()),cosine=cosine,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),header_sha256=hashlib.sha256(encoded).hexdigest(),tensor_sha256=hashes,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reference='Independent PyTorch F32 activations, F64 reductions/rotation, matrix-defined regular Hadamard and dynamic INT8 activations',
        production_bf16_parity=False,gpu_execution=False,full_video_generation=False,stdout=result.stdout.strip())
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='tensor_sha256'},indent=2))
    after=checkpoint.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
    assert np.isfinite(actual).all() and np.all(absolute<=.02+.005*np.abs(expected)) and cosine>=.9999

if __name__=='__main__':run(*(Path(p).resolve() for p in sys.argv[1:]))
