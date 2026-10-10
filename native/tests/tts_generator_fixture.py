import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def shapes():
    out={}
    def add(name,*shape):out[name]=shape
    def linear(p,i,o,bias=False):
        add(p+'.weight',o,i)
        if bias:add(p+'.bias',o)
    add('talker.model.text_embedding.weight',32,8)
    add('talker.model.codec_embedding.weight',24,8)
    for n in ('linear_fc1','linear_fc2'):linear('talker.text_projection.'+n,8,8,True)
    linear('talker.codec_head',8,24)
    linear('talker.code_predictor.small_to_mtp_projection',8,4,True)
    for i in range(3):
        add(f'talker.code_predictor.model.codec_embedding.{i}.weight',16,8)
        linear(f'talker.code_predictor.lm_head.{i}',4,16)
    for p,state,heads,kv,dim,layers,intermediate in [('talker.model',8,2,1,4,2,16),('talker.code_predictor.model',4,2,1,4,2,8)]:
        add(p+'.norm.weight',state)
        for i in range(layers):
            q=f'{p}.layers.{i}'
            for n in ('input_layernorm','post_attention_layernorm'):add(q+'.'+n+'.weight',state)
            for n in ('q_norm','k_norm'):add(q+'.self_attn.'+n+'.weight',dim)
            linear(q+'.self_attn.q_proj',state,heads*dim)
            for n in ('k_proj','v_proj'):linear(q+'.self_attn.'+n,state,kv*dim)
            linear(q+'.self_attn.o_proj',heads*dim,state)
            for n in ('gate_proj','up_proj'):linear(q+'.mlp.'+n,state,intermediate)
            linear(q+'.mlp.down_proj',intermediate,state)
    return out


def fixture(path,kind):
    data=bytearray();header={}
    for name,shape in shapes().items():
        value=0.0
        if 'norm.weight' in name or name=='talker.text_projection.linear_fc2.bias':value=1.0
        values=[value]*math.prod(shape)
        if name=='talker.codec_head.weight' and kind=='eos':values[23*8:24*8]=[1.0]*8
        start=len(data);data.extend(struct.pack('<'+'f'*len(values),*values));header[name]=dict(dtype='F32',shape=shape,data_offsets=[start,len(data)])
    if kind=='nan':data[:4]=struct.pack('<f',float('nan'))
    if kind=='wrong':header['talker.model.text_embedding.weight']['shape']=[16,16]
    encoded=json.dumps(header).encode();path.write_bytes(struct.pack('<Q',len(encoded))+encoded+data)


if __name__=='__main__':
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        for kind in ('valid','nan','wrong','eos'):fixture(root/(kind+'.safetensors'),kind)
        subprocess.run([sys.argv[1],str(root)],check=True)
