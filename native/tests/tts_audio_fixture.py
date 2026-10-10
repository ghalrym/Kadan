import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

CONFIG=(4,16,8,8,8,2,4,2,16,32,3)
def shapes():
    groups,entries,code,latent,state,heads,head,layers,intermediate,decoder,window=CONFIG
    tensors={}
    def add(p,*shape):tensors['decoder.'+p]=shape
    def linear(p,i,o,bias=True):
        add(p+'.weight',o,i)
        if bias:add(p+'.bias',o)
    def conv(p,i,o,k,transpose=False,depth=False):
        add(p+'.conv.weight',i if transpose else o,1 if depth else o if transpose else i,k);add(p+'.conv.bias',o)
    def snake(p,n):add(p+'.alpha',n);add(p+'.beta',n)
    for rest in (False,True):
        p='quantizer.rvq_rest' if rest else 'quantizer.rvq_first';add(p+'.output_proj.weight',code,code//2,1)
        for i in range(groups-1 if rest else 1):
            q=f'{p}.vq.layers.{i}._codebook';add(q+'.cluster_usage',entries);add(q+'.embedding_sum',entries,code//2)
    conv('pre_conv',code,latent,3);linear('pre_transformer.input_proj',latent,state);linear('pre_transformer.output_proj',state,latent);add('pre_transformer.norm.weight',state)
    for i in range(layers):
        p=f'pre_transformer.layers.{i}'
        for name in ('input_layernorm.weight','post_attention_layernorm.weight','self_attn_layer_scale.scale','mlp_layer_scale.scale'):add(p+'.'+name,state)
        for name in ('q_proj','k_proj','v_proj'):linear(p+'.self_attn.'+name,state,heads*head,False)
        linear(p+'.self_attn.o_proj',heads*head,state,False)
        for name in ('gate_proj','up_proj'):linear(p+'.mlp.'+name,state,intermediate,False)
        linear(p+'.mlp.down_proj',intermediate,state,False)
    for i in range(2):
        p=f'upsample.{i}';conv(p+'.0',latent,latent,2,True);conv(p+'.1.dwconv',latent,latent,7,depth=True)
        for name in ('norm.weight','norm.bias','gamma'):add(p+'.1.'+name,latent)
        linear(p+'.1.pwconv1',latent,4*latent);linear(p+'.1.pwconv2',4*latent,latent)
    conv('decoder.0',latent,decoder,7)
    for i,rate in enumerate((8,5,4,3)):
        a=decoder//2**i;b=a//2;p=f'decoder.{i+1}.block';snake(p+'.0',a);conv(p+'.1',a,b,2*rate,True)
        for j in range(2,5):
            q=f'{p}.{j}';snake(q+'.act1',b);snake(q+'.act2',b);conv(q+'.conv1',b,b,7);conv(q+'.conv2',b,b,1)
    snake('decoder.5',decoder//16);conv('decoder.6',decoder//16,1,7)
    return tensors

def fixture(path,kind):
    data=bytearray();header={}
    for name,shape in shapes().items():
        start=len(data);data.extend(bytes(math.prod(shape)*4));header[name]=dict(dtype='F32',shape=shape,data_offsets=[start,len(data)])
    if kind=='wrong':header['decoder.pre_conv.conv.weight']['shape']=[8,4,6]
    if kind=='nan':data[:4]=struct.pack('<f',float('nan'))
    encoded=json.dumps(header).encode();path.write_bytes(struct.pack('<Q',len(encoded))+encoded+data)

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        for name,kind in [('weights','valid'),('wrong','wrong'),('nan','nan')]:fixture(root/f'{name}.safetensors',kind)
        subprocess.run([sys.argv[1],str(root)],check=True)
