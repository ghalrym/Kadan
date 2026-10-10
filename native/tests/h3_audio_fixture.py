"""Small complete H3 decoder fixture; no model download or inference library."""
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

PROJECTION,INITIAL=8,128

def shapes(projection=PROJECTION,initial=INITIAL):
    s={}
    def add(p,*shape):s[p]=shape
    def conv(p,i,o,k,transpose=False,bias=True):
        add(p+'.weight',i if transpose else o,o if transpose else i,k)
        if bias:add(p+'.bias',o)
    def act(p,c):
        add(p+'.act.alpha',c);add(p+'.act.beta',c)
        add(p+'.upsample.filter',1,1,12);add(p+'.downsample.lowpass.filter',1,1,12)
    add('latents_mean',32);add('latents_std',32)
    conv('dec_in_proj',32,projection,1);conv('decoder.conv_pre',projection,initial,7)
    for i in range(7):
        c=initial//2**(i+1);conv(f'decoder.ups.{i}.0',c*2,c,9 if i<2 else 4,True)
        for j,k in enumerate((3,7,11)):
            p=f'decoder.resblocks.{3*i+j}'
            for a in range(3):conv(f'{p}.convs1.{a}',c,c,k);conv(f'{p}.convs2.{a}',c,c,k)
            for a in range(6):act(f'{p}.activations.{a}',c)
    act('decoder.activation_post',initial//128);conv('decoder.conv_post',initial//128,1,7,bias=False)
    return s

def write(path,tensors):
    header={};chunks=[];offset=0
    for name,(shape,data) in tensors.items():
        header[name]={'dtype':'F32','shape':shape,'data_offsets':[offset,offset+len(data)]};chunks.append(data);offset+=len(data)
    encoded=json.dumps(header).encode()
    with Path(path).open('wb') as f:
        f.write(struct.pack('<Q',len(encoded)));f.write(encoded)
        for b in chunks:f.write(b)

def fixture(path,kind):
    tensors={n:(s,bytes(math.prod(s)*4)) for n,s in shapes().items()}
    tensors['latents_std']=((32,),struct.pack('<32f',*[1.]*32))
    if kind=='wrong':tensors['dec_in_proj.weight']=((8,16,2),bytes(8*32*4))
    if kind=='nan':tensors['latents_mean']=((32,),struct.pack('<32f',float('nan'),*[0.]*31))
    if kind=='std':tensors['latents_std']=((32,),bytes(128))
    write(path,tensors)

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as folder:
        for name in ('weights','wrong','nan','std'):fixture(Path(folder)/(name+'.safetensors'),name)
        subprocess.run([sys.argv[1],folder],check=True)
