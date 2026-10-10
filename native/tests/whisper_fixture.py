"""CPU full-network fixture; independent seeded parity lives in whisper_reference.py."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

DIMS=(4,5,8,2,2,17,7,8,2,2)
def shapes(d=DIMS):
    m,ac,a,ah,al,v,tc,t,th,tl=d
    result={}
    def add(name,*shape):result[name]=shape
    def norm(p,n):add(p+'.weight',n);add(p+'.bias',n)
    def linear(p,i,o,bias=True):
        add(p+'.weight',o,i)
        if bias:add(p+'.bias',o)
    def attention(p,n):
        for part in ('query','key','value','out'):linear(p+'.'+part,n,n,part!='key')
    add('encoder.conv1.weight',a,m,3);add('encoder.conv1.bias',a)
    add('encoder.conv2.weight',a,a,3);add('encoder.conv2.bias',a)
    add('encoder.positional_embedding',ac,a)
    add('decoder.token_embedding.weight',v,t);add('decoder.positional_embedding',tc,t)
    for root,n,layers in [('encoder',a,al),('decoder',t,tl)]:
        for i in range(layers):
            p=f'{root}.blocks.{i}'
            norm(p+'.attn_ln',n);attention(p+'.attn',n)
            if root=='decoder':norm(p+'.cross_attn_ln',n);attention(p+'.cross_attn',n)
            norm(p+'.mlp_ln',n);linear(p+'.mlp.0',n,4*n);linear(p+'.mlp.2',4*n,n)
        norm(root+('.ln' if root=='decoder' else '.ln_post'),n)
    return result

def fixture(path,kind):
    import math
    rows={};data=bytearray();width=2 if kind=='half' else 4
    for name,shape in shapes().items():
        n=math.prod(shape);start=len(data);data.extend(bytes(n*width))
        rows[name]=dict(dtype='F16' if width==2 else 'F32',shape=shape,data_offsets=[start,len(data)])
    if kind=='wrong':rows['encoder.conv1.weight']['shape']=[8,3,4]
    if kind=='nan':data[:4]=struct.pack('<f',float('nan'))
    header=json.dumps(rows,separators=(',',':')).encode();path.write_bytes(struct.pack('<Q',len(header))+header+data)

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        for name,kind in [('weights','valid'),('wrong','wrong'),('nan','nan'),('half','half')]:fixture(root/f'{name}.safetensors',kind)
        subprocess.run([sys.argv[1],str(root)],check=True)
