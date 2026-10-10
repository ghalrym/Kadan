import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def fixtures(root):
    shapes={'embed_tokens.weight':(32,16),'norm.weight':(16,)}
    for i in range(2):
        p=f'layers.{i}.'
        for n in ('input_layernorm','post_attention_layernorm'):shapes[p+n+'.weight']=(16,)
        for n in ('q_norm','k_norm'):shapes[p+'self_attn.'+n+'.weight']=(8,)
        for n in ('q_proj','o_proj'):shapes[p+'self_attn.'+n+'.weight']=(16,16)
        for n in ('k_proj','v_proj'):shapes[p+'self_attn.'+n+'.weight']=(8,16)
        for n in ('gate_proj','up_proj'):shapes[p+'mlp.'+n+'.weight']=(48,16)
        shapes[p+'mlp.down_proj.weight']=(16,48)
    for shard in range(2):
        header={};data=bytearray()
        for i,(name,shape) in enumerate(shapes.items()):
            if i%2!=shard:continue
            values=[0.]*math.prod(shape)
            if name=='embed_tokens.weight':values=[(i//16+1)/8 for i in range(32*16)]
            if name=='norm.weight':values=[100.]*16
            start=len(data);data.extend(struct.pack('<'+'f'*len(values),*values));header['model.language_model.'+name]=dict(dtype='F32',shape=shape,data_offsets=[start,len(data)])
        encoded=json.dumps(header).encode();(root/('a.safetensors' if shard==0 else 'b.safetensors')).write_bytes(struct.pack('<Q',len(encoded))+encoded+data)
if __name__=='__main__':
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);fixtures(root);subprocess.run([sys.argv[1],str(root)],check=True)
