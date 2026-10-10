import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def write(path,kind):
    header={};data=bytearray()
    shapes={**{f'attn.{name}.weight':(16,16) for name in ('to_q','to_k','to_v','to_out.0')},**{f'attn.{name}.weight':(8,) for name in ('norm_q','norm_k')},'img_mlp.proj.weight':(48,16),'img_mlp.gate_layer.weight':(48,16),'img_mlp.out.weight':(16,48)}
    for name,shape in shapes.items():
        count=1
        for n in shape:count*=n
        start=len(data);data.extend(bytes(count*4));header['transformer_blocks.0.'+name]=dict(dtype='F32',shape=shape,data_offsets=[start,len(data)])
    if kind=='wrong':header['transformer_blocks.0.attn.to_q.weight']['shape']=[8,32]
    if kind=='nan':data[:4]=struct.pack('<f',float('nan'))
    encoded=json.dumps(header).encode();path.write_bytes(struct.pack('<Q',len(encoded))+encoded+data)

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        for name in ('weights','nan','wrong'):write(root/(name+'.safetensors'),name)
        subprocess.run([sys.argv[1],str(root)],check=True)
