import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from image_fixture import write


def fixtures(root):
    write(root/'block.safetensors','weights')
    raw=(root/'block.safetensors').read_bytes();size=struct.unpack('<Q',raw[:8])[0];block=json.loads(raw[8:8+size]);shapes={}
    for i in range(2):
        for name,item in block.items():shapes[name.replace('transformer_blocks.0.',f'transformer_blocks.{i}.')]=item['shape']
    shapes.update({'img_in.weight':[16,4],'txt_in.text_norm.weight':[16],'txt_in.in_layer.weight':[16,16],'txt_in.out_layer.weight':[16,16],'time_text_embed.timestep_embedder.linear_1.weight':[16,256],'time_text_embed.timestep_embedder.linear_2.weight':[16,16],'modulation.1.weight':[64,16],'norm_out.linear.weight':[16,16],'proj_out.weight':[4,16]})
    for shard in range(2):
        header={};data=bytearray()
        for i,(name,shape) in enumerate(shapes.items()):
            if i%2!=shard:continue
            start=len(data);data.extend(bytes(math.prod(shape)*4));header[name]=dict(dtype='F32',shape=shape,data_offsets=[start,len(data)])
        encoded=json.dumps(header).encode();(root/('a.safetensors' if shard==0 else 'b.safetensors')).write_bytes(struct.pack('<Q',len(encoded))+encoded+data)
        if shard==0:
            header['img_in.weight']['shape']=[8,8];encoded=json.dumps(header).encode();(root/'wrong.safetensors').write_bytes(struct.pack('<Q',len(encoded))+encoded+data)

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);fixtures(root);subprocess.run([sys.argv[1],str(root)],check=True)
