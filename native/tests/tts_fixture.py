"""Sparse BF16 fixture gives an independent closed-form two-linear/SiLU oracle."""
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def fixture(path,kind='valid'):
    tensors={};payload=bytearray()
    for layer,diagonal,bias in ((1,.25,.125),(2,.5,-.0625)):
        for suffix in ('weight','bias'):
            if suffix=='weight':
                data=bytearray(2048*2048*2)
                bits=struct.unpack('<I',struct.pack('<f',diagonal))[0]>>16
                for i in range(2048):struct.pack_into('<H',data,(i*2048+i)*2,bits)
                shape=[2048,2048]
            else:
                bits=struct.unpack('<I',struct.pack('<f',bias))[0]>>16
                data=bytearray(struct.pack('<H',bits)*2048);shape=[2048]
            if kind=='nan' and layer==1:struct.pack_into('<H',data,0,0x7fc0)
            start=len(payload);payload.extend(data)
            tensors[f'talker.text_projection.linear_fc{layer}.{suffix}']=dict(dtype='F16' if kind=='wrong' else 'BF16',shape=shape,data_offsets=[start,len(payload)])
    header=json.dumps(tensors).encode();path.write_bytes(struct.pack('<Q',len(header))+header+payload)


if __name__=='__main__':
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);fixture(root/'weights.safetensors')
        for kind in ('wrong','nan'):fixture(root/f'{kind}.safetensors',kind)
        subprocess.run([sys.argv[1],str(root)],check=True)
        values=[(i%19-9)/16 for i in range(2048)]
        (root/'input').write_bytes(struct.pack('<2048f',*values))
        result=json.loads(subprocess.check_output([sys.argv[2],str(root),'weights.safetensors',str(root/'input')]))
        expected=[.5*((.25*x+.125)/(1+math.exp(-(.25*x+.125))))-.0625 for x in values]
        assert max(abs(a-b) for a,b in zip(result['projected'],expected))<1e-7
        assert result['resident_bytes_at_publication']==2*2048*4
        assert result['resident_bytes']==0 and not result['full_tts_generation']
        print('PASS: bounded TTS projection closed-form oracle and cleanup')
