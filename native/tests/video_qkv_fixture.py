"""Sparse per-head QKV oracle independent of the component; stdlib-only CPU CI."""
import array
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

PREFIX = 'decoder.transformer_blocks.0.'


def fixture(root):
    header={}
    parts=[]
    def add(name,shape,data):
        start=sum(map(len,parts));parts.append(data)
        header[PREFIX+name]=dict(dtype='F16',shape=shape,data_offsets=[start,start+len(data)])
    add('norm1.weight',[2048],struct.pack('<2048e',*[(i%7+1)/8 for i in range(2048)]))
    matrix=array.array('H',[0])*(6144*2048)
    for r in range(6144):
        # Distinct Q/K/V within every head; two nonzero terms exercise layout.
        matrix[r*2048+r%2048]=struct.unpack('<H',struct.pack('<e',(r%13-6)/16))[0]
        matrix[r*2048+(r+17)%2048]=struct.unpack('<H',struct.pack('<e',(r%11+1)/32))[0]
    if sys.byteorder!='little':matrix.byteswap()
    add('attn.to_qkv.weight',[6144,2048],matrix.tobytes())
    add('attn.to_qkv.bias',[6144],struct.pack('<6144e',*[(r%23-11)/64 for r in range(6144)]))
    for kind in ('weights','wrong','shape','nan'):
        h=json.loads(json.dumps(header));payload=parts
        if kind=='wrong':h[PREFIX+'norm1.weight']['dtype']='BF16'
        if kind=='shape':h[PREFIX+'norm1.weight']['shape']=[1024,2]
        if kind=='nan':payload=[b'\x00\x7e'+parts[0][2:]]+parts[1:]
        encoded=json.dumps(h).encode()
        with (root/(kind+'.safetensors')).open('wb') as f:
            f.write(struct.pack('<Q',len(encoded)));f.write(encoded)
            for part in payload:f.write(part)


def verify(path,values):
    with path.open('rb') as f:
        assert f.readline()==b'KADAN_H3_DECODER_QKV_V1\n'
        assert list(map(int,f.readline().split()))==[len(values)//2048,32,3,64]
        assert f.readline()==b'F32LE\n'
        data=f.read()
    actual=struct.unpack('<'+'f'*(len(data)//4),data)
    expected=[]
    for t in range(len(values)//2048):
        x=values[t*2048:(t+1)*2048]
        inverse=1/math.sqrt(sum(v*v for v in x)/2048+1e-5)
        normalized=[v*inverse*((c%7+1)/8) for c,v in enumerate(x)]
        for r in range(6144):
            expected.append((r%23-11)/64+normalized[r%2048]*((r%13-6)/16)
                            +normalized[(r+17)%2048]*((r%11+1)/32))
    assert len(actual)==len(expected)
    error=max(abs(a-b) for a,b in zip(actual,expected))
    assert all(math.isfinite(a) and abs(a-b)<=2e-6 for a,b in zip(actual,expected)),error
    print(f'Sparse RMSNorm/per-head QKV oracle: {len(actual)} values, max_abs={error}')


if __name__=='__main__':
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);fixture(root)
        subprocess.run([sys.argv[1],str(root)],check=True)
        values=[(i%17-8)/8 for i in range(8*2048)]
        verify(root/'result',values)
        (root/'input').write_bytes(struct.pack('<'+str(len(values))+'f',*values))
        subprocess.run([sys.argv[2],str(root),'weights.safetensors',str(root/'input'),str(root/'cli')],check=True)
        verify(root/'cli',values)
