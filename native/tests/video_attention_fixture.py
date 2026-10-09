"""Independent scalar softmax and sparse output projection, stdlib CPU CTest."""
import array
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

PREFIX='decoder.transformer_blocks.0.attn.to_out.'


def fixture(root):
    matrix=array.array('H',[0])*(2048*2048)
    for r in range(2048):
        matrix[r*2048+r]=struct.unpack('<H',struct.pack('<e',(r%13-6)/16))[0]
        matrix[r*2048+(r+67)%2048]=struct.unpack('<H',struct.pack('<e',(r%11+1)/32))[0]
    if sys.byteorder!='little':matrix.byteswap()
    parts=[matrix.tobytes(),struct.pack('<2048e',*[(r%23-11)/64 for r in range(2048)])]
    header={PREFIX+'weight':dict(dtype='F16',shape=[2048,2048],data_offsets=[0,len(parts[0])]),
            PREFIX+'bias':dict(dtype='F16',shape=[2048],data_offsets=[len(parts[0]),sum(map(len,parts))])}
    for kind in ('weights','wrong','shape','nan'):
        h=json.loads(json.dumps(header));payload=parts
        if kind=='wrong':h[PREFIX+'weight']['dtype']='BF16'
        if kind=='shape':h[PREFIX+'weight']['shape']=[1024,4096]
        if kind=='nan':payload=[b'\x00\x7e'+parts[0][2:],parts[1]]
        encoded=json.dumps(h).encode()
        (root/(kind+'.safetensors')).write_bytes(struct.pack('<Q',len(encoded))+encoded+b''.join(payload))


def verify(path,values):
    tokens=len(values)//6144
    with path.open('rb') as f:
        assert f.readline()==b'KADAN_H3_ATTENTION_V1\n'
        assert list(map(int,f.readline().split()))==[tokens,2048]
        assert f.readline()==b'F32LE\n'
        data=f.read()
    actual=struct.unpack('<'+str(len(data)//4)+'f',data)
    expected=[]
    for t in range(tokens):
        attended=[]
        for h in range(32):
            scores=[sum(values[t*6144+h*192+c]*values[k*6144+h*192+64+c] for c in range(64))/8 for k in range(tokens)]
            exponentials=[math.exp(s-max(scores)) for s in scores]
            weights=[e/sum(exponentials) for e in exponentials]
            attended.extend(sum(weights[k]*values[k*6144+h*192+128+c] for k in range(tokens)) for c in range(64))
        expected.extend((r%23-11)/64+attended[r]*((r%13-6)/16)+attended[(r+67)%2048]*((r%11+1)/32) for r in range(2048))
    assert len(actual)==len(expected)
    error=max(abs(a-b) for a,b in zip(actual,expected))
    assert all(math.isfinite(a) and abs(a-b)<=3e-6 for a,b in zip(actual,expected)),error
    return error


if __name__=='__main__':
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);fixture(root)
        subprocess.run([sys.argv[1],str(root)],check=True)
        maximum=verify(root/'result',[(i%17-8)/8 for i in range(8*6144)])
        for tokens in range(1,9):
            for pattern in ('mixed','uniform','peaked'):
                values=[(i%17-8)/8 for i in range(tokens*6144)]
                for t in range(tokens):
                    for h in range(32):
                        for c in range(128):
                            if pattern=='uniform':values[t*6144+h*192+c]=0
                            if pattern=='peaked':values[t*6144+h*192+c]=32.0 if c<64 or t==tokens-1 else -32.0
                (root/'input').write_bytes(struct.pack('<'+str(len(values))+'f',*values))
                output=root/f'{tokens}-{pattern}'
                subprocess.run([sys.argv[2],str(root),'weights.safetensors',str(root/'input'),str(output)],check=True,capture_output=True)
                maximum=max(maximum,verify(output,values))
        print(f'25 scalar attention/output projection cases passed, max_abs={maximum}')
