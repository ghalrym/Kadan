"""Stdlib sparse oracle for H3 block0 residual, norm2 and gated feed-forward."""
import array
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

PREFIX='decoder.transformer_blocks.0.'


def fixture(root):
    header={};parts=[]
    def add(name,shape,data):
        start=sum(map(len,parts));parts.append(data)
        header[PREFIX+name]=dict(dtype='F16',shape=shape,data_offsets=[start,start+len(data)])
    def vector(name,count,fn):add(name,[count],struct.pack('<'+str(count)+'e',*[fn(i) for i in range(count)]))
    def matrix(name,rows,columns,offset,a,b):
        data=array.array('H',[0])*(rows*columns)
        for i in range(rows):
            data[i*columns+i%columns]=struct.unpack('<H',struct.pack('<e',a(i)))[0]
            data[i*columns+(i+offset)%columns]=struct.unpack('<H',struct.pack('<e',b(i)))[0]
        if sys.byteorder!='little':data.byteswap()
        add(name,[rows,columns],data.tobytes())
    vector('scale1',2048,lambda i:(i%7-3)/8)
    vector('norm2.weight',2048,lambda i:(i%7+1)/8)
    matrix('ff.w1.weight',16384,2048,17,lambda i:(i%13-6)/16,lambda i:(i%11+1)/32)
    vector('ff.w1.bias',16384,bias1)
    matrix('ff.w2.weight',2048,8192,67,lambda i:(i%11-5)/16,lambda i:(i%7+1)/32)
    vector('ff.w2.bias',2048,lambda i:(i%23-11)/64)
    vector('scale2',2048,lambda i:(i%5-2)/8)
    for kind in ('weights','wrong','shape','nan'):
        h=json.loads(json.dumps(header));payload=parts
        if kind=='wrong':h[PREFIX+'scale1']['dtype']='BF16'
        if kind=='shape':h[PREFIX+'scale1']['shape']=[1024,2]
        if kind=='nan':payload=[b'\x00\x7e'+parts[0][2:]]+parts[1:]
        encoded=json.dumps(h).encode()
        with (root/(kind+'.safetensors')).open('wb') as f:
            f.write(struct.pack('<Q',len(encoded)));f.write(encoded)
            for part in payload:f.write(part)


def bias1(i):return (80 if i==0 else -80) if i<2 else (i%11-5)/8


def verify(path,residual,attention):
    tokens=len(residual)//2048
    with path.open('rb') as f:
        assert f.readline()==b'KADAN_H3_FEED_FORWARD_V1\n'
        assert list(map(int,f.readline().split()))==[tokens,2048]
        assert f.readline()==b'F32LE\n';data=f.read()
    actual=struct.unpack('<'+str(len(data)//4)+'f',data);expected=[]
    for t in range(tokens):
        x=[residual[t*2048+c]+attention[t*2048+c]*((c%7-3)/8) for c in range(2048)]
        inverse=1/math.sqrt(sum(v*v for v in x)/2048+1e-5)
        norm=[v*inverse*((c%7+1)/8) for c,v in enumerate(x)]
        w1=[bias1(r)+norm[r%2048]*((r%13-6)/16)+norm[(r+17)%2048]*((r%11+1)/32) for r in range(16384)]
        gated=[(w1[c]/(1+math.exp(-w1[c])))*w1[c+8192] for c in range(8192)]
        for r in range(2048):
            out=(r%23-11)/64+gated[r]*((r%11-5)/16)+gated[(r+67)%8192]*((r%7+1)/32)
            expected.append(x[r]+out*((r%5-2)/8))
    assert len(actual)==len(expected)
    error=max(abs(a-b) for a,b in zip(actual,expected))
    assert all(math.isfinite(a) and abs(a-b)<=2e-5+2e-5*abs(b) for a,b in zip(actual,expected)),error
    return error


if __name__=='__main__':
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);fixture(root)
        subprocess.run([sys.argv[1],str(root)],check=True)
        residual=[(i%17-8)/8 for i in range(4096)];attention=[(i%13-6)/16 for i in range(4096)]
        maximum=verify(root/'result',residual,attention)
        for tokens in (1,2):
            for pattern in ('mixed','zero','tiny'):
                x=residual[:tokens*2048];a=attention[:tokens*2048]
                if pattern=='zero':x=[0.0]*len(x);a=[0.0]*len(a)
                if pattern=='tiny':x=[v/1048576 for v in x];a=[v/1048576 for v in a]
                for name,data in [('residual',x),('attention',a)]:
                    (root/name).write_bytes(struct.pack('<'+str(len(data))+'f',*data))
                dest=root/f'{tokens}-{pattern}'
                subprocess.run([sys.argv[2],str(root),'weights.safetensors',str(root/'residual'),str(root/'attention'),str(dest)],check=True,capture_output=True)
                maximum=max(maximum,verify(dest,x,a))
        print(f'7 residual/feed-forward scalar cases passed; max_abs={maximum}')
