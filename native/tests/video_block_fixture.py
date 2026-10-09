"""Independent scalar end-to-end block oracle with sparse F16 checkpoint weights."""
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

import video_qkv_fixture
import video_attention_fixture
import video_ff_fixture


def fixture(root):
    header={};parts=[];offset=0
    for index,module in enumerate((video_qkv_fixture,video_attention_fixture,video_ff_fixture)):
        directory=root/str(index);directory.mkdir();module.fixture(directory)
        path=directory/'weights.safetensors'
        with path.open('rb') as f:
            length=struct.unpack('<Q',f.read(8))[0];entries=json.loads(f.read(length))
        for name,tensor in entries.items():
            tensor['data_offsets']=[v+offset for v in tensor['data_offsets']];header[name]=tensor
        parts.append((path,8+length));offset+=path.stat().st_size-8-length
    for kind in ('weights','wrong'):
        h=json.loads(json.dumps(header))
        if kind=='wrong':h['decoder.transformer_blocks.0.scale1']['dtype']='BF16'
        encoded=json.dumps(h).encode()
        with (root/(kind+'.safetensors')).open('wb') as f:
            f.write(struct.pack('<Q',len(encoded)));f.write(encoded)
            for path,start in parts:
                with path.open('rb') as source:source.seek(start);shutil.copyfileobj(source,f,1024*1024)
    for index in range(3):shutil.rmtree(root/str(index))


def oracle(x,coords):
    tokens=len(x)//2048;qkv=[]
    for token in range(tokens):
        values=x[token*2048:(token+1)*2048]
        inv=1/math.sqrt(sum(v*v for v in values)/2048+1e-5)
        n=[v*inv*((i%7+1)/8) for i,v in enumerate(values)]
        qkv.extend((r%23-11)/64+n[r%2048]*((r%13-6)/16)+n[(r+17)%2048]*((r%11+1)/32) for r in range(6144))
    rotated=qkv[:]
    for t in range(tokens):
        angles=[2*math.pi*coords[t*3+a]*100**(-i/8) for a in range(3) for i in range(8)]
        for h in range(32):
            for kind in range(2):
                start=t*6144+h*192+kind*64;row=qkv[start:start+64]
                inv=1/math.sqrt(sum(v*v for v in row)/64+1e-5);n=[v*inv for v in row]
                rotated[start:start+64]=n
                for i,angle in enumerate(angles):
                    rotated[start+i]=n[i]*math.cos(angle)-n[i+24]*math.sin(angle)
                    rotated[start+i+24]=n[i+24]*math.cos(angle)+n[i]*math.sin(angle)
    output=[]
    for t in range(tokens):
        attended=[]
        for h in range(32):
            scores=[sum(rotated[t*6144+h*192+c]*rotated[k*6144+h*192+64+c] for c in range(64))/8 for k in range(tokens)]
            exponentials=[math.exp(s-max(scores)) for s in scores];weights=[e/sum(exponentials) for e in exponentials]
            attended.extend(sum(weights[k]*rotated[k*6144+h*192+128+c] for k in range(tokens)) for c in range(64))
        a=[(r%23-11)/64+attended[r]*((r%13-6)/16)+attended[(r+67)%2048]*((r%11+1)/32) for r in range(2048)]
        residual=[x[t*2048+c]+a[c]*((c%7-3)/8) for c in range(2048)]
        inv=1/math.sqrt(sum(v*v for v in residual)/2048+1e-5);n=[v*inv*((c%7+1)/8) for c,v in enumerate(residual)]
        w1=[video_ff_fixture.bias1(r)+n[r%2048]*((r%13-6)/16)+n[(r+17)%2048]*((r%11+1)/32) for r in range(16384)]
        gated=[w1[c]/(1+math.exp(-w1[c]))*w1[c+8192] for c in range(8192)]
        for r in range(2048):
            ff=(r%23-11)/64+gated[r]*((r%11-5)/16)+gated[(r+67)%8192]*((r%7+1)/32)
            output.append(residual[r]+ff*((r%5-2)/8))
    return output


def verify(path,x,coords):
    with path.open('rb') as f:
        assert f.readline()==b'KADAN_H3_BLOCK_V1\n'
        assert list(map(int,f.readline().split()))==[len(x)//2048,2048]
        assert f.readline()==b'F32LE\n';data=f.read()
    actual=struct.unpack('<'+str(len(data)//4)+'f',data);expected=oracle(x,coords)
    assert len(actual)==len(expected)
    maximum=max(abs(a-b) for a,b in zip(actual,expected))
    assert all(math.isfinite(a) and abs(a-b)<=3e-5+3e-5*abs(b) for a,b in zip(actual,expected)),maximum
    return maximum


if __name__=='__main__':
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);fixture(root)
        subprocess.run([sys.argv[1],str(root)],check=True)
        x=[(i%17-8)/8 for i in range(4096)];coords=[-.5,.25,.75,.5,-.25,-.75]
        maximum=max(verify(root/'result',x,coords),verify(root/'reuse',x[:2048],coords[:3]))
        for tokens in (1,2):
            for pattern in ('mixed','zero','tiny'):
                values=x[:tokens*2048]
                if pattern=='zero':values=[0.0]*len(values)
                if pattern=='tiny':values=[v/1048576 for v in values]
                c=coords[:tokens*3]
                (root/'input').write_bytes(struct.pack('<'+str(len(values))+'f',*values))
                (root/'coordinates').write_bytes(struct.pack('<'+str(len(c))+'f',*c))
                dest=root/f'{tokens}-{pattern}'
                subprocess.run([sys.argv[2],str(root),'weights.safetensors',str(root/'input'),str(root/'coordinates'),str(dest)],check=True,capture_output=True)
                maximum=max(maximum,verify(dest,values,c))
        print(f'8 end-to-end scalar block cases passed; max_abs={maximum}')
