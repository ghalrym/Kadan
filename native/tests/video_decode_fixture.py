"""Sparse real-dimension F16 graph and independent scalar 36-block frame oracle."""
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def fixture(path):
    specs={}
    def add(name,shape,values=()):
        specs[name]=(shape,values)
    add('latents_mean',[24],[(i,(i-12)/32) for i in range(24)])
    add('latents_std',[24],[(i,1) for i in range(24)])
    add('post_quant_conv.weight',[24,24,1,1,1],[(i*25,1) for i in range(24)])
    add('post_quant_conv.bias',[24])
    add('decoder.x_embedder.weight',[2048,24],[(i*24+i%24,1/8) for i in range(2048)])
    add('decoder.x_embedder.bias',[2048],[(i,(i%7-3)/32) for i in range(2048)])
    add('decoder.register_tokens',[1,4,2048],[(i,(i%13-6)/16) for i in range(8192)])
    for layer in range(36):
        p=f'decoder.transformer_blocks.{layer}.'
        add(p+'norm1.weight',[2048],[(i,1) for i in range(2048)])
        add(p+'attn.to_qkv.weight',[6144,2048],[(0,1/4),(64*2048,1/8),(128*2048+1,1/2)])
        add(p+'attn.to_qkv.bias',[6144],[(0,(layer+1)/64),(64,1/8)])
        add(p+'attn.to_out.weight',[2048,2048],[(0,1/4)])
        add(p+'attn.to_out.bias',[2048],[(1,(layer+1)/1024)])
        add(p+'scale1',[2048],[(i,1/8) for i in range(2048)])
        add(p+'norm2.weight',[2048],[(i,1) for i in range(2048)])
        add(p+'ff.w1.weight',[16384,2048],[(0,1/8),(8192*2048+1,1/4)])
        add(p+'ff.w1.bias',[16384],[(0,1/16),(8192,1/8)])
        add(p+'ff.w2.weight',[2048,8192],[(0,1/4)])
        add(p+'ff.w2.bias',[2048],[(2,(layer+1)/2048)])
        add(p+'scale2',[2048],[(i,1/8) for i in range(2048)])
    add('decoder.norm_out.weight',[2048],[(i,1) for i in range(2048)])
    add('decoder.norm_out.bias',[2048],[(i,(i%5-2)/64) for i in range(2048)])
    add('decoder.proj_out.weight',[3072,2048],[(r*2048+r%2048,1/4) for r in range(3072)])
    add('decoder.proj_out.bias',[3072],[(r,(r%23-11)/128) for r in range(3072)])
    header={};offset=0
    for name,(shape,_) in specs.items():
        end=offset+math.prod(shape)*2;header[name]=dict(dtype='F16',shape=shape,data_offsets=[offset,end]);offset=end
    encoded=json.dumps(header).encode()
    with path.open('wb') as f:
        f.write(struct.pack('<Q',len(encoded)));f.write(encoded);f.truncate(8+len(encoded)+offset)
        for name,(_,values) in specs.items():
            start=8+len(encoded)+header[name]['data_offsets'][0]
            for index,value in values:f.seek(start+index*2);f.write(struct.pack('<e',value))


def oracle(values,shape):
    t,h,w=shape;patches=t*h*w
    rows=[[(values[p*24+c%24]+(c%24-12)/32)/8+(c%7-3)/32 for c in range(2048)] for p in range(patches)]
    rows += [[((i*2048+c)%13-6)/16 for c in range(2048)] for i in range(4)]+[[0.0]*2048]
    coords=[(2*(ti+.5)/t-1,2*(hi+.5)/h-1,2*(wi+.5)/w-1) for ti in range(t) for hi in range(h) for wi in range(w)]+[(0,0,0)]*5
    for layer in range(36):
        qs=[];ks=[];vs=[]
        for row,coord in zip(rows,coords):
            inverse=1/math.sqrt(sum(v*v for v in row)/2048+1e-5)
            q=row[0]*inverse/4+(layer+1)/64;k=row[0]*inverse/8+1/8;vs.append(row[1]*inverse/2)
            q/=math.sqrt(q*q/64+1e-5);k/=math.sqrt(k*k/64+1e-5)
            angle=2*math.pi*coord[0]
            qs.append((q*math.cos(angle),q*math.sin(angle)));ks.append((k*math.cos(angle),k*math.sin(angle)))
        for i,row in enumerate(rows):
            scores=[(qs[i][0]*k[0]+qs[i][1]*k[1])/8 for k in ks]
            exp=[math.exp(s-max(scores)) for s in scores]
            row[0]+=sum(e*v for e,v in zip(exp,vs))/sum(exp)/32
            row[1]+=(layer+1)/8192
            inverse=1/math.sqrt(sum(v*v for v in row)/2048+1e-5)
            a=row[0]*inverse/8+1/16;b=row[1]*inverse/4+1/8
            row[0]+=a/(1+math.exp(-a))*b/32;row[2]+=(layer+1)/16384
    output=[0.0]*(patches*3072)
    for p,row in enumerate(rows[:patches]):
        mean=sum(row)/2048;inv=1/math.sqrt(sum((x-mean)**2 for x in row)/2048+1e-5)
        row=[(x-mean)*inv+(c%5-2)/64 for c,x in enumerate(row)]
        lt=p//(h*w);lh=(p//w)%h;lw=p%w
        for r in range(3072):
            ch=r//1024;pt=(r//256)%4;ph=(r//16)%16;pw=r%16
            index=((ch*t*4+lt*4+pt)*h*16+lh*16+ph)*w*16+lw*16+pw
            output[index]=row[r%2048]/4+(r%23-11)/128
    return output


def verify(path,values,shape):
    with path.open('rb') as f:
        assert f.readline()==b'KADAN_H3_FRAMES_V1\n'
        t,h,w=shape;assert list(map(int,f.readline().split()))==[3,t*4,h*16,w*16]
        assert f.readline()==b'F32LE\n';data=f.read()
    actual=struct.unpack('<'+str(len(data)//4)+'f',data);expected=oracle(values,shape)
    assert len(actual)==len(expected)
    maximum=max(abs(a-b) for a,b in zip(actual,expected))
    assert all(math.isfinite(a) and abs(a-b)<=1e-4+1e-4*abs(b) for a,b in zip(actual,expected)),maximum
    return maximum

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);fixture(root/'weights.safetensors')
        subprocess.run([sys.argv[1],str(root)],check=True)
        values=[(i%17-8)/8 for i in range(48)]
        maximum=max(verify(root/'result',values,(1,1,2)),verify(root/'reuse',values,(2,1,1)),verify(root/'height',values,(1,2,1)))
        # CLI rejects mismatched input before any checkpoint allocation.
        (root/'input').write_bytes(struct.pack('<48f',*values))
        rejected=subprocess.run([sys.argv[2],str(root),'weights.safetensors',str(root/'input'),'1','1','1',str(root/'invalid')],capture_output=True,text=True)
        assert rejected.returncode==1 and 'video_input_shape' in rejected.stderr and not (root/'invalid').exists()
        print(f'Complete 36-block frame oracle passed for spatial and temporal reconstruction; max_abs={maximum}')
