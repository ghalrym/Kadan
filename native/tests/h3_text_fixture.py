"""Sparse complete 50-layer H3 conditioning graph and independent scalar oracle."""
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def fixture(path):
    specs={}
    def add(name,dtype,shape,entries=()):specs[name]=(dtype,shape,entries)
    embedding=[(token*5120+c,(c%23-11+token)/32) for token in range(2) for c in range(5120)]
    add('model.embed_tokens.weight','BF16',[151936,5120],embedding)
    marker=json.dumps(dict(format='int8_tensorwise',convrot=True,convrot_groupsize=256)).encode()
    for layer in range(50):
        prefix=f'model.layers.{layer}.'
        for name,width in [('input_layernorm',5120),('post_attention_layernorm',5120),('self_attn.q_norm',128),('self_attn.k_norm',128)]:
            add(prefix+name+'.weight','BF16',[width],[(i,1) for i in range(width)])
        for name,rows,columns in [('self_attn.q_proj',8192,5120),('self_attn.k_proj',1024,5120),('self_attn.v_proj',1024,5120),('self_attn.o_proj',5120,8192),('mlp.gate_proj',25600,5120),('mlp.up_proj',25600,5120),('mlp.down_proj',5120,25600)]:
            add(prefix+name+'.weight','I8',[rows,columns],[(0,(layer%7+1)*(-1 if name=='mlp.up_proj' else 1))])
            add(prefix+name+'.weight_scale','F32',[rows,1],[(i,1/8) for i in range(rows)])
            add(prefix+name+'.comfy_quant','U8',[len(marker)],list(enumerate(marker)))
    widths={'BF16':2,'I8':1,'F32':4,'U8':1};header={};offset=0
    for name,(dtype,shape,_) in specs.items():
        end=offset+math.prod(shape)*widths[dtype];header[name]=dict(dtype=dtype,shape=shape,data_offsets=[offset,end]);offset=end
    encoded=json.dumps(header).encode()
    with path.open('wb') as f:
        f.write(struct.pack('<Q',len(encoded)));f.write(encoded);f.truncate(8+len(encoded)+offset)
        for name,(dtype,_,entries) in specs.items():
            start=8+len(encoded)+header[name]['data_offsets'][0]
            # Contiguous small arrays avoid millions of fixture-only seek/write calls.
            if dtype!='I8' and name!='model.embed_tokens.weight':
                if dtype=='U8':data=bytes(v for _,v in entries)
                elif dtype=='F32':data=struct.pack('<'+str(len(entries))+'f',*(v for _,v in entries))
                else:data=b''.join(struct.pack('<f',v)[2:] for _,v in entries)
                f.seek(start);f.write(data)
            else:
                for index,value in entries:
                    f.seek(start+index*widths[dtype]);f.write(struct.pack('<b',value) if dtype=='I8' else struct.pack('<f',value)[2:])


def rotated(values):
    # Kronecker matrix application by recursively splitting into four quarters;
    # independent order from the native stride-increasing in-place butterfly.
    signs=((1,1,1,-1),(1,1,-1,1),(1,-1,1,1),(-1,1,1,1))
    def transform(row):
        if len(row)==1:return row
        if not any(row):return [0.0]*len(row)
        width=len(row)//4;parts=[transform(row[i*width:(i+1)*width]) for i in range(4)]
        return [sum(signs[r][c]*parts[c][i] for c in range(4))/2 for r in range(4) for i in range(width)]
    return [value for at in range(0,len(values),256) for value in transform(values[at:at+256])]


def norm(row):
    inverse=1/math.sqrt(sum(x*x for x in row)/len(row)+1e-6)
    return [x*inverse for x in row]


def project(rows,width,layer,negative=False):
    result=[]
    for row in rows:
        values=rotated(row);scale=max(1e-10,max(abs(v) for v in values))/127
        code=max(-127,min(127,round(values[0]/scale)))
        first=code*scale*(layer%7+1)/8*(-1 if negative else 1)
        result.append([first]+[0.0]*(width-1))
    return result


def oracle(tokens):
    rows=[[(c%23-11+t)/32 for c in range(5120)] for t in range(tokens)]
    for layer in range(50):
        n=[norm(row) for row in rows]
        q=project(n,8192,layer);k=project(n,1024,layer);v=project(n,1024,layer)
        for t in range(tokens):
            for data in (q,k):
                data[t][:128]=norm(data[t][:128]);x=data[t][0];data[t][0]=x*math.cos(t);data[t][64]=x*math.sin(t)
        attended=[]
        for t in range(tokens):
            scores=[(q[t][0]*k[j][0]+q[t][64]*k[j][64])/math.sqrt(128) for j in range(t+1)]
            weights=[math.exp(s-max(scores)) for s in scores]
            first=sum(weights[j]*v[j][0] for j in range(t+1))/sum(weights)
            row=[first]+[0.0]*8191
            for head in range(1,8):row[head*128]=sum(v[j][0] for j in range(t+1))/(t+1)
            attended.append(row)
        a=project(attended,5120,layer)
        rows=[[x+y for x,y in zip(row,extra)] for row,extra in zip(rows,a)]
        n=[norm(row) for row in rows];gate=project(n,25600,layer);up=project(n,25600,layer,True)
        gated=[[g/(1+math.exp(-g))*u for g,u in zip(grow,urow)] for grow,urow in zip(gate,up)]
        projected=project(gated,5120,layer)
        rows=[[x+y for x,y in zip(row,extra)] for row,extra in zip(rows,projected)]
    return [v for row in rows for v in row]


def verify(path,tokens):
    with path.open('rb') as f:
        assert f.readline()==b'KADAN_H3_CONDITIONING_F32_V1\n'
        assert list(map(int,f.readline().split()))==[tokens,5120]
        assert f.readline()==b'F32LE\n';data=f.read()
    actual=struct.unpack('<'+str(len(data)//4)+'f',data);expected=oracle(tokens)
    maximum=max(abs(a-b) for a,b in zip(actual,expected))
    assert all(math.isfinite(a) and abs(a-b)<=2e-4+2e-4*abs(b) for a,b in zip(actual,expected)),maximum
    return maximum

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);fixture(root/'weights.safetensors')
        subprocess.run([sys.argv[1],str(root)],check=True)
        print('whole-model scalar oracle max_abs='+str(max(verify(root/'result',2),verify(root/'reuse',1))))
        (root/'ids').write_bytes(struct.pack('<2I',0,1))
        with (root/'weights.safetensors').open('rb') as f:
            length=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(length))
        p='model.layers.0.'
        corruptions=[(p+'self_attn.q_proj.weight_scale',struct.pack('<f',math.nan),'h3_text_scale'),
                     (p+'input_layernorm.weight',bytes((128,127)),'h3_text_nonfinite_weight'),
                     (p+'self_attn.q_proj.comfy_quant',json.dumps(dict(format='int8_tensorwise',convrot=True,convrot_groupsize=128)).encode(),'h3_text_quant_marker')]
        marker_name=p+'self_attn.q_proj.comfy_quant'
        marker_bytes=header[marker_name]['data_offsets'][1]-header[marker_name]['data_offsets'][0]
        malformed_markers=[
            b'{"format":"int8_tensor wise","convrot":true,"convrot_groupsize":256}',
            b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":2 56}',
            b'{"format":"int8_tensorwise","convrot":tr ue,"convrot_groupsize":256}',
            b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":0256}',
            b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}x',
        ]
        for marker in malformed_markers:
            assert len(marker)<=marker_bytes
            corruptions.append((marker_name,marker.ljust(marker_bytes,b' '),'h3_text_quant_marker'))
        for index,(name,replacement,error) in enumerate(corruptions):
            at=8+length+header[name]['data_offsets'][0]
            with (root/'weights.safetensors').open('r+b') as f:
                f.seek(at);original=f.read(len(replacement));f.seek(at);f.write(replacement)
            dest=root/f'invalid-{index}'
            result=subprocess.run([sys.argv[2],str(root),'weights.safetensors',str(root/'ids'),str(dest)],capture_output=True,text=True)
            assert result.returncode==1 and error in result.stderr and not dest.exists(),result.stderr
            with (root/'weights.safetensors').open('r+b') as f:f.seek(at);f.write(original)
        print('Malformed scale, norm and quantization marker fail closed')
