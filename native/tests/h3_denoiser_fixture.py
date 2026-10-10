"""Sparse full-size H3 graph fixture with an independent scalar LoRA oracle."""
import json,math,struct,subprocess,sys,tempfile
from pathlib import Path
H=5376;INNER=7168;FF=14336
class Writer:
    def __init__(self):self.header={};self.payload=[];self.offset=0
    def add(self,name,dtype,shape,entries=()):
        count=math.prod(shape);width={'I8':1,'U8':1,'BF16':2,'F32':4}[dtype];size=count*width
        self.header[name]=dict(dtype=dtype,shape=shape,data_offsets=[self.offset,self.offset+size]);self.payload.append((self.offset,dtype,list(entries)));self.offset+=size
    def finish(self,path):
        raw=json.dumps(self.header,separators=(',',':')).encode();raw+=b' '*((-len(raw))%8)
        with path.open('wb')as f:
            f.write(struct.pack('<Q',len(raw)));f.write(raw);f.truncate(8+len(raw)+self.offset)
            for offset,dtype,entries in self.payload:
                for index,value in entries:
                    if dtype=='BF16':data=struct.pack('<f',value)[2:]
                    elif dtype=='F32':data=struct.pack('<f',value)
                    else:data=bytes((value&255,))
                    f.seek(8+len(raw)+offset+index*len(data));f.write(data)

def fixture(root):
    w=Writer();a=Writer()
    def dense(p,inside,out,bias=False,entries=()):
        w.add(p+'.weight','BF16',[out,inside],entries)
        if bias:w.add(p+'.bias','BF16',[out])
    def quant(p,inside,out,bias=False,group=256,bias_entries=()):
        w.add(p+'.weight','I8',[out,inside]);w.add(p+'.weight_scale','F32',[out,1],((i,1.)for i in range(out)))
        marker=json.dumps(dict(format='int8_tensorwise',convrot=True,convrot_groupsize=group)).encode();w.add(p+'.comfy_quant','U8',[len(marker)],enumerate(marker))
        if bias:w.add(p+'.bias','BF16',[out],bias_entries)
    def norm(p,width):w.add(p+'.weight','BF16',[width],((i,1.)for i in range(width)))
    def lora(p,inside,out,rank,A,B):
        p='diffusion_model.'+p;a.add(p+'.alpha','F32',[],[(0,float(rank))]);a.add(p+'.lora_A.weight','BF16',[rank,inside],A);a.add(p+'.lora_B.weight','BF16',[out,rank],B)
    def block(p,quantized):
        projection=quant if quantized else dense
        norm(p+'.norm1',H);norm(p+'.norm2',H);norm(p+'.attn.q_norm',128);norm(p+'.attn.k_norm',128)
        projection(p+'.attn.qkv_proj',H,INNER*3);projection(p+'.attn.out_proj',INNER,H);projection(p+'.mlp.fc1',H,FF*2);projection(p+'.mlp.fc2',FF,H)
        lora(p+'.attn.qkv_proj',H,INNER*3,384,[(0,.125)],[(2*INNER*384,.25)])
        lora(p+'.attn.out_proj',INNER,H,128,[(0,.25)],[(0,.5)])
        lora(p+'.mlp.fc1',H,FF*2,128,[(0,.25)],[(0,.5),(FF*128,.25)])
        lora(p+'.mlp.fc2',FF,H,128,[(0,.5)],[(0,.5)])
    dense('condition_proj',5120,H,True,((i*5120+i,1.)for i in range(4)))
    dense('video_patch_proj',96,H,True,((i*96+i,1.)for i in range(96)))
    dense('audio_patch_proj',32,H,True,((i*32+i,1.)for i in range(32)))
    for i in range(2):block(f'token_refiner.blocks.{i}',False)
    norm('token_refiner.final_norm',H)
    dense('time_embedder.proj_in',256,H,True);dense('time_embedder.proj_out',H,2688,True)
    w.add('rope.inv_freq','F32',[16],((i,10000**(-i/16))for i in range(16)))
    for i in range(50):
        p=f'blocks.{i}';block(p,True)
        bias=[(m*6*H+2*H+c,.25)for m in range(3)for c in range(H)]+[(m*6*H+5*H+c,.125)for m in range(3)for c in range(H)]
        quant(p+'.adaln_proj.linear',2688,96768,True,64,bias)
    norm('final_layer.norm',H);dense('final_layer.adaln_proj.linear',2688,H*2,True)
    dense('final_layer.video_out',H,96,True,((i*H+i,1.)for i in range(96)))
    dense('final_layer.audio_out',H,32,True,((i*H+i,1.)for i in range(32)))
    w.finish(root/'base.safetensors');a.finish(root/'turbo.safetensors')

def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def normalize(row):
    scale=1/math.sqrt(sum(x*x for x in row)/H+1e-5)
    return [f32(x*scale)for x in row]
def attend(rows):
    # Only head0/value0 is nonzero. Q/K are zero, so exact dense attention
    # reduces to the arithmetic mean, including the joint modality rows.
    values=[f32(f32(x[0]*.125)*.25)for x in rows]
    mean=f32(sum(values)/len(values));return f32(f32(mean*.25)*.5)
def feed(row):
    down=f32(row[0]*.25);gate=f32(down*.5);value=f32(down*.25)
    activated=f32(f32(gate/(1+math.exp(-gate)))*value)
    return f32(f32(activated*.5)*.5)
def oracle():
    text=[(i+1)*.125 if i<4 else 0. for i in range(H)]
    for _ in range(2):
        text[0]=f32(text[0]+attend([normalize(text)]));text[0]=f32(text[0]+feed(normalize(text)))
    rows=[normalize(text),[(i%9-4)*.125 if i<96 else 0. for i in range(H)],[(i%7-3)*.0625 if i<32 else 0. for i in range(H)]]
    for _ in range(50):
        delta=attend([normalize(row)for row in rows])
        for row in rows:row[0]=f32(row[0]+f32(delta*.25))
        for row in rows:row[0]=f32(row[0]+f32(feed(normalize(row))*.125))
    return normalize(rows[1])[:96]+normalize(rows[2])[:32]
if __name__=='__main__':
    with tempfile.TemporaryDirectory()as d:
        root=Path(d);fixture(root);subprocess.run([sys.argv[1],str(root)],check=True)
        raw=(root/'result.f32').read_bytes();actual=struct.unpack('<128f',raw);expected=oracle();error=max(abs(a-b)for a,b in zip(actual,expected));assert all(abs(a-b)<2e-5+2e-5*abs(b)for a,b in zip(actual,expected)),error
        print('Full graph scalar oracle max_abs',error)
