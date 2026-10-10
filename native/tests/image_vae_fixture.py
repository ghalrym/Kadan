import json,math,struct,subprocess,sys,tempfile
from pathlib import Path

def fixtures(root):
 shapes={}
 def conv(p,i,o,k):shapes[p+'.weight']=(o,i,k,k);shapes[p+'.bias']=(o,)
 def norm(p,n,image=False):shapes[p+'.gamma']=(n,1,1) if image else (n,1,1,1)
 def res(p,i,o):
  norm(p+'.norm1',i);conv(p+'.conv1',i,o,3);norm(p+'.norm2',o);conv(p+'.conv2',o,o,3)
  if i!=o:conv(p+'.conv_shortcut',i,o,1)
 conv('post_quant_conv',2,2,1);conv('decoder.conv_in',2,16,3);res('decoder.mid_block.resnets.0',16,16);norm('decoder.mid_block.attentions.0.norm',16,True);conv('decoder.mid_block.attentions.0.to_qkv',16,48,1);conv('decoder.mid_block.attentions.0.proj',16,16,1);res('decoder.mid_block.resnets.1',16,16)
 dims=[16,16,16,8,4,2]
 for i in range(5):
  p=f'decoder.up_blocks.{i}';res(p+'.resnets.0',dims[i],dims[i+1])
  if i<4:conv(p+'.upsampler.resample.1',dims[i+1],dims[i+1],3)
 norm('decoder.norm_out',2);conv('decoder.conv_out',2,4,3);header={};data=bytearray()
 for name,shape in shapes.items():
  values=[.25 if name=='decoder.conv_out.bias' else 0.]*math.prod(shape);start=len(data);data.extend(struct.pack('<'+'f'*len(values),*values));header[name]=dict(dtype='F32',shape=shape,data_offsets=[start,len(data)])
 encoded=json.dumps(header).encode();(root/'model.safetensors').write_bytes(struct.pack('<Q',len(encoded))+encoded+data)
if __name__=='__main__':
 with tempfile.TemporaryDirectory() as directory:
  root=Path(directory);fixtures(root);subprocess.run([sys.argv[1],str(root)],check=True)
