"""Independent scalar reference and bounded synthetic terminal-head fixtures."""
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

SHAPES = {
    'scorer.0.weight': [1024], 'scorer.0.bias': [1024],
    'scorer.1.weight': [1024,1024], 'scorer.1.bias': [1024],
    'scorer.3.weight': [1,1024], 'scorer.3.bias': [1],
    'act_head.0.weight': [256,1028], 'act_head.0.bias': [256],
    'act_head.2.weight': [2,256], 'act_head.2.bias': [2],
}

def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]

def reference(weights, values):
    marker,action=values[:1024],values[1024:]
    mean=sum(marker)/1024
    inverse=1/math.sqrt(sum((x-mean)**2 for x in marker)/1024+1e-5)
    norm=[f32(f32(f32((v-mean)*inverse)*weights['scorer.0.weight'][i])+weights['scorer.0.bias'][i]) for i,v in enumerate(marker)]
    def linear(data,prefix,activation):
        w,b=weights[prefix+'.weight'],weights[prefix+'.bias'];result=[]
        for r,value in enumerate(b):
            for c,x in enumerate(data):value=f32(value+f32(w[r*len(data)+c]*x))
            if activation:value=f32(.5*value*(1+math.erf(value/math.sqrt(2))))
            result.append(value)
        return result
    score=linear(linear(norm,'scorer.1',True),'scorer.3',False)
    acts=linear(linear(action,'act_head.0',True),'act_head.2',False)
    return score+acts


def fixture(path, kind='valid'):
    weights={};header={};payload=bytearray()
    for name,shape in SHAPES.items():
        size=math.prod(shape)
        values=[1.0]*size if name=='scorer.0.weight' else [((i*7)%17-8)/4096 for i in range(size)]
        if kind=='nan' and name=='scorer.0.weight':values[0]=math.nan
        data=struct.pack('<'+'e'*size,*values);start=len(payload);payload.extend(data)
        header[name]=dict(dtype='BF16' if kind=='wrong' else 'F16',shape=shape,data_offsets=[start,len(payload)])
        weights[name]=values
    encoded=json.dumps(header).encode();path.write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
    return weights


if __name__=='__main__':
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);weights=fixture(root/'weights.safetensors')
        for kind in ('wrong','nan'):fixture(root/f'{kind}.safetensors',kind)
        subprocess.run([sys.argv[1],str(root)],check=True)
        values=[(i%19-9)/16 for i in range(2052)]
        (root/'input').write_bytes(struct.pack('<2052f',*values))
        result=json.loads(subprocess.check_output([sys.argv[2],str(root),'weights.safetensors',str(root/'input')]))
        expected=reference(weights,values);actual=[result['score'],*result['actions']]
        assert max(abs(a-b) for a,b in zip(actual,expected))<1e-7,(actual,expected)
        assert result['resident_bytes']==0 and not result['full_decision_inference']
        print('PASS: scalar scorer/action reference and cleanup')
