"""Independent scalar Q/K normalization and partial NeoX rotary equations."""
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def verify(path, values, coordinates):
    tokens=len(values)//6144
    with path.open('rb') as f:
        assert f.readline()==b'KADAN_H3_QK_ROPE_V1\n'
        assert list(map(int,f.readline().split()))==[tokens,32,3,64]
        assert f.readline()==b'F32LE\n'
        payload=f.read()
    assert len(payload)==len(values)*4
    actual=struct.unpack('<'+str(len(values))+'f',payload);expected=[]
    for token in range(tokens):
        angles=[2*math.pi*coordinates[token*3+axis]*100**(-i/8) for axis in range(3) for i in range(8)]
        for head in range(32):
            start=token*6144+head*192
            for kind in range(2):
                x=values[start+kind*64:start+(kind+1)*64]
                scale=1/math.sqrt(sum(v*v for v in x)/64+1e-5)
                n=[v*scale for v in x];out=n[:]
                for i,angle in enumerate(angles):
                    out[i]=n[i]*math.cos(angle)-n[i+24]*math.sin(angle)
                    out[i+24]=n[i+24]*math.cos(angle)+n[i]*math.sin(angle)
                expected.extend(out)
            expected.extend(values[start+128:start+192])
            assert payload[(start+128)*4:(start+192)*4]==struct.pack('<64f',*values[start+128:start+192])
    error=max(abs(a-b) for a,b in zip(actual,expected))
    assert all(math.isfinite(a) and abs(a-b)<=3e-6 for a,b in zip(actual,expected)),error
    print(f'QK/RoPE scalar oracle: {len(actual)} values; max_abs={error}; V bitwise unchanged')


if __name__=='__main__':
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);subprocess.run([sys.argv[1],str(root)],check=True)
        values=[(i%29-14)/8 for i in range(8*6144)];values[128]=-0.0
        coordinates=[v for t in range(8) for v in (t%2-.5,(t//2)%2-.5,t//4-.5)]
        verify(root/'result',values,coordinates)
        for count in range(1,9):
            for label in ('pattern','zero','tiny'):
                x=values[:count*6144] if label=='pattern' else [0.0 if label=='zero' else 1e-20]*(count*6144)
                c=coordinates[:count*3] if label=='pattern' else [0.0]*(count*3)
                packed=struct.pack('<'+str(len(x))+'f',*x);x=list(struct.unpack('<'+str(len(x))+'f',packed))
                (root/'input').write_bytes(packed);(root/'coordinates').write_bytes(struct.pack('<'+str(len(c))+'f',*c))
                output=root/f'{count}-{label}'
                subprocess.run([sys.argv[2],str(root/'input'),str(root/'coordinates'),str(output)],check=True,capture_output=True)
                verify(output,x,c)
        # CLI rejects inconsistent token counts before reserving/reading arrays.
        (root/'coordinates').write_bytes(struct.pack('<3f',0,0,0))
        result=subprocess.run([sys.argv[2],str(root/'input'),str(root/'coordinates'),str(root/'bad')],capture_output=True)
        assert result.returncode!=0 and not (root/'bad').exists()
