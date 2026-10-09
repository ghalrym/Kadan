"""Dependency-free CLI tests; real pinned Whisper comparison is explicit CPU-only."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

def main(binary):
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);bank=root/'bank';pcm=root/'pcm'
        for bins in (80,128):
            bank.write_bytes(struct.pack('<'+str(bins*201)+'f',*([0.]*(bins*201))))
            for n in (201,319,320,15999,16000):
                pcm.write_bytes(bytes(n*4))
                result=json.loads(subprocess.check_output([binary,str(bins),str(bank),str(pcm)]))
                assert result['frames']==n//160 and result['bins']==bins
                assert result['mel']==[-1.5]*(bins*(n//160))
                assert result['resident_bytes_at_publication']==(bins*201+n+bins*(n//160))*4
                assert result['resident_bytes']==0 and not result['full_transcription'] and not result['gpu_execution']
            for raw in (bytes(200*4),bytes(16001*4),bytes(805),struct.pack('<f',float('nan'))+bytes(200*4)):
                pcm.write_bytes(raw)
                assert subprocess.run([binary,str(bins),str(bank),str(pcm)],capture_output=True).returncode!=0
        bank.write_bytes(b'');assert subprocess.run([binary,'80',str(bank),str(pcm)],capture_output=True).returncode!=0
if __name__=='__main__':main(sys.argv[1])
