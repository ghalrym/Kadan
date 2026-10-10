import struct
from pathlib import Path
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary);data=bytearray(b'KDWVOC01')
    def n(value):data.extend(struct.pack('<I',value))
    for value in (257,256,1,65,0,1,256):n(value)
    for i in range(257):
        piece=bytes([i]) if i<256 else b'';n(len(piece));data.extend(piece)
    (root/'english.tokens').write_bytes(data);(root/'truncated.tokens').write_bytes(data[:-2])
    cases=[b'',b'Hello Andrew!',bytes(range(128)),b'\xe2\x82\xac',b'\xe2\x82',b'\xf0\x9f\x98\x80',b'\xf0\x9f',b'\xed\xa0\x80',b'\xe0\x80\x80',b'\xff\xfe',b'\xc0\x80',b'\xe2x',bytes(range(256))]
    for i,raw in enumerate(cases):
        (root/f'{i}.ids').write_bytes(struct.pack(f'<{len(raw)}I',*raw));(root/f'{i}.expected').write_bytes(raw.decode('utf8',errors='replace').encode())
    subprocess.run([sys.argv[1],str(root),'257',str(len(cases))],check=True)
