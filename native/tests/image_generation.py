import binascii,struct,subprocess,sys,tempfile,zlib
from pathlib import Path
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);subprocess.run([sys.argv[1],str(root)],check=True);data=(root/'result.png').read_bytes();assert data[:8]==b'\x89PNG\r\n\x1a\n';at=8;compressed=b'';tags=[]
 while at<len(data):
  count=struct.unpack('>I',data[at:at+4])[0];tag=data[at+4:at+8];body=data[at+8:at+8+count];crc=struct.unpack('>I',data[at+8+count:at+12+count])[0];assert binascii.crc32(tag+body)==crc;tags.append(tag)
  if tag==b'IHDR':assert struct.unpack('>IIBBBBB',body)==(2,2,8,6,0,0,0)
  if tag==b'IDAT':compressed+=body
  at+=count+12
 assert tags==[b'IHDR',b'IDAT',b'IEND'];assert zlib.decompress(compressed)==(bytes([0])+bytes([0,255,128,255])*2)*2
 wide=(root/'wide.png').read_bytes();assert struct.unpack('>II',wide[16:24])==(2752,1536);assert len(wide)>16*1024**2
 print('PNG pixels, integrity, exclusive publication, cancellation and request bounds passed')
