import sys,bisect,json
symbols=[]
for line in open(sys.argv[1]):
 p=line.strip().split(' ',2)
 if len(p)==3 and p[1] in 'tT':
  try:symbols.append((int(p[0],16),p[2]))
  except ValueError:pass
symbols.sort();addresses=[x[0] for x in symbols]
rows=[];extra=[]
for line in open(sys.argv[2]):
 p=line.strip().split('\t')
 if len(p)==4 and p[0]!='kernel_offset':
  a=int(p[0],16);name=symbols[bisect.bisect_right(addresses,a)-1][1];rows.append(dict(name=name,calls=int(p[1]),gpu_ms=float(p[2]),launch_host_ms=float(p[3])))
 elif p[0]!='kernel_offset':extra.append(p)
rows.sort(key=lambda r:-r['gpu_ms'])
print(json.dumps(dict(kernels=rows,totals=extra),indent=2))
