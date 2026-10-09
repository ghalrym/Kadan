"""Proposed image-A operator. Requires separate exact-source/bundle execution approval."""
import sys,os,json,time,signal,tempfile,hashlib,http.client,threading,urllib.request,socket
from pathlib import Path
TOOLS=Path('/home/andrew/Documents/Codex/2026-10-08/task-4/machine-local-benchmark-tools/harness')
sys.path.insert(0,str(TOOLS))
import launch_trajectory as host
from launch_api_baseline import restore_exact
from api_baseline import compare, verify
from thermal_guard import check_cpu
from cuda_wait_monitor import ThermalWatch, cgroup_identity, task_counters, close_watchdog, publish_watchdog_error
from bf16_contracts import require_ci
from durable_evidence import append
from window_supervisor import load_context, arm, record_memory_baseline, LABEL

ROOT=Path('/home/andrew/Documents/Codex/2026-10-08/task-4')
COMMIT=os.environ['KADAN_REVIEWED_QUEUE_COMMIT']
WINDOW_DIR=Path(os.environ['KADAN_WINDOW_DIRECTORY'])
WINDOW=load_context(WINDOW_DIR)
assert time.monotonic()<WINDOW['work_deadline']
host.COMMAND_DEADLINE=WINDOW['work_deadline']
BASELINE_COMMIT='b57db7173d6735f93244c6300539ef421da9a9f6'
NAME='kadan-api-dual-reviewed'
KIND=sys.argv[1]
assert KIND=='a', 'This reviewed window permits exactly image A'
BUNDLE=Path(__file__).resolve().parent
binding=json.loads((BUNDLE/'binding.json').read_text())
assert binding['source_commit']==COMMIT
assert binding['protocol']=='external-durable-image-a-bundle-v3'
assert binding['external_tools']==str(TOOLS)
for name,digest in binding['external_runtime'].items():
 assert hashlib.sha256((TOOLS/name).read_bytes()).hexdigest()==digest
assert binding['wrapper_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
for name,digest in binding['runtime'].items():
 assert hashlib.sha256((host.REPO/name).read_bytes()).hexdigest()==digest
review=json.loads(Path(sys.argv[2]).read_text())
assert review==dict(protocol='single-api-image-a-v3',source_commit=COMMIT,
 binding_sha256=hashlib.sha256((BUNDLE/'binding.json').read_bytes()).hexdigest(),
 decision='approved-for-bounded-execution')
ci=json.loads(host.run('gh','run','list','--repo','ghalrym/Kadan','--commit',COMMIT,'--limit','100',
 '--json','name,status,conclusion,event,headSha').stdout)
require_ci(ci,COMMIT)
predecessor=ROOT/'real-block-wait-reviewed-7480ba3'
assert hashlib.sha256((predecessor/'manifest.json').read_bytes()).hexdigest()==binding['predecessor_manifest_sha256']
for name,row in json.loads((predecessor/'manifest.json').read_text()).items():
 assert hashlib.sha256((predecessor/name).read_bytes()).hexdigest()==row['sha256']
assert json.loads((predecessor/'summary.json').read_text())['passed']
assert json.loads((predecessor/'pause-result.json').read_text())['restored']
E=ROOT/('api-queue-reviewed-'+COMMIT[:7]+'-'+KIND)
assert not E.exists()
assert host.run('git','-C',str(host.REPO),'rev-parse','HEAD').stdout.strip()==COMMIT
assert not host.run('git','-C',str(host.REPO),'status','--porcelain').stdout
for case in ('A','B'):
 verify(ROOT/f'api-baseline-reviewed-b57db71-{case}'/'trajectory-evidence/baseline',case,BASELINE_COMMIT)
 assert json.loads((ROOT/f'api-baseline-reviewed-b57db71-{case}'/'pause-result.json').read_text())['restored']
assert hashlib.sha256((BUNDLE/'retained-roots.json').read_bytes()).hexdigest()==binding['retained_roots_sha256']
roots={Path(p).resolve(strict=True) for p in json.loads((BUNDLE/'retained-roots.json').read_text())}
assert not any(a!=b and a.is_relative_to(b) for a in roots for b in roots)
assert not any(E.resolve().is_relative_to(p) or p.is_relative_to(E.resolve()) for p in roots)
assert predecessor in roots and BUNDLE in roots
used=sum(f.stat().st_size for p in roots for f in p.rglob('*') if f.is_file())
# The complete reserved envelope was admitted earlier. Every new window checks
# current bytes plus its own bounded final/transient contribution again.
assert used+512*1024**2+1024**2<=32*1024**3
assert int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024 >= 280*1024**3
E.mkdir();(E/'trajectory-evidence').mkdir();host.THERMAL_EVIDENCE=E
(E/'ci.json').write_text(json.dumps(ci));(E/'review.json').write_text(json.dumps(review))
(E/'binding.json').write_text((BUNDLE/'binding.json').read_text())
(E/'storage.json').write_text(json.dumps(dict(retained_bytes=used,window_cap_bytes=512*1024**2,roots=[str(p) for p in sorted(roots)])))
for i in range(5):
 check_cpu(E,'cooldown-admission',limit=60)
 if i<4:time.sleep(2)
before=host.inspect_container(host.API);api_id=before['Id'];assert before['State']['Running'] and before['Image']==host.IMAGE
assert host.run('docker','inspect',NAME,check=False).returncode!=0
others={n:host.inspect_container(n)['Id'] for n in ('kadan-redis-1','kadan-postgres-1','kadan-frontend-1')}
def redis(*args):return json.loads(host.run('docker','exec','kadan-redis-1','redis-cli','--json',*args).stdout)
assert redis('SCARD','kadan:inference:unfinished')==0 and redis('LLEN','kadan:inference:pending')==0
desktop=host.desktop_baseline(api_id)
identity=hashlib.sha256(json.dumps(host.container_identity(before),sort_keys=True).encode()).hexdigest()
(E/'identity.json').write_text(json.dumps(dict(api_id=api_id,config_sha256=identity,others=others,source_commit=COMMIT,baseline_commit=BASELINE_COMMIT)))
payloads=ROOT/'mr125-validation-plan-a447672'
assert hashlib.sha256((payloads/'image-a.json').read_bytes()).hexdigest()==binding['payload_sha256']
assert hashlib.sha256((BUNDLE/'logging.json').read_bytes()).hexdigest()==binding['logging_sha256']
started=WINDOW['started'];end=WINDOW['operator_end'];measure_end=WINDOW['work_deadline'];paused=False;owned=False;baseline=None;passed=False
calls=[];seen=set();last_sample=0;monitor=None;cgroup=None
def cpu_sample(pid):
 result=dict(pid=pid,threads=[])
 try:
  for task in (Path('/proc')/str(pid)/'task').iterdir():
   try:
    fields=(task/'stat').read_text().rsplit(')',1)[1].split()
    mask=next(l.split(':',1)[1].strip() for l in (task/'status').read_text().splitlines() if l.startswith('Cpus_allowed_list:'))
    result['threads'].append(dict(tid=int(task.name),utime=int(fields[11]),stime=int(fields[12]),cpu=int(fields[36]),allowed=mask))
   except (FileNotFoundError,ProcessLookupError):pass
 except (FileNotFoundError,ProcessLookupError):pass
 return result
class Call:
 def __init__(self,label,payload,route):
  self.label=label;self.payload=payload;self.route=route;self.result=None;self.error=None;self.job=None;self.start=time.monotonic();self.finish=None
  self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
 def run(self):
  try:
   c=http.client.HTTPConnection('127.0.0.1',18000,timeout=930);self.connection=c
   c.request('POST',self.route,json.dumps(self.payload),{'Content-Type':'application/json'})
   r=c.getresponse();body=r.read(1024**2);self.result=dict(status=r.status,body=json.loads(body));c.close()
  except BaseException as exc:self.error=str(exc)
  finally:
   finished=time.monotonic()
   try:append(E/'request-events.jsonl',dict(event='response',label=self.label,job=self.job,start=self.start,finish=finished,result=self.result,error=self.error))
   except Exception as exc:self.error='Response evidence failure: '+str(exc)
   finally:self.finish=finished
def get(path):
 with urllib.request.urlopen('http://127.0.0.1:18000'+path,timeout=3) as r:return json.load(r)
def sample():
 global last_sample
 if time.monotonic()>=measure_end:raise TimeoutError('Measurement deadline')
 if time.monotonic()-last_sample<1:return
 monitor.check()
 last_sample=time.monotonic();row=host.guards();row['monotonic']=last_sample
 row['processes']=host.run('nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits').stdout
 row['lifecycle']=get('/model-lifecycle')
 row['ranks']=[]
 lines=host.run('docker','top',NAME,'-eo','pid,args').stdout.splitlines()[1:]
 row['task_counters']=task_counters(cgroup)
 row['cpu_processes']=[cpu_sample(int(line.split(None,1)[0])) for line in lines]
 row['host_cpu_stat']=[line for line in Path('/proc/stat').read_text().splitlines() if line.startswith('cpu')]
 row['loadavg']=Path('/proc/loadavg').read_text().strip()
 row['cgroup_cpu_stat']=host.run('docker','exec',NAME,'cat','/sys/fs/cgroup/cpu.stat',check=False).stdout
 for line in lines:
  if ' -m api.inference.image.rank_worker ' in line:
   tail=line.split(' -m api.inference.image.rank_worker ',1)[1];config=json.loads(tail.split(None,1)[1])
   row['ranks'].append(dict(pid=int(line.split(None,1)[0]),rank=config['rank'],session=config['session'],cpus=config['cpus']))
 if row['ranks']:
  allowed=set(row['ranks'][0]['cpus']);assert len(allowed)<=2
  assert all(set(r['cpus'])==allowed for r in row['ranks'])
  rank_pids={r['pid'] for r in row['ranks']}
  for proc in row['cpu_processes']:
   if proc['pid'] not in rank_pids:continue
   for thread in proc['threads']:
    actual=set()
    for part in thread['allowed'].split(','):
     ends=[int(v) for v in part.split('-')];actual.update(range(ends[0],ends[-1]+1))
    assert actual<=allowed,'Rank helper thread escaped aggregate CPU mask'
 row['pending']=redis('LRANGE','kadan:inference:pending','0','-1')
 row['unfinished']=redis('SMEMBERS','kadan:inference:unfinished')
 with (E/'telemetry.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 if sum(f.stat().st_size for f in E.rglob('*') if f.is_file())>48*1024**2:raise RuntimeError('Window telemetry/log export reserve reached')
def job_keys():
 cursor='0';keys=set()
 while True:
  cursor,page=redis('SCAN',cursor,'MATCH','kadan:inference:job:*','COUNT','256');keys.update(page)
  if str(cursor)=='0':return keys
def submit(label,file,route):
 previous=job_keys()
 call=Call(label,json.loads((payloads/file).read_text()),route);calls.append(call)
 deadline=time.monotonic()+10
 while True:
  fresh={key.rsplit(':',1)[1] for key in job_keys()-previous}-seen
  if len(fresh)==1:
   call.job=fresh.pop();seen.add(call.job)
   append(E/'request-events.jsonl',dict(event='admitted',label=label,job=call.job,start=call.start,monotonic=time.monotonic()))
   record=redis('HGET','kadan:inference:job:'+call.job,'job')
   if record is not None:assert all(json.loads(record)['payload'].get(k)==v for k,v in call.payload.items())
   return call
  if call.finish is not None:raise RuntimeError('Request completed before its FIFO identity was observed: '+str(call.result or call.error))
  if time.monotonic()>=deadline:raise TimeoutError('FIFO admission not observed')
  sample();time.sleep(.1)
def wait(call):
 while call.finish is None:sample();time.sleep(.2)
 assert not call.error and call.result['status']==200,(call.error,call.result)
 return call.result['body']
def interrupted(signum,frame):raise InterruptedError(str(signum))
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,interrupted)
signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGUSR1,interrupted)
try:
 monitor=ThermalWatch(E,host.GPUS);monitor.start()
 arm(WINDOW_DIR,WINDOW,api_id=api_id,identity=identity,others=others,desktop=desktop,temporary_name=NAME,commit=COMMIT)
 host.arm_deadline(measure_end);paused=True;host.run('docker','stop','--time','30',api_id,timeout=45)
 stop_end=time.monotonic()+30
 while not host.no_gpu_owners(desktop) and time.monotonic()<stop_end:time.sleep(1)
 assert host.no_gpu_owners(desktop)
 baseline=host.gpu_snapshot();assert all(baseline[g]['free']>=23040 for g in host.GPUS)
 record_memory_baseline(WINDOW_DIR,WINDOW,baseline)
 env=dict(v.split('=',1) for v in before['Config']['Env'])
 env.update(KADAN_IMAGE_BACKEND='dual',KADAN_IMAGE_DEVICES='[0,1]',KADAN_GPU='1',KADAN_GPU_BUDGET_BYTES='{"0":24159191040,"1":24159191040}',CUDA_VISIBLE_DEVICES='0,1',NVIDIA_VISIBLE_DEVICES=','.join(host.GPUS),PYTHONDONTWRITEBYTECODE='1')
 env['PYTHONPATH']='/run/kadan-window-tools'+(':'+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
 fd,envfile=tempfile.mkstemp(prefix='kadan-probe-env-');os.fchmod(fd,0o600)
 with os.fdopen(fd,'w') as f:
  for k,v in env.items():
   assert '\n' not in v
   f.write(k+'='+v+'\n')
 cmd=['docker','run','-d','--name',NAME,'--label',LABEL+'='+WINDOW['token'],'--network',before['HostConfig']['NetworkMode'],'--memory','256g','--memory-swap','256g','--shm-size','1g','--gpus','"device='+','.join(host.GPUS)+'"','-p','127.0.0.1:18000:8000','--log-driver','json-file','--log-opt','max-size=16m','--log-opt','max-file=4','--env-file',envfile]
 for mount in before['Mounts']:
  spec='type='+mount['Type']+',src='+str(mount.get('Name') if mount['Type']=='volume' else mount['Source'])+',dst='+mount['Destination']
  if not mount['RW']:spec+=',readonly'
  cmd+=['--mount',spec]
 cmd+=['--mount',f'type=bind,src={E},dst=/run/kadan-window-evidence',
  '--mount',f'type=bind,src={TOOLS},dst=/run/kadan-window-tools,readonly',
  '--mount',f'type=bind,src={BUNDLE}/logging.json,dst=/run/kadan-logging.json,readonly','--entrypoint','python',host.IMAGE,*before['Config']['Cmd'][1:],'--log-config','/run/kadan-logging.json']
 try:
  owned=True;host.run(*cmd)
 finally:Path(envfile).unlink(missing_ok=True)
 cgroup,limits=cgroup_identity(host.inspect_container(NAME)['State']['Pid'])
 (E/'actual-limits.json').write_text(json.dumps(limits))
 ready_end=time.monotonic()+180
 while True:
  try:
   status=get('/model-lifecycle')
   if status['state']=='ready':break
   if status['state']=='error':raise RuntimeError(str(status))
  except (OSError,ValueError):pass
  if time.monotonic()>ready_end:raise TimeoutError('Probe text startup')
  host.guards();time.sleep(2)
 (E/'initial-lifecycle.json').write_text(json.dumps(status))
 measure_end=min(measure_end,time.monotonic()+900)
 host.arm_deadline(measure_end)
 (E/'cpu-profile.txt').write_text(host.run('docker','exec',NAME,'cat','/sys/fs/cgroup/cpu.max').stdout)
 if KIND=='a':
  first=submit('A','image-a.json','/v1/images/generations');wait(first)
 elif KIND=='aba':
  first=submit('A1','image-a.json','/v1/images/generations')
  while redis('HGET','kadan:inference:job:'+first.job,'state')=='queued':sample();time.sleep(.1)
  assert redis('HGET','kadan:inference:job:'+first.job,'state')=='running'
  second=submit('B','image-b.json','/v1/images/generations');third=submit('A2','image-a.json','/v1/images/generations')
  assert redis('LRANGE','kadan:inference:pending','0','-1')==[second.job,third.job]
  for call in calls:wait(call)
 elif KIND=='handoff':
  first=submit('text1','text.json','/v1/chat/completions');wait(first)
  image=submit('A','image-a.json','/v1/images/generations');last=submit('text2','text.json','/v1/chat/completions')
  wait(image);wait(last)
 else:
  failed=submit('interrupted-B','image-b.json','/v1/images/generations')
  ranks={};stable=0;active_end=min(time.monotonic()+600,measure_end-180)
  while stable<2:
   sample()
   lines=host.run('docker','top',NAME,'-eo','pid,args').stdout.splitlines()[1:]
   ranks={}
   for line in lines:
    if ' -m api.inference.image.rank_worker ' in line:
     pid=int(line.split(None,1)[0]);tail=line.split(' -m api.inference.image.rank_worker ',1)[1]
     config=json.loads(tail.split(None,1)[1]);assert config['devices']==[0,1]
     ranks[config['rank']]=dict(pid=pid,session=config['session'],starttime=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19])
   rows={int(v[1].strip()):int(v[2].strip()) for v in (l.split(',') for l in host.run('nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits').stdout.splitlines())}
   active=set(ranks)=={0,1} and len({r['session'] for r in ranks.values()})==1 and all(rows.get(r['pid'],0)>=8192 for r in ranks.values())
   stable=stable+1 if active else 0
   if failed.finish is not None:raise RuntimeError('Image failed before active fault trigger: '+str(failed.result or failed.error))
   if time.monotonic()>active_end:raise TimeoutError('Active fault trigger not reached')
   time.sleep(1)
  next_text=submit('recovery-text','text.json','/v1/chat/completions')
  fault=dict(kind=KIND,ranks=ranks,monotonic=time.monotonic())
  if KIND=='cancel':
   failed.connection.sock.shutdown(socket.SHUT_RDWR);failed.connection.close()
  else:
   target=ranks[1];pid=target['pid']
   assert Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]==target['starttime']
   os.kill(pid,signal.SIGKILL)
  release_end=min(time.monotonic()+35,measure_end)
  while True:
   sample()
   pids={int(line.split(',')[1].strip()) for line in host.run('nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits').stdout.splitlines()}
   accounting=get('/model-lifecycle')['memory']['reservations']
   if all(r['pid'] not in pids for r in ranks.values()) and not any(k.startswith('image:ranks:') for k in accounting):break
   if time.monotonic()>release_end:raise TimeoutError('Fault cleanup exceeded shared bound plus driver-probe allowance')
   time.sleep(.2)
  fault['physical_and_accounting_cleanup_seconds']=time.monotonic()-fault['monotonic']
  wait(next_text)
  fault['job_state']=redis('HGET','kadan:inference:job:'+failed.job,'state')
  assert fault['job_state']==('cancelled' if KIND=='cancel' else 'failed')
  (E/'fault.json').write_text(json.dumps(fault,indent=2))
  recover=submit('A' if KIND=='cancel' else 'B','image-a.json' if KIND=='cancel' else 'image-b.json','/v1/images/generations');wait(recover)
 for call in calls:
  if '/images/' in call.route and call.result and call.result['status']==200:
   data=call.result['body']['image'];file=Path('/home/andrew/Documents/Codex/2026-10-08/task-4/home-api-mr119/media/images')/data['id']/'0.png'
   case='B' if call.label=='B' else 'A';result=compare(file,ROOT/f'api-baseline-reviewed-b57db71-{case}'/'trajectory-evidence/baseline',case,BASELINE_COMMIT)
   (E/(call.label+'-comparison.json')).write_text(json.dumps(result));assert result['passed'],result
 assert len(calls)==1 and calls[0].label=='A'
 assert redis('LLEN','kadan:inference:pending')==0
 assert redis('SCARD','kadan:inference:unfinished')==0
 rows=[json.loads(line) for line in (E/'telemetry.jsonl').read_text().splitlines()]
 if KIND in ('a','aba','handoff'):
  pairs={tuple(sorted((r['rank'],r['pid'],r['session']) for r in row['ranks'])) for row in rows if len(row['ranks'])==2}
  assert len(pairs)==1,'Rank pair changed within successful consecutive requests'
  proof=dict(rank_pair=list(next(iter(pairs))))
  if KIND=='handoff':
   status=get('/model-lifecycle');res=status['memory']['reservations']
   assert not any(k.startswith('image:ranks:') and k.endswith(':execution') for k in res)
   assert any(k.startswith('image:ranks:') and k.endswith(':host') for k in res)
   assert any(k.startswith('image:ranks:') and k.endswith(':context') for k in res)
   memory={int(v[1].strip()):int(v[2].strip()) for v in (line.split(',') for line in host.run('nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits').stdout.splitlines())}
   assert all(pid in memory and memory[pid]<=512 for rank,pid,session in proof['rank_pair'])
   proof.update(parked_reservations=res,parked_driver_mib={pid:memory[pid] for rank,pid,session in proof['rank_pair']})
  (E/'residency-proof.json').write_text(json.dumps(proof,indent=2))
 passed=True
except BaseException as exc:
 (E/'failure.json').write_text(json.dumps(dict(type=type(exc).__name__,message=str(exc)[:4000])));raise
finally:
 host.arm_deadline(end)
 try:
  cleanup_end=min(time.monotonic()+30,end);host.arm_deadline(cleanup_end)
  signal.signal(signal.SIGUSR1,signal.SIG_IGN)
  signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
  monitor_failure=close_watchdog(monitor)
  if monitor_failure is not None:passed=False
  logs=None;state=None
  if owned and host.run('docker','inspect',NAME,check=False).returncode==0:
   owned_container=host.inspect_container(NAME)
   assert owned_container['Config'].get('Labels',{}).get(LABEL)==WINDOW['token'],'QUARANTINE: temporary name has another owner'
   owned_id=owned_container['Id']
   host.run('docker','stop','--time','5',owned_id,check=False)
   logs=host.run('docker','logs',owned_id,check=False)
   state=host.inspect_container(owned_id)['State'];assert not state['Running'];host.run('docker','rm',owned_id)
  while not host.no_gpu_owners(desktop) and time.monotonic()<cleanup_end:time.sleep(1)
  assert host.no_gpu_owners(desktop),'QUARANTINE: GPU owners remain'
  if baseline is not None:
   after=host.gpu_snapshot();assert all(after[g]['used']<=baseline[g]['used']+128 for g in host.GPUS),'QUARANTINE: memory not returned'
  host.arm_deadline(end)
  if paused:restore_exact(before,others,COMMIT,E,end,started,passed,identity)
  assert redis('SCARD','kadan:inference:unfinished')==0 and redis('LLEN','kadan:inference:pending')==0
  if logs is not None:(E/'api.log').write_text((logs.stdout+logs.stderr)[-16*1024**2:])
  if state is not None:(E/'container-state.json').write_text(json.dumps(state))
  (E/'requests.json').write_text(json.dumps([dict(label=c.label,job=c.job,start=c.start,finish=c.finish,elapsed_seconds=None if c.finish is None else c.finish-c.start,result=c.result,error=c.error) for c in calls],indent=2))
  if monitor_failure is not None:
   publish_watchdog_error(E,monitor_failure);raise RuntimeError(monitor_failure)
 finally:signal.setitimer(signal.ITIMER_REAL,0);host.COMMAND_DEADLINE=None
