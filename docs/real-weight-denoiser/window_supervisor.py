"""One absolute review window, including preflight and Docker/API restoration.

No GPU workload or service action on import. The operator alone creates a reviewed
workload. Recovery only removes its uniquely labelled container and starts the
captured original API after ownership/configuration checks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

import launch_trajectory as host
from durable_evidence import append

TOTAL=1800
RECOVERY=900
STOP_GRACE=450
OWNER_GRACE=300
REAP=10
LAUNCH_LIMIT=15
LABEL='kadan.review.window'


def context(start,token,boot):
    return dict(protocol='local-review-window-v2',started=start,work_deadline=start+TOTAL-RECOVERY,
                end=start+TOTAL,operator_end=start+TOTAL-STOP_GRACE,token=token,boot_id=boot)


def write_once(path,value):
    path=Path(path)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
    try:
        data=(json.dumps(value,allow_nan=False)+'\n').encode()
        if len(data)>16384:raise ValueError('window_record_limit')
        view=memoryview(data)
        while view:
            count=os.write(fd,view)
            if count<=0:raise OSError('window_record_write')
            view=view[count:]
        os.fsync(fd)
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:os.close(fd)


def load_context(directory):
    value=json.loads((Path(directory)/'window.json').read_text())
    if (value['protocol']!='local-review-window-v2' or value['end']-value['started']!=TOTAL
            or value['work_deadline']!=value['end']-RECOVERY or value['operator_end']!=value['end']-STOP_GRACE
            or value['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip()):
        raise ValueError('window_context_mismatch')
    return value


def arm(directory,ctx,*,api_id,identity,others,desktop,temporary_name,commit):
    if time.monotonic()>=ctx['work_deadline']:raise TimeoutError('preflight exhausted work deadline')
    if temporary_name!='kadan-api-dual-reviewed':raise ValueError('unexpected workload name')
    write_once(Path(directory)/'recovery.json',dict(token=ctx['token'],end=ctx['end'],api_id=api_id,
        identity=identity,others=others,desktop=[list(row) for row in desktop],temporary_name=temporary_name,
        source_dir=str(host.REPO),source_commit=commit))


def require_memory_baseline(baseline):
    if (not isinstance(baseline,dict) or set(baseline)!=set(host.GPUS)
            or any(type(value) is not int or value<0 for value in baseline.values())):
        raise RuntimeError('QUARANTINE: released-memory baseline unavailable or invalid')
    return baseline


def record_memory_baseline(directory,ctx,rows):
    if time.monotonic()>=ctx['work_deadline']:raise TimeoutError('work deadline exhausted before baseline')
    baseline={gpu:rows[gpu]['used'] for gpu in host.GPUS}
    require_memory_baseline(baseline)
    write_once(Path(directory)/'memory-baseline.json',dict(token=ctx['token'],end=ctx['end'],gpu_used_mib=baseline))


def require_memory_released(baseline,rows):
    require_memory_baseline(baseline)
    if not isinstance(rows,dict):raise RuntimeError('QUARANTINE: GPU memory reading unavailable')
    for gpu,used in baseline.items():
        current=rows.get(gpu,{}).get('used')
        if type(current) is not int or current<0 or current>used+128:
            raise RuntimeError('QUARANTINE: GPU memory not returned to baseline plus 128 MiB')



class DockerOps:
    def __init__(self,end):self.end=end

    def remaining(self,cap,end=None):
        left=min(self.end,end if end is not None else self.end)-time.monotonic()
        if left<=0:raise TimeoutError('absolute restoration deadline exhausted')
        return min(cap,left)

    def run(self,*args,cap=3,end=None):
        return subprocess.run(args,capture_output=True,text=True,timeout=self.remaining(cap,end))

    def inspect(self,name,*,end=None):
        result=self.run('docker','inspect',name,end=end)
        if result.returncode:
            if 'No such object:' in result.stderr or 'No such container:' in result.stderr:return None
            raise RuntimeError('Docker inspection unavailable')
        return json.loads(result.stdout)[0]

    def source_matches(self,record):
        head=self.run('git','-C',record['source_dir'],'rev-parse','HEAD')
        status=self.run('git','-C',record['source_dir'],'status','--porcelain')
        return head.returncode==status.returncode==0 and head.stdout.strip()==record['source_commit'] and not status.stdout.strip()

    def remove(self,container_id):
        result=self.run('docker','rm','--force',container_id,cap=15)
        if result.returncode:raise RuntimeError('owned Docker cleanup failed')

    def physical_clear(self,desktop):
        end=min(self.end,time.monotonic()+30)
        while True:
            # Existing driver identity logic includes PID start times.
            host.COMMAND_DEADLINE=end
            try:clear=host.no_gpu_owners({tuple(row) for row in desktop})
            finally:host.COMMAND_DEADLINE=None
            if clear:return
            if time.monotonic()>=end:raise RuntimeError('QUARANTINE: physical owners remain')
            time.sleep(min(.2,end-time.monotonic()))

    def memory_released(self,baseline):
        previous=host.COMMAND_DEADLINE
        host.COMMAND_DEADLINE=min(self.end,time.monotonic()+3)
        try:rows=host.gpu_snapshot()
        finally:host.COMMAND_DEADLINE=previous
        require_memory_released(baseline,rows)

    def start(self,container_id):
        result=self.run('docker','start',container_id,cap=45)
        if result.returncode:raise RuntimeError('original API start failed')

    def ready(self,container_id):
        end=min(self.end,time.monotonic()+180)
        while True:
            try:
                with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=self.remaining(3,end)) as response:
                    if response.status!=200:raise RuntimeError('health')
                with urllib.request.urlopen('http://127.0.0.1:8000/model-lifecycle',timeout=self.remaining(3,end)) as response:
                    if json.load(response)['state']!='ready':raise RuntimeError('lifecycle')
                row=self.inspect(container_id,end=end)
                if not row or not row['State']['Running'] or row['State'].get('Health',{}).get('Status')!='healthy':
                    raise RuntimeError('container not ready')
                return
            except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired):
                if time.monotonic()>=end:raise
                time.sleep(min(.2,end-time.monotonic()))


def digest(row):
    return hashlib.sha256(json.dumps(host.container_identity(row),sort_keys=True).encode()).hexdigest()


def recover(ctx,record,ops):
    if record is None:return dict(armed=False,restored=False)
    if record['token']!=ctx['token'] or record['end']!=ctx['end']:raise ValueError('recovery_record_mismatch')
    temporary=ops.inspect(record['temporary_name'])
    if temporary is not None:
        if temporary['Config'].get('Labels',{}).get(LABEL)!=ctx['token']:
            raise RuntimeError('QUARANTINE: temporary name has another owner')
        ops.remove(temporary['Id'])
        if ops.inspect(temporary['Id']) is not None or ops.inspect(record['temporary_name']) is not None:
            raise RuntimeError('QUARANTINE: Docker cleanup unconfirmed')
    if not ops.source_matches(record):raise RuntimeError('QUARANTINE: reviewed source changed')
    original=ops.inspect(record['api_id'])
    named=ops.inspect(host.API)
    if original is None or named is None or named['Id']!=record['api_id'] or digest(original)!=record['identity']:
        raise RuntimeError('QUARANTINE: original API identity changed')
    if original['State']['Running'] and temporary is not None:
        raise RuntimeError('QUARANTINE: original API overlaps owned workload')
    if not original['State']['Running']:
        baseline=require_memory_baseline(record.get('gpu_used_mib'))
        ops.physical_clear(record['desktop'])
        ops.memory_released(baseline)
        ops.start(record['api_id'])
    ops.ready(record['api_id'])
    if digest(ops.inspect(record['api_id']))!=record['identity']:
        raise RuntimeError('original API configuration changed during restoration')
    for name,identity in record['others'].items():
        row=ops.inspect(name)
        if row is None or row['Id']!=identity:raise RuntimeError('other service identity changed')
    return dict(armed=True,restored=True,api_id=record['api_id'])


def supervise(child,ctx,*,stopping,clock=time.monotonic,sleep=time.sleep):
    while child.poll() is None and clock()<ctx['work_deadline'] and not stopping():sleep(.1)
    if child.poll() is None:
        child.terminate()  # Once only: later stop signals must not interrupt restore.
        grace=min(ctx['work_deadline']+OWNER_GRACE,clock()+OWNER_GRACE)
        while child.poll() is None and clock()<grace:sleep(.1)
        if child.poll() is None:
            child.kill();child.wait(timeout=REAP)
    # The service manager always runs recovery in ExecStopPost, including after
    # supervisor death. Do not spend its independent restoration reserve here.
    return child.returncode


def recovery_action(directory,ctx):
    path=Path(directory)/'recovery.json'
    try:
        record=json.loads(path.read_text()) if path.exists() else None
        baseline_path=Path(directory)/'memory-baseline.json'
        if record is not None and baseline_path.exists():
            baseline=json.loads(baseline_path.read_text())
            if baseline['token']!=ctx['token'] or baseline['end']!=ctx['end']:
                raise ValueError('memory_baseline_context_mismatch')
            record=dict(record,gpu_used_mib=require_memory_baseline(baseline.get('gpu_used_mib')))
        result=recover(ctx,record,DockerOps(ctx['end']))
    except Exception as exc:
        append(Path(directory)/'recovery-events.jsonl',dict(unix_time=time.time(),monotonic=time.monotonic(),
            restored=False,error=type(exc).__name__+': '+str(exc)[:2048]))
        raise
    append(Path(directory)/'recovery-events.jsonl',dict(unix_time=time.time(),monotonic=time.monotonic(),**result))
    return result


class Operator:
    def __init__(self,argv,env):self.process=subprocess.Popen(argv,env=env,start_new_session=True)
    def poll(self):return self.process.poll()
    @property
    def returncode(self):return self.process.returncode
    def terminate(self):self.process.terminate()
    def kill(self):
        try:os.killpg(self.process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    def wait(self,timeout):return self.process.wait(timeout=timeout)


def require_launch_budget(ctx,now):
    if now-ctx['started']>LAUNCH_LIMIT or now>=ctx['work_deadline']:
        raise TimeoutError('service activation missed launch budget; no operator started')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True)
    parser.add_argument('--recover',action='store_true');parser.add_argument('argv',nargs=argparse.REMAINDER)
    args=parser.parse_args();ctx=load_context(args.directory)
    # Repeated service stop requests become a flag, never an exception in cleanup.
    stopped=[False]
    def stop(signum,frame):stopped[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.recover:recovery_action(args.directory,ctx);return
    argv=args.argv[1:] if args.argv and args.argv[0]=='--' else args.argv
    if not argv:raise RuntimeError('operator command missing')
    require_launch_budget(ctx,time.monotonic())
    env=dict(os.environ,KADAN_WINDOW_DIRECTORY=str(Path(args.directory).resolve()))
    child=Operator(argv,env)
    code=supervise(child,ctx,stopping=lambda:stopped[0])
    raise SystemExit(code if code is not None else 1)


if __name__=='__main__':main()
