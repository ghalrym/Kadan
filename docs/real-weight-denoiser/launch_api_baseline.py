"""Default read-only plan. One reviewed isolated baseline per exact API pause.

Reuse the established launcher's identity/physical/thermal/CI primitives, without
changing its frozen full-trajectory cases or source. No tensor snapshot ladder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time
import urllib.request

import launch_trajectory as host
from api_baseline import CASES, PROTOCOL, RUN_CAP, settings, verify
from bf16_contracts import LIMIT, require_ci
from trajectory_contracts import CRITERIA
from thermal_guard import check_cpu

NAME = 'kadan-api-isolated-baseline'
PAUSE_SECONDS = 1800
RECOVERY_SECONDS = 600


def require_review(record, commit, case):
    if (record.get('protocol') != PROTOCOL or record.get('source_commit') != commit
            or record.get('case') != case or record.get('settings') != settings(case)
            or record.get('criteria') != CRITERIA
            or record.get('decision') != 'approved-for-bounded-execution'
            or not record.get('reviewer') or not record.get('review_reference')):
        raise ValueError('Independent exact-source/case/settings approval required')


def storage(roots, new_root):
    roots = [p.resolve(strict=True) for p in roots]
    if len(roots) != len(set(roots)) or any(a.is_relative_to(b) or b.is_relative_to(a)
            for i,a in enumerate(roots) for b in roots[i+1:]):
        raise ValueError('Distinct non-overlapping retained artifact roots required')
    if any(new_root.resolve().is_relative_to(p) or p.is_relative_to(new_root.resolve()) for p in roots):
        raise ValueError('New evidence must be separate from retained artifacts')
    used = sum(p.stat().st_size for root in roots for p in root.rglob('*') if p.is_file())
    if used + RUN_CAP > LIMIT:
        raise ValueError('Baseline exceeds shared 32 GiB artifact ceiling')
    return used


def restore_exact(before, others, commit, evidence, end, start, passed, identity_digest):
    api_id = before['Id']
    if host.run('git','-C',str(host.REPO),'rev-parse','HEAD').stdout.strip()!=commit or host.run('git','-C',str(host.REPO),'status','--porcelain').stdout:
        raise RuntimeError('Source changed during pause')
    host.restore_container(before)
    ready_end=min(time.monotonic()+180,end)
    while True:
        try:
            with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=host.remaining(end,3)) as response:
                if response.status!=200: raise RuntimeError('API health failed')
            with urllib.request.urlopen('http://127.0.0.1:8000/model-lifecycle',timeout=host.remaining(end,3)) as response: lifecycle=json.load(response)
            restored=host.inspect_container(api_id)
            if lifecycle['state']!='ready' or not restored['State']['Running'] or restored['State'].get('Health',{}).get('Status')!='healthy': raise RuntimeError('API not ready')
            break
        except Exception:
            if time.monotonic()>=ready_end: raise
            time.sleep(2)
    if host.container_identity(restored)!=host.container_identity(before) or host.inspect_container(host.API)['Id']!=api_id:
        raise RuntimeError('API identity/config changed')
    if any(host.inspect_container(name)['Id']!=value for name,value in others.items()): raise RuntimeError('Other service replaced')
    (evidence/'pause-result.json').write_text(json.dumps(dict(restored=True,api_id=api_id,
        config_sha256=identity_digest,elapsed_seconds=time.monotonic()-start,limit_seconds=PAUSE_SECONDS,
        baseline_passed=passed,lifecycle=lifecycle)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-reviewed')
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--review-record', type=Path)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--retained-roots', nargs='+', type=Path)
    args = parser.parse_args()
    if not args.execute_reviewed:
        print(json.dumps(dict(protocol=PROTOCOL, cases={case:settings(case) for case in CASES},
            criteria=CRITERIA, ram_gib=128, swap_bytes=0, cpu_quota=2, shm_gib=1,
            allocator_gib=22, stage_seconds=900, pause_seconds=PAUSE_SECONDS,
            recovery_seconds=RECOVERY_SECONDS, run_cap_bytes=RUN_CAP, gpu_access=False)))
        return
    if not all((args.case,args.review_record,args.evidence,args.retained_roots)) or args.evidence.exists():
        raise ValueError('Case, review, retained roots and a new evidence path required')
    commit = args.execute_reviewed
    require_review(json.loads(args.review_record.read_text()),commit,args.case)
    if host.run('git','-C',str(host.REPO),'rev-parse','HEAD').stdout.strip()!=commit:
        raise ValueError('Source SHA differs')
    if host.run('git','-C',str(host.REPO),'status','--porcelain').stdout:
        raise ValueError('Clean reviewed source required')
    ci = json.loads(host.run('gh','run','list','--repo','ghalrym/Kadan','--commit',commit,
        '--limit','100','--json','name,status,conclusion,event,headSha').stdout)
    require_ci(ci,commit)
    if host.run('docker','inspect',NAME,check=False).returncode == 0:
        raise ValueError('Probe name already owned')
    retained = storage(args.retained_roots,args.evidence)
    if shutil.disk_usage(args.evidence.parent).free < 64*1024**3:
        raise ValueError('Require 64 GiB free disk')
    available = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
    if available < 160*1024**3:
        raise ValueError('Require 160 GiB available host RAM')
    evidence = args.evidence.resolve()
    evidence.mkdir()
    work = evidence/'trajectory-evidence'
    work.mkdir()
    host.THERMAL_EVIDENCE = evidence
    (evidence/'review.json').write_text(args.review_record.read_text())
    (evidence/'ci.json').write_text(json.dumps(ci,indent=2))
    (evidence/'storage.json').write_text(json.dumps(dict(retained_bytes=retained,run_cap=RUN_CAP,limit=LIMIT,
        roots=[str(p.resolve()) for p in args.retained_roots])))
    for index in range(5):
        check_cpu(evidence,'cooldown-admission',limit=60)
        if index < 4: time.sleep(2)
    before = host.inspect_container(host.API)
    if before['Image'] != host.IMAGE or not before['State']['Running']:
        raise ValueError('Unexpected API image/state')
    api_id = before['Id']
    model = next(m for m in before['Mounts'] if m['Destination']=='/var/lib/kadan/models')
    if model['Type'] != 'volume': raise ValueError('Expected existing model volume')
    for command,key in [('SCARD','unfinished'),('LLEN','pending')]:
        if host.run('docker','exec','kadan-redis-1','redis-cli',command,'kadan:inference:'+key).stdout.strip()!='0':
            raise ValueError('FIFO must be idle')
    others = {name:host.inspect_container(name)['Id'] for name in ('kadan-redis-1','kadan-postgres-1','kadan-frontend-1')}
    desktop = host.desktop_baseline(api_id)
    identity_digest = hashlib.sha256(json.dumps(host.container_identity(before),sort_keys=True).encode()).hexdigest()
    (evidence/'identity.json').write_text(json.dumps(dict(api_id=api_id,config_sha256=identity_digest,others=others,
        original_shm_bytes=before['HostConfig']['ShmSize'],desktop=sorted(desktop))))
    start = time.monotonic()
    end = start+PAUSE_SECONDS
    measurement_end = end-RECOVERY_SECONDS
    paused = False
    baseline = None
    owned = False
    passed = False
    def interrupted(signum, frame): raise InterruptedError(f'Baseline launcher signal {signum}')
    old_term = signal.signal(signal.SIGTERM,interrupted)
    old_alarm = signal.signal(signal.SIGALRM,interrupted)
    try:
        host.arm_deadline(measurement_end)
        paused = True
        host.run('docker','stop','--time','30',api_id,timeout=45)
        release_end = min(time.monotonic()+30,measurement_end)
        while not host.no_gpu_owners(desktop) and time.monotonic()<release_end: time.sleep(1)
        if not host.no_gpu_owners(desktop): raise RuntimeError('Existing GPU owners did not release')
        baseline = host.gpu_snapshot()
        if any(baseline[g]['free']<22*1024 for g in host.GPUS): raise RuntimeError('Insufficient GPU headroom')
        host.guards()
        stage = min(900,int(measurement_end-time.monotonic())-30)
        if stage<1: raise TimeoutError('Measurement window exhausted')
        command = ['docker','run','-d','--name',NAME,'--network','none','--cpus','2',
            '--memory','128g','--memory-swap','128g','--shm-size','1g',
            '--log-driver','json-file','--log-opt','max-size=16m','--log-opt','max-file=2',
            '--gpus','device='+host.GPUS[0],
            '--mount',f'type=bind,src={host.REPO}/docs/real-weight-denoiser,dst=/probe,readonly',
            '--mount',f'type=bind,src={work},dst=/evidence',
            '--mount',f'type=volume,src={model["Name"]},dst=/models,readonly',
            '--env',f'KADAN_STAGE_SECONDS={stage}','--env',f'KADAN_EXPECTED_GPU_UUID={host.GPUS[0]}',
            '--env','PYTHONDONTWRITEBYTECODE=1',
            '--env','OMP_NUM_THREADS=1','--env','MKL_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1',
            '--entrypoint','python',host.IMAGE,'/probe/supervisor_api_baseline.py',args.case,commit]
        owned = True  # A timed-out docker create may still have created this unique name.
        host.run(*command)
        while host.inspect_container(NAME)['State']['Running']:
            if time.monotonic()>=measurement_end: raise TimeoutError('Measurement deadline')
            sample = host.guards()
            sample['monotonic']=time.monotonic()
            sample['compute_processes']=host.run('nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits').stdout
            sample['cgroup_memory']=host.run('docker','exec',NAME,'cat','/sys/fs/cgroup/memory.current',check=False).stdout.strip()
            with (evidence/'samples.jsonl').open('a') as stream: stream.write(json.dumps(sample)+'\n')
            if sum(p.stat().st_size for p in evidence.rglob('*') if p.is_file()) > RUN_CAP-32*1024**2:
                raise RuntimeError('Evidence cap reached; reserve logs and restoration metadata')
            time.sleep(1)
        state = host.inspect_container(NAME)['State']
        if state['OOMKilled'] or state['ExitCode'] != 0: raise RuntimeError('Baseline process failed')
        verify(work/'baseline',args.case,commit)
        passed = True
    except BaseException as exc:
        (evidence/'failure.json').write_text(json.dumps(dict(type=type(exc).__name__,message=str(exc)[:2000])))
        raise
    finally:
        host.arm_deadline(end)
        try:
            if owned and host.run('docker','inspect',NAME,check=False).returncode == 0:
                host.run('docker','stop','--time','5',NAME,check=False)
                state = host.inspect_container(NAME)['State']
                (evidence/'container-state.json').write_text(json.dumps(state))
                logs = host.run('docker','logs',NAME,check=False)
                # Docker's rotating log store is bounded; retain an explicit bounded export.
                (evidence/'output.txt').write_text((logs.stdout+logs.stderr)[-16*1024**2:])
                if state['Running']: raise RuntimeError('Probe teardown unconfirmed')
                host.run('docker','rm',NAME)
            cleanup_end = min(time.monotonic()+30,end)
            while not host.no_gpu_owners(desktop) and time.monotonic()<cleanup_end: time.sleep(1)
            clean = host.no_gpu_owners(desktop)
            if clean and baseline is not None:
                after=host.gpu_snapshot()
                clean=all(after[g]['used']<=baseline[g]['used']+128 for g in host.GPUS)
            if not clean: raise RuntimeError('QUARANTINE: physical cleanup unconfirmed; API stays stopped')
            if paused:
                restore_exact(before,others,commit,evidence,end,start,passed,identity_digest)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            host.COMMAND_DEADLINE=None
            signal.signal(signal.SIGTERM,old_term)
            signal.signal(signal.SIGALRM,old_alarm)


if __name__ == '__main__':
    main()
