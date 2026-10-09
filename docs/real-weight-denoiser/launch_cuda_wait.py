"""Bounded wait probe using established baseline/trajectory supervision primitives."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time

import launch_trajectory as host
from launch_api_baseline import restore_exact
from cuda_wait_probe import PLAN, compare
from cuda_wait_monitor import cgroup_identity, task_counters, close_watchdog, publish_watchdog_error

CASES = ("control", "blocking")
PROTOCOL = "cuda-wait-short-v1"
RUN_CAP = 16*1024**2
def settings(case): return dict(PLAN, policy=case, cpus=[0,8], memory_gib=8, stage_seconds=30)
def verify(path, case, commit):
    for device in (0,1):
        row=json.loads((path/f"rank-{device}.json").read_text())
        if row["policy"]!=case or row["device"]!=device or row["commit"]!=commit or row["gpu_uuid"]!=host.GPUS[device]:
            raise ValueError("Probe result identity differs")
        flags=row['flags']
        if flags['after_window']!=flags['after']:
            raise ValueError('Probe flags changed during window')
        if case=='control' and (flags['before'] & 7 == 4 or flags['after_window'] & 7 == 4):
            raise ValueError('Inconclusive: control already BlockingSync; stop before candidate')
        if case=='blocking' and flags['after_window'] & 7 != 4:
            raise ValueError('Candidate BlockingSync unconfirmed')
from bf16_contracts import LIMIT, require_ci
CRITERIA = dict(output_sha256_equal=True, physical_uuid_equal=True, intervention_required=True)

NAME = 'kadan-cuda-wait-reviewed'
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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-reviewed')
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--review-record', type=Path)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--retained-roots', nargs='+', type=Path)
    parser.add_argument('--control-evidence', type=Path)
    args = parser.parse_args()
    if not args.execute_reviewed:
        print(json.dumps(dict(protocol=PROTOCOL, cases={case:settings(case) for case in CASES},
            criteria=CRITERIA, ram_gib=8, swap_bytes=0, cpu_placement=[0,8], shm_gib=1,
            allocator_mib=512, stage_seconds=30, pause_seconds=PAUSE_SECONDS,
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
    if args.case=='blocking':
        if args.control_evidence is None:raise ValueError('Reviewed same-head control evidence required')
        verify(args.control_evidence/'trajectory-evidence','control',commit)
        if not json.loads((args.control_evidence/'pause-result.json').read_text())['restored']:
            raise ValueError('Control API restoration unconfirmed')
        if not any(args.control_evidence.resolve().is_relative_to(p.resolve()) for p in args.retained_roots):
            raise ValueError('Control evidence must be included in retained storage roots')
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
    host.RUN_EVIDENCE = evidence
    (evidence/'review.json').write_text(args.review_record.read_text())
    (evidence/'ci.json').write_text(json.dumps(ci,indent=2))
    (evidence/'storage.json').write_text(json.dumps(dict(retained_bytes=retained,run_cap=RUN_CAP,limit=LIMIT,
        roots=[str(p.resolve()) for p in args.retained_roots])))
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
    monitor = None
    def interrupted(signum, frame): raise InterruptedError(f'Baseline launcher signal {signum}')
    old_term = signal.signal(signal.SIGTERM,interrupted)
    old_int = signal.signal(signal.SIGINT,interrupted)
    old_alarm = signal.signal(signal.SIGALRM,interrupted)
    old_watch = signal.signal(signal.SIGUSR1,interrupted)
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
        stage = min(30,int(measurement_end-time.monotonic())-30)
        if stage<1: raise TimeoutError('Measurement window exhausted')
        command = ['docker','run','-d','--name',NAME,'--network','none','--cpuset-cpus','0,8',
            '--memory','8g','--memory-swap','8g','--shm-size','1g',
            '--log-driver','json-file','--log-opt','max-size=1m','--log-opt','max-file=2',
            '--gpus','"device='+','.join(host.GPUS)+'"',
            '--mount',f'type=bind,src={host.REPO}/docs/real-weight-denoiser,dst=/probe,readonly',
            '--mount',f'type=bind,src={work},dst=/evidence',
            '--mount',f'type=bind,src={host.REPO}/api,dst=/app/api,readonly',
            '--env','PYTHONPATH=/app',
            '--mount',f'type=volume,src={model["Name"]},dst=/models,readonly',
            '--env',f'KADAN_STAGE_SECONDS={stage}','--env','KADAN_EXPECTED_GPU_UUIDS='+json.dumps(host.GPUS),
            '--env','PYTHONDONTWRITEBYTECODE=1',
            '--env','OMP_NUM_THREADS=1','--env','MKL_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1',
            '--entrypoint','python',host.IMAGE,'/probe/supervisor_cuda_wait.py',args.case,commit]
        owned = True  # A timed-out docker create may still have created this unique name.
        stage_end=time.monotonic()+30
        host.arm_deadline(min(stage_end,measurement_end))
        host.run(*command)
        container=host.inspect_container(NAME)
        cgroup,limits=cgroup_identity(container['State']['Pid'])
        (evidence/'actual-limits.json').write_text(json.dumps(limits))
        while host.inspect_container(NAME)['State']['Running']:
            if time.monotonic()>=stage_end: raise TimeoutError('Host 30-second stage deadline')
            if time.monotonic()>=measurement_end: raise TimeoutError('Measurement deadline')
            host.guards()
            sample=task_counters(cgroup)
            with (evidence/'samples.jsonl').open('a') as stream:stream.write(json.dumps(sample)+'\n')
            if sum(p.stat().st_size for p in evidence.rglob('*') if p.is_file()) > RUN_CAP-6*1024**2:
                raise RuntimeError('Evidence cap reached; reserve logs and restoration metadata')
            time.sleep(1)
        state = host.inspect_container(NAME)['State']
        if state['OOMKilled'] or state['ExitCode'] != 0: raise RuntimeError('Baseline process failed')
        verify(work,args.case,commit)
        if args.case=='blocking':
            comparisons=[compare(json.loads((args.control_evidence/'trajectory-evidence'/f'rank-{d}.json').read_text()),
                json.loads((work/f'rank-{d}.json').read_text())) for d in (0,1)]
            (evidence/'comparison.json').write_text(json.dumps(comparisons))
        passed = True
    except BaseException as exc:
        (evidence/'failure.json').write_text(json.dumps(dict(type=type(exc).__name__,message=str(exc)[:2000])))
        raise
    finally:
        host.arm_deadline(end)
        try:
            cleanup_end=min(time.monotonic()+30,end)
            host.arm_deadline(cleanup_end)
            # Ignore a late external interruption during controlled recovery.
            signal.signal(signal.SIGUSR1,signal.SIG_IGN)
            monitor_failure=close_watchdog(monitor)
            if monitor_failure is not None:passed=False
            if owned and host.run('docker','inspect',NAME,check=False).returncode == 0:
                host.run('docker','stop','--time','5',NAME,check=False)
                state = host.inspect_container(NAME)['State']
                (evidence/'container-state.json').write_text(json.dumps(state))
                logs = host.run('docker','logs',NAME,check=False)
                # Docker's rotating log store is bounded; retain an explicit bounded export.
                (evidence/'output.txt').write_text((logs.stdout+logs.stderr)[-2*1024**2:])
                if state['Running']: raise RuntimeError('Probe teardown unconfirmed')
                host.run('docker','rm',NAME)
            while not host.no_gpu_owners(desktop) and time.monotonic()<cleanup_end: time.sleep(1)
            clean = host.no_gpu_owners(desktop)
            if clean and baseline is not None:
                after=host.gpu_snapshot()
                clean=all(after[g]['used']<=baseline[g]['used']+128 for g in host.GPUS)
            if not clean: raise RuntimeError('QUARANTINE: physical cleanup unconfirmed; API stays stopped')
            host.arm_deadline(end)
            if paused:
                restore_exact(before,others,commit,evidence,end,start,passed,identity_digest)
                for command,key in [('SCARD','unfinished'),('LLEN','pending')]:
                    if host.run('docker','exec','kadan-redis-1','redis-cli',command,'kadan:inference:'+key).stdout.strip()!='0':
                        raise RuntimeError('FIFO not empty after restoration')
            if monitor_failure is not None:
                publish_watchdog_error(evidence,monitor_failure)
                raise RuntimeError(monitor_failure)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            host.COMMAND_DEADLINE=None
            signal.signal(signal.SIGTERM,old_term)
            signal.signal(signal.SIGINT,old_int)
            signal.signal(signal.SIGALRM,old_alarm)
            signal.signal(signal.SIGUSR1,old_watch)


if __name__ == '__main__':
    main()
