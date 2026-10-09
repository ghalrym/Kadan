"""Replay retained step39 contexts without recapture or mutation. Default is read-only plan, never execution.

Run with --execute-reviewed COMMIT only after source review and exact-head CI clear.
The operator coordinates the interruption. This script does not establish approval.
"""
import argparse
import hashlib
import json
import signal
import shutil
from pathlib import Path
import subprocess
import time
import urllib.request

from bf16_contracts import LIMIT, ULYSSES_SOURCE, require_ci, verify_capture
from trace_binding import sha256
from replay_only_contracts import PROTOCOL, CAPTURE_COMMIT, CONTEXT_SHA, require_review, verify_retained

IMAGE = 'sha256:eed6c0b1208855f89e27093c2ad6abaada302919260787524e5c901ee56f57aa'
GPUS = ['GPU-e30b6419-2c6d-f550-61d6-16166a920dac', 'GPU-2a2378dd-08c1-6f69-6317-a253d90e76b3']
REPO = Path(__file__).resolve().parents[2]
API = 'kadan-api-1'
NAME = 'kadan-step39-replay-only-probe'
PAUSE_SECONDS = 35*60
RECOVERY_SECONDS = 10*60
COMMAND_DEADLINE = None
RUN_EVIDENCE = None


class PauseBudget:
    def __init__(self, started):
        self.started = started
        self.end = started + PAUSE_SECONDS
        self.measurement_end = self.end - RECOVERY_SECONDS

    def stage_seconds(self):
        # Leave30s for launch/supervisor termination before recovery reserve.
        seconds = min(900, int(self.measurement_end-time.monotonic())-30)
        if seconds < 1:
            raise TimeoutError('Measurement budget exhausted; restore API now')
        return seconds


def remaining(deadline, cap):
    seconds = min(cap, deadline-time.monotonic())
    if seconds <= 0:
        raise TimeoutError('Overall API pause deadline exhausted')
    return seconds


def arm_deadline(deadline):
    global COMMAND_DEADLINE
    COMMAND_DEADLINE = deadline
    signal.setitimer(signal.ITIMER_REAL, remaining(deadline, PAUSE_SECONDS))


def run(*args, check=True, timeout=20):
    if COMMAND_DEADLINE is not None:
        timeout = remaining(COMMAND_DEADLINE, timeout)
    return subprocess.run(args, check=check, capture_output=True, text=True, timeout=timeout)


def container_identity(inspect):
    """Exclude runtime state/network addresses that legitimately change on start."""
    result = {key: inspect[key] for key in ('Id', 'Image', 'Config', 'HostConfig', 'Mounts', 'Path', 'Args')}
    result['Mounts'] = sorted(result['Mounts'], key=lambda mount: mount['Destination'])
    return result


def inspect_container(container_id):
    return json.loads(run('docker', 'inspect', container_id).stdout)[0]


def restore_container(before):
    """Start only the captured container; never reevaluate deployment files."""
    container_id = before['Id']
    assert container_identity(inspect_container(container_id)) == container_identity(before), 'Stopped API identity/config changed'
    run('docker', 'start', container_id, timeout=45)
    restored = inspect_container(container_id)
    assert container_identity(restored) == container_identity(before), 'Restored API identity/config changed'
    assert restored['State']['Running'], 'Restored API is not running'
    return restored


def gpu_snapshot():
    lines = run('nvidia-smi', '--query-gpu=uuid,memory.used,memory.free', '--format=csv,noheader,nounits').stdout.splitlines()
    return {v[0].strip(): dict(used=int(v[1]), free=int(v[2]))
            for v in (line.split(',') for line in lines)}


def gpu_owners():
    lines = run('nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader').stdout.splitlines()
    owners = set()
    for line in lines:
        gpu, pid = (part.strip() for part in line.split(','))
        if gpu not in GPUS:
            continue
        try:
            # starttime rejects PID reuse; comm is inspected separately below.
            stat = Path('/proc', pid, 'stat').read_text()
        except FileNotFoundError:
            continue
        starttime = stat.rsplit(')', 1)[1].split()[19]
        owners.add((gpu, pid, starttime))
    return owners


def desktop_baseline(api_id):
    api_pids = set(run('docker', 'top', api_id, '-eo', 'pid').stdout.splitlines()[1:])
    api_pids = {pid.strip() for pid in api_pids}
    desktop = {owner for owner in gpu_owners() if owner[1] not in api_pids}
    allowed = {'cosmic-workspac', 'cosmic-files-ap', 'xdg-desktop-por', 'chrome'}
    for _, pid, _ in desktop:
        assert Path('/proc', pid, 'comm').read_text().strip() in allowed, 'Unrecognized external GPU owner; do not pause'
    return desktop


def no_gpu_owners(desktop):
    # Existing desktop contexts may disappear; new/reused PIDs cannot pass.
    return gpu_owners() <= desktop


def guards():
    snap = gpu_snapshot()
    assert all(snap[g]['free'] >= 256 for g in GPUS), snap
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    assert int(memory['MemAvailable'].split()[0]) >= 16 * 1024**2, 'Host free memory guard'
    phases={}
    for path in (RUN_EVIDENCE/'holdout-replay-evidence').glob('phase-rank-*.json'):
        phases[path.name]=json.loads(path.read_text())
    return dict(gpu=snap,workload=phases)


def main():
    global COMMAND_DEADLINE, RUN_EVIDENCE
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-reviewed', metavar='COMMIT')
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--review-record', type=Path)
    parser.add_argument('--retained-evidence',type=Path)
    parser.add_argument('--prior-step20-evidence',type=Path)
    args = parser.parse_args()
    if not args.execute_reviewed:
        print(json.dumps(dict(image=IMAGE, gpus=GPUS, protocol=PROTOCOL, cases=['replay-only-step-39'],shared_weights=True,retained_context_sha256=CONTEXT_SHA,cpu_quota=2,intraop_per_rank=1,interop_per_rank=1, prior_fp32='failed',
            overall_pause_deadline_s=PAUSE_SECONDS, recovery_reserve_s=RECOVERY_SECONDS, per_case_host_deadline_s=930, collective_timeout_s=120, container_ram_gib=128, capture_disk_gib=32,
            torch_reserved_gib_per_rank=22, estimated_interruption_minutes='up to 35 including restore',
            note='Requires exact-head review/CI and coordinated API-only pause. No GPU access performed.')))
        return
    assert args.evidence and not args.evidence.exists(), 'Choose a new evidence directory'
    assert args.capture and args.review_record, 'Existing capture and independent review record required'
    assert args.retained_evidence and args.prior_step20_evidence, 'Retained step39 and passed step20 evidence required'
    require_review(json.loads(args.review_record.read_text()),args.execute_reviewed)
    ci=json.loads(run('gh','run','list','--repo','ghalrym/Kadan','--commit',args.execute_reviewed,'--limit','100','--json','name,status,conclusion,event,headSha').stdout)
    require_ci(ci,args.execute_reviewed)
    assert sha256(REPO/'api/inference/image/sequence_parallel_attention.py')==ULYSSES_SOURCE, 'Ulysses arithmetic changed'
    capture=args.capture.resolve()
    base=verify_capture(capture)
    retained=args.retained_evidence.resolve()
    verify_retained(retained,base,args.prior_step20_evidence)
    contexts=retained/'contexts'
    assert sum(p.stat().st_size for p in capture.iterdir())<LIMIT, 'Capture exceeds artifact envelope'
    assert run('git', '-C', str(REPO), 'rev-parse', 'HEAD').stdout.strip() == args.execute_reviewed
    assert not run('git', '-C', str(REPO), 'status', '--porcelain').stdout, 'Probe source must be clean'
    assert run('docker','inspect',NAME,check=False).returncode != 0, 'Probe name is already owned'
    assert shutil.disk_usage(args.evidence.parent).free >= 64*1024**3, 'Require64 GiB disk headroom'
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    assert available >= 160*1024**3, 'Require160 GiB available host RAM'
    args.evidence.mkdir(parents=True)
    evidence = args.evidence.resolve()
    RUN_EVIDENCE=evidence
    prior_bytes=sum(p.stat().st_size for root in (retained,args.prior_step20_evidence) for p in root.rglob('*') if p.is_file())
    base_bytes=sum(p.stat().st_size for p in capture.iterdir())
    assert base_bytes+prior_bytes+512*1024**2<=LIMIT, 'Retained artifacts plus evidence reserve exceed envelope'
    (evidence/'retained-admission.json').write_text(json.dumps(dict(context_manifest_sha256=CONTEXT_SHA,capture_commit=CAPTURE_COMMIT,
        base_bytes=base_bytes,prior_bytes=prior_bytes,evidence_reserve=512*1024**2,limit=LIMIT)))
    (evidence/'review-record.json').write_text(args.review_record.read_text())
    (evidence/'exact-head-ci.json').write_text(json.dumps(ci,indent=2))
    inspect = inspect_container(API)
    api_id = inspect['Id']
    assert inspect['Image'] == IMAGE and inspect['State']['Running']
    assert inspect['Config']['Labels']['org.opencontainers.image.revision'] == 'dea7b0fade2b068b8e142a1666bf39e495ad0d99'
    mounts = inspect['Mounts']
    model_mount = next(m for m in mounts if m['Destination'] == '/var/lib/kadan/models')
    assert model_mount['Type'] == 'volume'
    source_mount = next(m for m in mounts if m['Destination'] == '/app/api')
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(source_mount['Source']).rglob('*.py')}
    others = {n: json.loads(run('docker','inspect', n).stdout)[0]['Id']
              for n in ('kadan-redis-1','kadan-postgres-1','kadan-frontend-1')}
    for command, key in [('SCARD','unfinished'), ('LLEN','pending')]:
        assert run('docker','exec','kadan-redis-1','redis-cli',command,'kadan:inference:'+key).stdout.strip() == '0', 'Live queue not idle'
    desktop = desktop_baseline(api_id)
    (evidence/'desktop-baseline.json').write_text(json.dumps(sorted(desktop), indent=2))
    (evidence/'before.json').write_text(json.dumps(inspect, indent=2))
    def interrupted(signum, frame):
        raise InterruptedError(f'Launcher interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    def expired(signum, frame):
        raise TimeoutError('API pause phase deadline reached; stop measurement and recover')
    signal.signal(signal.SIGALRM, expired)
    paused = False
    baseline = None
    cleanup_confirmed = False
    budget = PauseBudget(time.monotonic())
    (evidence/'pause-budget.json').write_text(json.dumps(dict(overall_s=PAUSE_SECONDS, recovery_reserve_s=RECOVERY_SECONDS, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))))
    try:
        arm_deadline(budget.measurement_end)
        # Mark before stopping so a command timeout still enters restoration logic.
        paused = True
        run('docker', 'stop', '--time', '30', api_id, timeout=45)
        assert not inspect_container(api_id)['State']['Running']
        deadline=time.monotonic()+30
        while not no_gpu_owners(desktop) and time.monotonic()<deadline:
            time.sleep(1)
        assert no_gpu_owners(desktop), 'Existing GPU ownership did not release'
        baseline=gpu_snapshot()
        assert all(baseline[g]['free'] >= 22*1024 for g in GPUS)
        guards()
        for case in ('holdout-replay',):
            arm_deadline(budget.measurement_end)
            stage_seconds=budget.stage_seconds()
            case_dir=evidence/(case+'-evidence')
            case_dir.mkdir()
            command=['docker','run','-d','--name',NAME,'--network','none','--cpus','2',
                '--memory','128g','--memory-swap','128g','--shm-size','1g',
                '--gpus','"device='+','.join(GPUS)+'"',
                '--mount',f'type=bind,src={REPO}/api,dst=/app/api,readonly',
                '--mount',f'type=bind,src={REPO}/docs/real-weight-denoiser,dst=/probe,readonly',
                '--mount',f'type=bind,src={case_dir},dst=/evidence',
                '--mount',f'type=bind,src={capture},dst=/capture,readonly',
                '--mount',f'type=bind,src={contexts},dst=/contexts,readonly',
                '--mount',f'type=volume,src={model_mount["Name"]},dst=/models,readonly',
                '--env',f'KADAN_STAGE_SECONDS={stage_seconds}',
                '--env','KADAN_HELDOUT_STEP=39','--env','KADAN_RECORD_PHASE=1',
                '--env',f'KADAN_CAPTURE_COMMIT={CAPTURE_COMMIT}',
                '--env',f'KADAN_PRIOR_EVIDENCE_BYTES={prior_bytes}',
                '--env',f'KADAN_IMAGE_DIGEST={IMAGE}',
                '--env',f'KADAN_REVIEWED_COMMIT={args.execute_reviewed}',
                '--env','PYTHONPATH=/app','--env','PYTHONDONTWRITEBYTECODE=1',
                '--env','OMP_NUM_THREADS=1','--env','MKL_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1','--env','NUMEXPR_NUM_THREADS=1','--env','GLOO_SOCKET_IFNAME=lo',
                '--env','NCCL_DEBUG=INFO','--env','NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,SHM,P2P',
                '--env','NCCL_DEBUG_FILE=/evidence/nccl-%h-%p.log',
                '--entrypoint','python',IMAGE,'/probe/supervisor_replay_only.py',case]
            started=time.monotonic()
            run(*command)
            samples=[]
            try:
                while json.loads(run('docker','inspect',NAME).stdout)[0]['State']['Running']:
                    assert time.monotonic() < min(started+930,budget.measurement_end), 'Host measurement deadline expired'
                    sample=guards()
                    sample['utilization_csv']=run('nvidia-smi','--query-gpu=uuid,utilization.gpu,utilization.memory,power.draw','--format=csv,noheader,nounits').stdout
                    sample['container_memory']=run('docker','exec',NAME,'cat','/sys/fs/cgroup/memory.current',check=False).stdout.strip()
                    sample['cpu_stat']=run('docker','exec',NAME,'cat','/sys/fs/cgroup/cpu.stat',check=False).stdout
                    samples.append(sample)
                    assert sum(p.stat().st_size for p in capture.iterdir())+sum(p.stat().st_size for p in evidence.rglob('*') if p.is_file())+prior_bytes<=LIMIT, 'Capture disk guard'
                    time.sleep(1)
            finally:
                # Cleanup consumes the reserved recovery budget, not fresh time.
                arm_deadline(budget.end)
                run('docker','stop','--time','5',NAME,check=False)
                state=json.loads(run('docker','inspect',NAME).stdout)[0]['State']
                (case_dir/'container-state.json').write_text(json.dumps(state))
                (case_dir/'output.txt').write_text(run('docker','logs',NAME,check=False).stdout+run('docker','logs',NAME,check=False).stderr)
                (case_dir/'samples.json').write_text(json.dumps(samples))
                assert not state['Running'], 'Container cleanup unconfirmed'
                run('docker','rm',NAME)
            deadline=time.monotonic()+20
            while not no_gpu_owners(desktop) and time.monotonic()<deadline: time.sleep(1)
            assert no_gpu_owners(desktop), 'Rank CUDA cleanup unconfirmed'
            after=gpu_snapshot()
            assert all(after[g]['used'] <= baseline[g]['used']+128 for g in GPUS), 'Physical memory did not return'
            assert not state['OOMKilled'], 'Unexpected container host OOM'
            assert state['ExitCode']==0, 'Real-weight stage failed; stop ladder'
            for rank in range(2):
                result=json.loads((case_dir/f'verdict-rank-{rank}.json').read_text())
                assert result['heldout_step']=='passed' and result['step_index']==39 and result['fp32_protocol']=='failed'
                assert result['context_manifest_sha256']==CONTEXT_SHA
        cleanup_confirmed=True
    finally:
        arm_deadline(budget.end)
        # No blanket process kills: only this launcher's named container is owned.
        found=run('docker','inspect',NAME,check=False)
        if found.returncode==0:
            run('docker','stop','--time','5',NAME,check=False)
            run('docker','rm',NAME,check=False)
        cleanup_deadline=time.monotonic()+20
        while not no_gpu_owners(desktop) and time.monotonic()<cleanup_deadline:
            time.sleep(1)
        cleanup_confirmed = no_gpu_owners(desktop)
        if cleanup_confirmed and baseline is not None:
            released=gpu_snapshot()
            cleanup_confirmed=all(released[g]['used']<=baseline[g]['used']+128 for g in GPUS)
        if paused and cleanup_confirmed:
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in source_hashes.items()), 'Live source changed'
            restore_container(inspect)
            deadline=min(time.monotonic()+180,budget.end)
            while True:
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=remaining(budget.end,3)) as response:
                        assert response.status==200
                    with urllib.request.urlopen('http://127.0.0.1:8000/model-lifecycle',timeout=remaining(budget.end,3)) as response:
                        lifecycle=json.load(response)
                    assert lifecycle['state']=='ready', lifecycle
                    state=inspect_container(api_id)['State']
                    assert state['Running'] and state.get('Health', {}).get('Status')=='healthy', state
                    (evidence/'restored-lifecycle.json').write_text(json.dumps(lifecycle))
                    break
                except Exception:
                    if time.monotonic()>deadline: raise
                    time.sleep(2)
            assert all(json.loads(run('docker','inspect',n).stdout)[0]['Id']==i for n,i in others.items())
            restored=inspect_container(api_id)
            assert container_identity(restored)==container_identity(inspect), 'Ready API identity/config changed'
            assert restored['State']['Running'] and restored['State'].get('Health', {}).get('Status')=='healthy', 'Container health not ready'
            assert inspect_container(API)['Id']==api_id, 'API name no longer resolves to the captured container'
            (evidence/'restored.json').write_text(json.dumps(restored, indent=2))
            (evidence/'pause-result.json').write_text(json.dumps(dict(elapsed_s=time.monotonic()-budget.started, limit_s=PAUSE_SECONDS, restored=True)))
        elif paused:
            raise RuntimeError('QUARANTINE: rank cleanup unconfirmed; API remains stopped. Investigate before releasing ownership.')

    # Retained inputs and the interrupted attempt remain immutable, even after a pass.
    (evidence/'replay-complete.json').write_text(json.dumps(dict(context_manifest_sha256=CONTEXT_SHA,status='passed',contexts_retained=True)))
    signal.setitimer(signal.ITIMER_REAL,0)
    COMMAND_DEADLINE = None


if __name__ == '__main__':
    main()
