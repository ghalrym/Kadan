"""Explicit maintenance-window launcher. Default is read-only plan, never execution.

Run with --execute-reviewed COMMIT only after source review and exact-head CI clear.
The operator coordinates the interruption. This script does not establish approval.
"""
import argparse
import hashlib
import json
import signal
from pathlib import Path
import subprocess
import time
import urllib.request

IMAGE = 'sha256:eed6c0b1208855f89e27093c2ad6abaada302919260787524e5c901ee56f57aa'
GPUS = ['GPU-e30b6419-2c6d-f550-61d6-16166a920dac', 'GPU-2a2378dd-08c1-6f69-6317-a253d90e76b3']
REPO = Path(__file__).resolve().parents[2]
API = 'kadan-api-1'
NAME = 'kadan-dual-gpu-reviewed-probe'


def run(*args, check=True, timeout=20):
    return subprocess.run(args, check=check, capture_output=True, text=True, timeout=timeout)


def container_identity(inspect):
    """Exclude runtime state/network addresses that legitimately change on start."""
    return {key: inspect[key] for key in ('Id', 'Image', 'Config', 'HostConfig', 'Mounts', 'Path', 'Args')}


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
    lines = run('nvidia-smi', '--query-gpu=uuid,memory.used,memory.free,temperature.gpu', '--format=csv,noheader,nounits').stdout.splitlines()
    return {v[0].strip(): dict(used=int(v[1]), free=int(v[2]), temperature=int(v[3]))
            for v in (line.split(',') for line in lines)}


def no_gpu_owners():
    lines = run('nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader').stdout.splitlines()
    return not any(line.split(',')[0].strip() in GPUS for line in lines)


def guards():
    snap = gpu_snapshot()
    assert all(snap[g]['free'] >= 256 and snap[g]['temperature'] < 90 for g in GPUS), snap
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    assert int(memory['MemAvailable'].split()[0]) >= 16 * 1024**2, 'Host free memory guard'
    temperatures = [int(p.read_text()) / 1000 for p in Path('/sys/class/hwmon').glob('hwmon*/temp*_input')
                    if p.parent.joinpath('name').read_text().strip() in ('k10temp', 'coretemp')]
    assert temperatures and max(temperatures) < 80, 'CPU temperature unavailable or above guard'
    return dict(gpu=snap, cpu_temperature=max(temperatures))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-reviewed', metavar='COMMIT')
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    if not args.execute_reviewed:
        print(json.dumps(dict(image=IMAGE, gpus=GPUS, cases=['parity/communication', 'peer_exit', 'oom'],
            per_case_host_deadline_s=150, collective_timeout_s=45, container_ram_gib=8,
            torch_reserved_gib_per_rank=2, estimated_interruption_minutes='5–8 including restore',
            note='Requires exact-head review/CI and coordinated API-only pause. No GPU access performed.')))
        return
    assert args.evidence and not args.evidence.exists(), 'Choose a new evidence directory'
    assert run('git', '-C', str(REPO), 'rev-parse', 'HEAD').stdout.strip() == args.execute_reviewed
    assert not run('git', '-C', str(REPO), 'status', '--porcelain').stdout, 'Probe source must be clean'
    assert run('docker','inspect',NAME,check=False).returncode != 0, 'Probe name is already owned'
    args.evidence.mkdir(parents=True)
    evidence = args.evidence.resolve()
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
    (evidence/'before.json').write_text(json.dumps(inspect, indent=2))
    def interrupted(signum, frame):
        raise InterruptedError(f'Launcher interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    paused = False
    baseline = None
    cleanup_confirmed = False
    try:
        # Mark before stopping so a command timeout still enters restoration logic.
        paused = True
        run('docker', 'stop', '--time', '30', api_id, timeout=45)
        assert not inspect_container(api_id)['State']['Running']
        deadline=time.monotonic()+30
        while not no_gpu_owners() and time.monotonic()<deadline:
            time.sleep(1)
        assert no_gpu_owners(), 'Existing GPU ownership did not release'
        baseline=gpu_snapshot()
        assert all(baseline[g]['free'] >= 4096 for g in GPUS)
        guards()
        for case in ('normal', 'peer_exit', 'oom'):
            case_dir=evidence/case
            case_dir.mkdir()
            command=['docker','run','-d','--name',NAME,'--network','none','--cpus','4',
                '--memory','8g','--memory-swap','8g','--shm-size','1g',
                '--gpus','"device='+','.join(GPUS)+'"',
                '--mount',f'type=bind,src={REPO}/api,dst=/app/api,readonly',
                '--mount',f'type=bind,src={REPO}/docs/dual-gpu,dst=/probe,readonly',
                '--mount',f'type=bind,src={case_dir},dst=/evidence',
                '--mount',f'type=volume,src={model_mount["Name"]},dst=/models,readonly',
                '--env','PYTHONPATH=/app','--env','PYTHONDONTWRITEBYTECODE=1',
                '--env','OMP_NUM_THREADS=1','--env','MKL_NUM_THREADS=1','--env','GLOO_SOCKET_IFNAME=lo',
                '--env','NCCL_DEBUG=INFO','--env','NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,SHM,P2P',
                '--env','NCCL_DEBUG_FILE=/evidence/nccl-%h-%p.log',
                '--entrypoint','python',IMAGE,'/probe/rank_supervisor.py']
            if case!='normal': command+=['--failure',case]
            started=time.monotonic()
            run(*command)
            samples=[]
            try:
                while json.loads(run('docker','inspect',NAME).stdout)[0]['State']['Running']:
                    assert time.monotonic()-started < 150, 'Host watchdog expired'
                    samples.append(guards())
                    time.sleep(1)
            finally:
                run('docker','stop','--time','5',NAME,check=False)
                state=json.loads(run('docker','inspect',NAME).stdout)[0]['State']
                (case_dir/'container-state.json').write_text(json.dumps(state))
                (case_dir/'output.txt').write_text(run('docker','logs',NAME,check=False).stdout+run('docker','logs',NAME,check=False).stderr)
                (case_dir/'samples.json').write_text(json.dumps(samples))
                assert not state['Running'], 'Container cleanup unconfirmed'
                run('docker','rm',NAME)
            deadline=time.monotonic()+20
            while not no_gpu_owners() and time.monotonic()<deadline: time.sleep(1)
            assert no_gpu_owners(), 'Rank CUDA cleanup unconfirmed'
            after=gpu_snapshot()
            assert all(after[g]['used'] <= baseline[g]['used']+128 for g in GPUS), 'Physical memory did not return'
            assert not state['OOMKilled'], 'Unexpected container host OOM'
            if case=='normal':
                assert state['ExitCode']==0, 'Parity/communication failed'
                assert all((case_dir/f'communication-rank-{r}.json').exists() for r in range(2))
            else:
                assert state['ExitCode']!=0 and (case_dir/f'injected-{case}.json').exists(), 'Failure injection not observed'
        cleanup_confirmed=True
    finally:
        # No blanket process kills: only this launcher's named container is owned.
        found=run('docker','inspect',NAME,check=False)
        if found.returncode==0:
            run('docker','stop','--time','5',NAME,check=False)
            run('docker','rm',NAME,check=False)
        cleanup_deadline=time.monotonic()+20
        while not no_gpu_owners() and time.monotonic()<cleanup_deadline:
            time.sleep(1)
        cleanup_confirmed = no_gpu_owners()
        if cleanup_confirmed and baseline is not None:
            released=gpu_snapshot()
            cleanup_confirmed=all(released[g]['used']<=baseline[g]['used']+128 for g in GPUS)
        if paused and cleanup_confirmed:
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in source_hashes.items()), 'Live source changed'
            restore_container(inspect)
            deadline=time.monotonic()+180
            while True:
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3) as response:
                        assert response.status==200
                    with urllib.request.urlopen('http://127.0.0.1:8000/model-lifecycle',timeout=3) as response:
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
        elif paused:
            raise RuntimeError('QUARANTINE: rank cleanup unconfirmed; API remains stopped. Investigate before releasing ownership.')


if __name__ == '__main__':
    main()
