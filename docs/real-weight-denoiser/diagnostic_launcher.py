"""Host-specific reproduction launcher; explicit execution only, free GPU0 only."""
import argparse
import json,subprocess,time
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    if not parser.parse_args().execute:
        parser.exit(message="Read-only default. Inspect the pinned paths and GPU UUID before --execute.\n")
    root=Path('/home/andrew/Documents/Codex/2026-10-08/task-4');out=root/'block7-diagnostic-shape-control';name='kadan-block7-diagnostic'
    out.mkdir(exist_ok=False)
    def run(*args):return subprocess.run(args,capture_output=True,text=True,timeout=10,check=True)
    assert ' 2,' in run('nvidia-smi','--id=GPU-e30b6419-2c6d-f550-61d6-16166a920dac','--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits').stdout or int(run('nvidia-smi','--id=GPU-e30b6419-2c6d-f550-61d6-16166a920dac','--query-gpu=memory.used','--format=csv,noheader,nounits').stdout.strip())<32
    before=json.loads(run('docker','inspect','kadan-api-1').stdout)[0]
    cmd=['docker','run','-d','--name',name,'--network','none','--cpus','4','--memory','8g','--memory-swap','8g','--gpus','device=GPU-e30b6419-2c6d-f550-61d6-16166a920dac',
    '--mount',f'type=bind,src={root}/Kadan/api,dst=/app/api,readonly','--mount',f'type=bind,src={root}/Kadan/docs/real-weight-denoiser,dst=/probe,readonly',
    '--mount',f'type=bind,src={root}/real-weight-reviewed-9304851/capture,dst=/capture,readonly','--mount',f'type=bind,src={out},dst=/evidence',
    '--env','PYTHONPATH=/app:/probe','--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=1','--entrypoint','python','sha256:eed6c0b1208855f89e27093c2ad6abaada302919260787524e5c901ee56f57aa','/probe/diagnose_block7.py']
    samples=[];started=time.monotonic();run(*cmd)
    try:
     while json.loads(run('docker','inspect',name).stdout)[0]['State']['Running']:
      assert time.monotonic()-started<150,'Diagnostic deadline'
      values=run('nvidia-smi','--id=GPU-e30b6419-2c6d-f550-61d6-16166a920dac','--query-gpu=memory.used,memory.free,temperature.gpu','--format=csv,noheader,nounits').stdout.strip()
      used,free,temp=map(int,values.split(','));assert free>=2048 and temp<85
      temperatures=[int(p.read_text())/1000 for p in Path('/sys/class/hwmon').glob('hwmon*/temp*_input') if p.parent.joinpath('name').read_text().strip() in ('k10temp','coretemp')]
      assert temperatures and max(temperatures)<80
      samples.append(dict(elapsed_s=time.monotonic()-started,gpu_used_mib=used,gpu_temperature_c=temp,cpu_temperature_c=max(temperatures)))
      time.sleep(1)
    finally:
     subprocess.run(['docker','stop','--timeout','5',name],capture_output=True,timeout=15)
     state=json.loads(run('docker','inspect',name).stdout)[0]['State'];(out/'container-state.json').write_text(json.dumps(state));logs=run('docker','logs',name);(out/'output.txt').write_text(logs.stdout+logs.stderr)
     assert not state['Running'];run('docker','rm',name)
     (out/'samples.json').write_text(json.dumps(samples))
     after=json.loads(run('docker','inspect','kadan-api-1').stdout)[0]
     assert after['Id']==before['Id'] and after['State']['StartedAt']==before['State']['StartedAt'] and after['State']['Health']['Status']=='healthy'
     (out/'api-preserved.json').write_text(json.dumps(dict(id=after['Id'],started_at=after['State']['StartedAt'],health=after['State']['Health']['Status'])))
    print(json.dumps(state))

if __name__=="__main__":
    main()
