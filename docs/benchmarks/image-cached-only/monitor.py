"""Host-side bounded supervisor; touches only the named isolated test container."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'evidence'
CONFIG=json.loads((ROOT/'manifest.json').read_text())
NAME=CONFIG['container']
UUID=CONFIG['physical_gpu_uuid']


def command(args):
    return subprocess.check_output(args,text=True,timeout=10).strip()


def sample():
    query='uuid,temperature.gpu,memory.free,memory.used,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown'
    rows=list(csv.reader(command(['nvidia-smi','--query-gpu='+query,'--format=csv,noheader,nounits']).splitlines()))
    gpus={row[0].strip():{'temperature_c':float(row[1]),'free_mib':float(row[2]),'used_mib':float(row[3]),
        'thermal':row[4].strip(),'power_brake':row[5].strip()} for row in rows}
    values={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
    temps=[]
    for label in Path('/sys/class/hwmon').glob('hwmon*/temp*_label'):
        if label.read_text().strip()=='Tctl':temps.append(float(label.with_name(label.name.replace('_label','_input')).read_text())/1000)
    if not temps:raise RuntimeError('CPU Tctl telemetry unavailable')
    return {'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'cpu_tctl_c':max(temps),
        'host_available_bytes':int(values['MemAvailable'].split()[0])*1024,'gpus':gpus}


def cgroup_sample(pid):
    membership=next(line.split(':',2)[2] for line in Path(f'/proc/{pid}/cgroup').read_text().splitlines() if line.startswith('0::'))
    base=Path('/sys/fs/cgroup')/membership.lstrip('/')
    stat=dict(line.split() for line in (base/'memory.stat').read_text().splitlines())
    return {'current':int((base/'memory.current').read_text()), 'peak':int((base/'memory.peak').read_text()), 'max':int((base/'memory.max').read_text()), 'anon':int(stat['anon']), 'file':int(stat['file']), 'shmem':int(stat['shmem']), 'events':(base/'memory.events').read_text()}



def reason(row, initial=False):
    gpu=row['gpus'][UUID]
    if row.get('cgroup',{}).get('current',0) > CONFIG['host_budget_bytes'] - 2*1024**3: return 'cgroup RAM safety margin'
    cache=ROOT/'compiler-cache'
    if cache.exists() and sum(x.stat().st_size for x in cache.rglob('*') if x.is_file()) > CONFIG['compiler_cache_limit_bytes']:
        return 'compiler disk cache limit'
    if row['cpu_tctl_c']>=CONFIG['cpu_abort_c']:return 'CPU Tctl threshold'
    if gpu['temperature_c']>=CONFIG['gpu_abort_c']:return 'GPU1 temperature threshold'
    if gpu['thermal']!='Not Active' or gpu['power_brake']!='Not Active':return 'GPU1 thermal/power-brake flag'
    if row['host_available_bytes']<(CONFIG['physical_host_start_min_bytes'] if initial else CONFIG['host_abort_available_bytes']):return 'host RAM headroom'
    minimum=CONFIG['gpu_budget_bytes']/1024**2 if initial else CONFIG['gpu_abort_free_mib']
    if gpu['free_mib']<minimum:return 'GPU1 physical headroom'
    return None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:raise SystemExit('Explicit --execute required')
    baseline=sample();(OUT/'physical-before.json').write_text(json.dumps(baseline,indent=2))
    why=reason(baseline,True)
    if why:raise SystemExit('Preflight rejected: '+why)
    (OUT/'compute-before.txt').write_text(command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv']))
    start=time.monotonic();aborted=None;abort_at=None;log_process=None
    command(['docker','start',NAME])
    with (OUT/'container.log').open('w') as logs:
        log_process=subprocess.Popen(['docker','logs','--follow','--timestamps',NAME],stdout=logs,stderr=subprocess.STDOUT)
        try:
            while True:
                status=json.loads(command(['docker','inspect',NAME,'--format','{{json .State}}']))
                if not status['Running']:
                    (OUT/'container-exit.json').write_text(json.dumps(status,indent=2));break
                try:
                    row=sample();row['cgroup']=cgroup_sample(status['Pid']);row['compiler_cache_bytes']=sum(x.stat().st_size for x in (ROOT/'compiler-cache').rglob('*') if x.is_file());row['elapsed_s']=round(time.monotonic()-start,3)
                    with (OUT/'physical.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                    why=reason(row)
                except Exception as exc:
                    why='telemetry failure: '+repr(exc)
                if time.monotonic()-start>CONFIG['run_deadline_s']:why='overall run deadline'
                if (OUT/'ABORT').exists():why=(OUT/'ABORT').read_text()
                if why and aborted is None:
                    aborted=why;abort_at=time.monotonic();(OUT/'ABORT').write_text(why)
                    print('ABORT '+why,flush=True)
                    # Deliver an immediate cooperative signal as well as the file.
                    command(['docker','kill','--signal=SIGTERM',NAME])
                if abort_at is not None and time.monotonic()-abort_at>CONFIG['cleanup_deadline_s']:
                    command(['docker','kill','--signal=SIGKILL',NAME]);print('Forced isolated-container kill after cleanup deadline',flush=True)
                time.sleep(5)
        finally:
            if log_process:
                try:log_process.wait(timeout=10)
                except subprocess.TimeoutExpired:log_process.terminate();log_process.wait(timeout=5)
    final=sample();(OUT/'physical-after.json').write_text(json.dumps(final,indent=2))
    (OUT/'compute-after.txt').write_text(command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv']))
    summary={'elapsed_s':round(time.monotonic()-start,3),'abort_reason':aborted,'container':status}
    (OUT/'supervisor-result.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
