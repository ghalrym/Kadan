"""Pure GPU admission/run/cleanup gates. Provider values are never guessed."""
from dataclasses import dataclass
import csv
import io
from .artifacts import require

MIB=1024**2
UUID='GPU-e30b6419-2c6d-f550-61d6-16166a920dac'
DEVICE_BUDGET=21434868224
ENTRY_FREE=22508610048


@dataclass(frozen=True)
class Snapshot:
    devices: dict
    processes: tuple
    xid: bool


def parse_snapshot(gpu_csv, process_csv, cgroups, journal):
    require(len(gpu_csv)<=65536 and len(process_csv)<=65536 and len(journal)<=1024**2,'telemetry_bound')
    devices={}
    for row in csv.reader(io.StringIO(gpu_csv)):
        require(len(row)==4,'telemetry_gpu_columns')
        index,uuid,free,temp=[s.strip() for s in row]
        require(index.isdecimal() and free.isdecimal() and temp.isdecimal() and uuid not in devices,'telemetry_gpu_values')
        devices[uuid]={'index':int(index),'free_bytes':int(free)*MIB,'temperature':int(temp)}
    require(devices and UUID in devices and devices[UUID]['index']==0,'gpu_uuid_index')
    processes=[]
    for row in csv.reader(io.StringIO(process_csv)):
        require(len(row)==3,'telemetry_process_columns')
        uuid,pid,used=[s.strip() for s in row]
        require(uuid in devices and pid.isdecimal() and int(pid)>0 and used.isdecimal(),'telemetry_process_values')
        require(pid in cgroups and isinstance(cgroups[pid],str),'telemetry_cgroup_missing')
        processes.append({'uuid':uuid,'pid':int(pid),'used_bytes':int(used)*MIB,'cgroup':cgroups[pid].strip()})
    require(len({(p['uuid'],p['pid']) for p in processes})==len(processes),'telemetry_duplicate_process')
    return Snapshot(devices,tuple(processes),'NVRM: Xid' in journal)


class GpuGate:
    def __init__(self, baseline):
        require(baseline.xid is False,'baseline_xid')
        require(not any(p['uuid']==UUID for p in baseline.processes),'baseline_gpu0_owner')
        require(baseline.devices[UUID]['free_bytes']>=ENTRY_FREE,'entry_free_memory')
        require(all(d['temperature']<=65 for d in baseline.devices.values()),'entry_temperature')
        self.baseline=baseline
        self.others={(p['uuid'],p['pid'],p['cgroup']) for p in baseline.processes}

    def evaluate(self,current,cid,cleanup=False):
        require(current.xid is False,'new_xid')
        require(set(current.devices)==set(self.baseline.devices),'gpu_inventory_drift')
        require(all(current.devices[u]['index']==self.baseline.devices[u]['index'] for u in current.devices),'gpu_index_drift')
        require(all(d['temperature']<80 for d in current.devices.values()),'temperature_abort')
        other={(p['uuid'],p['pid'],p['cgroup']) for p in current.processes if p['uuid']!=UUID}
        require(other==self.others,'unexplained_other_gpu_process')
        owned=[p for p in current.processes if p['uuid']==UUID]
        require(all(p['cgroup']==f'0::/system.slice/docker-{cid}.scope' for p in owned),'unexpected_gpu0_owner')
        require(current.devices[UUID]['free_bytes']>=1024**3,'run_free_memory')
        require(sum(p['used_bytes'] for p in owned)<=DEVICE_BUDGET+64*MIB,'owned_device_budget')
        if cleanup:
            require(not owned,'owned_compute_not_gone')
            require(all(current.devices[u]['free_bytes']>=self.baseline.devices[u]['free_bytes']-64*MIB
                        for u in current.devices),'gpu_baseline_not_restored')
        return True
