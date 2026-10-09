"""One-BOS GPU correctness transport; inert unless an exact reviewed plan is supplied.

Snapshot hash/writer exclusion and reference capture are prior plan gates.
No service stop/start, network creation, benchmark loop or tolerance override.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time

from .ancestor_observer import AncestorObserver
from .artifacts import ExclusiveOutput, require
from .capture import REPO
from .capture_actual import bounded_json, plain_path
from .container_stage import DockerReferenceStage, log_limit
from .gpu_gate import GpuGate, Snapshot, UUID, parse_snapshot

IMAGE='sha256:5ef7c20e69bf87bebcd2af6eb581055348749227b36b0f76f95cd50600bfb75a'
OVERLAY_REVISION='e52cae04ea9edf699e006c58d2c61f0da270bfc7'
# model_correctness.cpp reserves diagnostics alongside Model::host_bytes(o).
NATIVE_HOST_BYTES=271525888
VOCABULARY_SIZE=248320
LOGITS_DIAGNOSTICS_BYTES=VOCABULARY_SIZE * 4  # vocab * sizeof(float)
HOST_BYTES=NATIVE_HOST_BYTES + LOGITS_DIAGNOSTICS_BYTES
NATIVE_ARGS=['/opt/kadan/bin/kadan-model-correctness','--execute','/snapshot','0','1',
             str(HOST_BYTES),'21434868224','536870912','900','/evidence/native.capture','248044']


def query_limit():resource.setrlimit(resource.RLIMIT_FSIZE,(1024**2,1024**2))


def process_start_time(raw):
    # /proc/PID/stat comm may contain spaces and parentheses; field 22 follows it.
    tail=raw[raw.rfind(')')+2:].split()
    require(len(tail)>=20 and tail[19].isdecimal(),'telemetry_process_stat')
    return int(tail[19])


def journal_records(raw,cursor):
    records=[json.loads(line) for line in raw.splitlines() if line.strip()]
    require(records and records[0].get('__CURSOR')==cursor,'kernel_journal_cursor_unavailable')
    boot=records[0].get('_BOOT_ID')
    require(isinstance(boot,str) and len(boot)==32,'kernel_journal_boot_unknown')
    for record in records:
        require(record.get('_TRANSPORT')=='kernel' and record.get('_BOOT_ID')==boot
                and isinstance(record.get('__CURSOR'),str)
                and isinstance(record.get('MESSAGE'),str),'kernel_journal_incomplete')
    return records[-1]['__CURSOR'],'\n'.join(r['MESSAGE'] for r in records)


class HostTelemetry:
    def __init__(self,journal_cursor,evidence=None):
        require(isinstance(journal_cursor,str) and 0<len(journal_cursor)<=4096 and '\n' not in journal_cursor,'journal_cursor')
        self.cursor=journal_cursor
        self.children=[]
        self.evidence=evidence
        self.audit_bytes=0
        self.sequence=0

    def command(self,args,timeout):
        require(timeout>0,'telemetry_deadline')
        with tempfile.TemporaryFile() as stdout,tempfile.TemporaryFile() as stderr:
            child=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,preexec_fn=query_limit)
            self.children.append(child)
            expired=None
            try:code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                expired=error;child.kill();code=child.poll()
            stdout.seek(0);out=stdout.read(1024**2+1)
            stderr.seek(0);err=stderr.read(1024**2+1)
            record=json.dumps({'argv':args,'exit':code,'stdout':out.decode('utf-8',errors='replace'),
                               'stderr':err.decode('utf-8',errors='replace')}).encode()
            self.audit_bytes+=len(record)
            require(self.audit_bytes<=16*1024**2,'telemetry_audit_bound')
            if self.evidence is not None:
                log=ExclusiveOutput(self.evidence/f'telemetry-{self.sequence:05d}.json')
                try:log.write(record);log.finish()
                finally:log.close()
            self.sequence+=1
            if expired is not None:raise expired
            require(code==0 and len(out)<=1024**2 and len(err)<=1024**2,'telemetry_command')
            require(not err.strip(),'telemetry_warning_or_access_incomplete')
            return out.decode('utf-8',errors='strict')

    def sample(self,timeout):
        end=time.monotonic()+timeout
        gpu=self.command(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],end-time.monotonic())
        processes=self.command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],end-time.monotonic())
        cgroups={};starts={}
        for line in processes.splitlines():
            fields=line.split(',');require(len(fields)==3,'telemetry_process_columns')
            # Unrelated GPU1 desktop churn is not an execution-stage gate.
            if fields[0].strip()!=UUID:continue
            pid=fields[1].strip();require(pid.isdecimal(),'telemetry_pid')
            stat_path=Path('/proc',pid,'stat')
            before=process_start_time(stat_path.read_text())
            raw=Path('/proc',pid,'cgroup').read_text()
            require(len(raw)<=65536,'telemetry_cgroup_bound')
            require(process_start_time(stat_path.read_text())==before,'telemetry_pid_reused_during_sample')
            cgroups[pid]=raw;starts[pid]=before
        processes='\n'.join(line for line in processes.splitlines() if line.split(',')[0].strip()==UUID)
        # Inclusive cursor must reappear: empty success is not proof of access.
        journal=self.command(['journalctl','-k','--cursor='+self.cursor,'--no-pager','-o','json'],end-time.monotonic())
        cursor,messages=journal_records(journal,self.cursor)
        require(time.monotonic()<=end,'telemetry_deadline')
        sample=parse_snapshot(gpu,processes,cgroups,messages,starts)
        self.cursor=cursor
        return sample


class GPUCorrectnessStage(DockerReferenceStage):
    def __init__(self,manifest,digest,telemetry=None):
        self.manifest_path=plain_path(manifest)
        self.manifest_digest=digest
        self.manifest=bounded_json(self.manifest_path,1024**2,digest)
        m=self.manifest
        require(m['schema']==1 and m['stage']=='one-bos-gpu-correctness' and m['technical_release'] is True,'gpu_stage_release')
        require(m['image_id']==IMAGE and m['overlay_revision']==OVERLAY_REVISION,'gpu_image_pin')
        require(m['snapshot_hashes_verified'] is True and m['snapshot_writer_exclusion'] is True
                and m['reference_capture_reviewed'] is True,'gpu_prior_gates')
        require(m['input_ids']==[248044] and m['capacity']==1 and m['absolute_tolerance']==m['relative_tolerance']==0,'gpu_numerical_contract')
        self.evidence=self.manifest_path.parent
        require(self.evidence.stat().st_mode & 0o777==0o700,'gpu_evidence_mode')
        self.snapshot=plain_path(m['snapshot_root']);self.reference=plain_path(m['reference_directory'])
        self.telemetry=telemetry or HostTelemetry(m['kernel_journal_cursor'],self.evidence)
        self.evidence_children=self.telemetry.children
        self.child=None;self.logs=[];self.verified_id=None;self.ancestor_observer=None
        self.gate=None;self.final_oom=None

    def verify(self,cid,timeout):
        end=time.monotonic()+timeout
        m=self.manifest
        head=subprocess.run(['git','-C',str(REPO),'rev-parse','HEAD'],check=True,capture_output=True,text=True,timeout=timeout).stdout.strip()
        require(head==m['source_revision'],'gpu_source_revision')
        v=self.inspect(cid,max(0.001,end-time.monotonic()));h=v['HostConfig'];c=v['Config']
        require(v['Image']==IMAGE and c.get('Labels',{}).get('kadan.reference.run')==m['run_id']
                and c.get('Labels',{}).get('kadan.reference.stage')=='gpu-correctness','gpu_owned_identity')
        require(v['State']['Running'] is True and v['Path']=='/bin/sleep' and v['Args']==['infinity'],'gpu_inert_command')
        require(c.get('Healthcheck',{}).get('Test')==['NONE'] and h['AutoRemove'] is False
                and h['RestartPolicy']['Name']=='no','gpu_no_background_or_restart')
        require(h['Runtime']=='runc' and h['NetworkMode']=='none' and h['PidMode']==''
                and h['IpcMode']=='private' and h['CgroupnsMode']=='private','gpu_private_namespaces')
        require(h['Privileged'] is False and h['ReadonlyRootfs'] is True and h['CapDrop']==['ALL']
                and not h.get('CapAdd') and 'no-new-privileges' in h['SecurityOpt'],'gpu_capabilities')
        require(h['NanoCpus']==2*10**9 and h['Memory']==4*1024**3 and h['MemorySwap']==h['Memory']
                and h['PidsLimit']==256,'gpu_host_limits')
        requests=h.get('DeviceRequests')
        require(isinstance(requests,list) and len(requests)==1,'gpu_device_request')
        req=requests[0]
        require(req['Driver']=='nvidia' and req['DeviceIDs']==[UUID] and req['Capabilities']==[['gpu']]
                and req['Count']==0 and not req.get('Options') and not h.get('Devices')
                and not h.get('DeviceCgroupRules'),'gpu0_only')
        env=dict(item.split('=',1) for item in c['Env'])
        require(env.get('CUDA_VISIBLE_DEVICES')=='0' and env.get('NVIDIA_VISIBLE_DEVICES')==UUID
                and not env.get('LD_PRELOAD'),'gpu_visibility')
        mounts={(p['Source'],p['Destination'],p['RW'],p['Type']) for p in v['Mounts']}
        require(mounts=={(str(self.snapshot),'/snapshot',False,'bind'),
                         (str(self.reference),'/reference',False,'bind'),
                         (str(self.evidence),'/evidence',True,'bind')},'gpu_mounts')
        pid=v['State']['Pid'];leaf=Path('/sys/fs/cgroup/system.slice')/f'docker-{cid}.scope'
        require(Path(f'/proc/{pid}/cgroup').read_text().strip()==f'0::/system.slice/docker-{cid}.scope','gpu_init_cgroup')
        require((leaf/'cgroup.procs').read_text().split()==[str(pid)],'gpu_unexpected_child')
        self.cgroup=leaf;st=leaf.stat();self.cgroup_identity=(st.st_dev,st.st_ino)
        self.initial_oom=0
        self.ancestor_observer=AncestorObserver(leaf,pid,m['existing_ancestor'])
        sample=self.telemetry.sample(end-time.monotonic())
        self.gate=GpuGate(Snapshot(m['gpu_baseline']['devices'],tuple(m['gpu_baseline']['processes']),False))
        GpuGate(sample)
        # Baseline is explicitly supplied by the stopped-production plan, not
        # inferred from a later sample after a foreign allocation appeared.
        self.gate.evaluate(sample,cid,cleanup=True)
        self.verified_id=cid
        return True

    def start(self,cid,timeout):
        require(cid==self.verified_id and self.child is None,'gpu_start_once')
        sample=self.telemetry.sample(timeout);GpuGate(sample)
        self.gate.evaluate(sample,cid,cleanup=True)
        for name in ('stdout.raw','stderr.raw'):self.logs.append(ExclusiveOutput(self.evidence/name))
        self.child=subprocess.Popen(['docker','exec','-e','LD_LIBRARY_PATH=/opt/kadan/lib',cid,*NATIVE_ARGS],
                                    stdout=self.logs[0].fd,stderr=self.logs[1].fd,
                                    stdin=subprocess.DEVNULL,preexec_fn=log_limit)

    def poll(self,cid,timeout):
        end=time.monotonic()+timeout
        self.gate.evaluate(self.telemetry.sample(timeout),cid)
        require(time.monotonic()<end,'gpu_poll_deadline')
        return super().poll(cid,end-time.monotonic())

    def validate_artifacts(self,cid,timeout):
        require(cid==self.verified_id and self.child.poll()==0,'gpu_child_exit')
        return self.evidence_check('gpu',self.manifest_path,timeout)

    def observe(self,cid,timeout):
        end=time.monotonic()+timeout
        observation=super().observe(cid,timeout)
        sample=self.telemetry.sample(end-time.monotonic())
        self.gate.evaluate(sample,cid,cleanup=observation.running is False)
        count=sum(p['uuid']==UUID for p in sample.processes)
        return replace(observation,compute_process_count=count,
                       gpu_baseline_restored=observation.running is False and count==0)


def main():
    import argparse
    import signal
    from threading import Event
    from .supervisor import supervise, Policy
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-reviewed-correctness',action='store_true')
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--container-id',required=True)
    args=parser.parse_args()
    require(args.execute_reviewed_correctness,'gpu_execution_intent')
    cancelled=Event()
    signal.signal(signal.SIGTERM,lambda *_:cancelled.set())
    signal.signal(signal.SIGINT,lambda *_:cancelled.set())
    stage=GPUCorrectnessStage(args.manifest,args.manifest_sha256)
    result=supervise(stage,args.container_id,stage.evidence/'supervisor.json',execute=True,
                     policy=Policy(deadline=915),cancelled=cancelled.is_set)
    print(json.dumps(result))
    return 0 if result['accepted'] else 2


if __name__=='__main__':raise SystemExit(main())
