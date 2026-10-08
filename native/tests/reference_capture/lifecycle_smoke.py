"""Explicit harmless CPU lifecycle smoke. Never launches capture_actual or a model.

A reviewer/operator creates and starts the exact inert container separately.
This command supervises only the recorded immutable disposable ID.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from .artifacts import ExclusiveOutput, require, sha256
from .capture import REPO
from .container_stage import DockerReferenceStage, log_limit
from .ancestor_observer import AncestorObserver, counter, identity, require_hierarchical_events
from .supervisor import Policy, supervise

IMAGE='sha256:985fc03e9ad5fee38fc50b5f088305c9ab94c9a7b3ac53addf2b20622f7c4e4e'
LABEL='observer-smoke-20261008'
WORK="a=bytearray(16*1024*1024); a[::4096]=bytes([1])*4096; print('SMOKE_DONE',len(a))"


class SmokeStage(DockerReferenceStage):
    def __init__(self,evidence,revision):
        self.evidence=Path(evidence).absolute()
        self.evidence.mkdir(mode=0o700)
        self.child=None
        self.logs=[]
        self.evidence_children=[]
        self.ancestor_observer=None
        self.verified_id=None
        self.revision=revision
        self.manifest_path=self.evidence/'smoke-manifest.json'
        data=json.dumps({'image':IMAGE,'source_revision':revision,'label':LABEL,'work':WORK},sort_keys=True).encode()
        manifest=ExclusiveOutput(self.manifest_path)
        try:manifest.write(data);manifest.finish()
        finally:manifest.close()
        self.manifest_digest=sha256(data)

    def verify(self,cid,timeout):
        require_hierarchical_events()
        head=subprocess.run(['git','-C',str(REPO),'rev-parse','HEAD'],check=True,
                            capture_output=True,text=True,timeout=timeout).stdout.strip()
        require(head==self.revision,'smoke_source_revision')
        v=self.inspect(cid,timeout);h=v['HostConfig'];c=v['Config']
        require(v['Image']==IMAGE and c.get('Labels',{}).get('kadan.reference.run')==LABEL,'smoke_identity')
        require(v['State']['Running'] is True and v['Path']=='/bin/sleep' and v['Args']==['infinity'],'smoke_inert_command')
        require(c.get('Healthcheck',{}).get('Test')==['NONE'],'smoke_healthcheck_disabled')
        require(h['AutoRemove'] is False and h['RestartPolicy']['Name']=='no','smoke_retained_metadata')
        require(h['Runtime']=='runc' and h['NetworkMode']=='none' and h['PidMode']==''
                and h['IpcMode']=='private' and h['CgroupnsMode']=='private','smoke_namespaces')
        require(h['Privileged'] is False and h['ReadonlyRootfs'] is True and h['CapDrop']==['ALL']
                and not h.get('CapAdd') and 'no-new-privileges' in h['SecurityOpt'],'smoke_capabilities')
        require(h['NanoCpus']==10**9 and h['Memory']==256*1024**2 and h['MemorySwap']==h['Memory']
                and h['PidsLimit']==32,'smoke_limits')
        require(not h.get('DeviceRequests') and not h.get('Devices') and not h.get('DeviceCgroupRules'),'smoke_no_devices')
        require(not v['Mounts'] and h.get('Tmpfs')=={'/tmp':'rw,size=16m'},'smoke_no_model_or_bind_mounts')
        env=dict(item.split('=',1) for item in c['Env'])
        require(env.get('NVIDIA_VISIBLE_DEVICES')=='void' and env.get('CUDA_VISIBLE_DEVICES')=='','smoke_gpu_env')
        pid=v['State']['Pid']
        lines=Path(f'/proc/{pid}/cgroup').read_text().splitlines()
        require(len(lines)==1 and lines[0]==f'0::/system.slice/docker-{cid}.scope','smoke_cgroup_identity')
        self.cgroup=Path('/sys/fs/cgroup')/lines[0][4:]
        require((self.cgroup/'cgroup.procs').read_text().split()==[str(pid)],'smoke_unexpected_children')
        self.cgroup_identity=tuple(identity(self.cgroup))
        self.initial_oom=counter(self.cgroup/'memory.events','oom_kill')
        self.final_oom=None
        require(self.initial_oom==0,'smoke_initial_oom')
        parent=self.cgroup.parent
        pin={'path':str(parent),'identity':identity(parent),'oom_kill':counter(parent/'memory.events','oom_kill'),
             'memory_current_ceiling':int((parent/'memory.current').read_text())+64*1024**2,
             'memory_max':(parent/'memory.max').read_text().strip(),
             'peers':sorted(p.name for p in parent.iterdir() if p.is_dir() and p!=self.cgroup),
             'host_floor_bytes':44*1024**3,'ancestor_floor_bytes':44*1024**3}
        self.ancestor_observer=AncestorObserver(self.cgroup,pid,pin)
        evidence=ExclusiveOutput(self.evidence/'observer-baseline.json')
        try:evidence.write(json.dumps(pin,sort_keys=True,indent=2).encode());evidence.finish()
        finally:evidence.close()
        self.verified_id=cid
        return True

    def start(self,cid,timeout):
        require(cid==self.verified_id and self.child is None,'smoke_start_once')
        for name in ('stdout.raw','stderr.raw'):
            self.logs.append(ExclusiveOutput(self.evidence/name))
        self.child=subprocess.Popen(['docker','exec',cid,'/usr/local/bin/python','-I','-S','-c',WORK],
                                    stdout=self.logs[0].fd,stderr=self.logs[1].fd,
                                    stdin=subprocess.DEVNULL,preexec_fn=log_limit)

    def validate_artifacts(self,cid,timeout):
        require(cid==self.verified_id and self.child.poll()==0,'smoke_exit')
        return self.evidence_check('smoke',self.evidence,timeout)



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-approved-smoke',action='store_true')
    parser.add_argument('--container-id',required=True)
    parser.add_argument('--source-revision',required=True)
    parser.add_argument('--evidence',required=True)
    args=parser.parse_args()
    require(args.execute_approved_smoke,'smoke_intent_required')
    # Signals are handled as flags, never exceptions through cleanup.
    import signal
    from threading import Event
    cancelled=Event()
    signal.signal(signal.SIGTERM,lambda *_:cancelled.set())
    signal.signal(signal.SIGINT,lambda *_:cancelled.set())
    stage=SmokeStage(args.evidence,args.source_revision)
    report=supervise(stage,args.container_id,stage.evidence/'supervisor.json',execute=True,
                     policy=Policy(deadline=10),cancelled=cancelled.is_set)
    print(json.dumps(report))
    return 0 if report['accepted'] else 2


if __name__=='__main__':raise SystemExit(main())
