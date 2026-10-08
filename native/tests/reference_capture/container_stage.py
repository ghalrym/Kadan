"""CPU-reference Docker transport for a pre-created, explicitly owned container.

No create/start-service/restart/restore operation exists here. A stopped cgroup
that disappears before a final memory.events sample leaves cleanup uncertain.
"""
import json
import os
from pathlib import Path
import resource
import subprocess
import tempfile

from .artifacts import ExclusiveOutput, read, require, sha256
from .capture_actual import bounded_json, validate_manifest, plain_path, BUFFER
from .capture import REPO, PINS
from .supervisor import Observation

MAX_LOG = 8 * BUFFER


def log_limit():
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_LOG, MAX_LOG))


class DockerReferenceStage:
    def __init__(self, manifest, manifest_digest, run_id, source, snapshot, evidence):
        self.manifest_path = plain_path(manifest)
        self.manifest_digest = manifest_digest
        self.manifest = bounded_json(self.manifest_path, BUFFER, manifest_digest)
        validate_manifest(self.manifest)
        require(run_id and len(run_id) <= 128, 'run_id')
        self.run_id = run_id
        self.source, self.snapshot, self.evidence = map(plain_path,(source,snapshot,evidence))
        require(self.source == REPO, 'reviewed_source_mount')
        require(self.snapshot.name == self.manifest['snapshot'], 'snapshot_mount')
        require(self.manifest_path.parent == self.evidence, 'manifest_location')
        require(self.evidence.stat().st_mode & 0o777 == 0o700, 'evidence_mode')
        self.child = None
        self.logs = []
        self.cgroup = None
        self.initial_oom = None
        self.final_oom = None
        self.verified_id = None

    def rpc(self, args, timeout):
        # Bounded disk-backed capture: no unbounded communicate() buffers.
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            completed = subprocess.run(['docker',*args],stdout=out,stderr=err,
                                       timeout=timeout,check=False,preexec_fn=log_limit)
            require(completed.returncode == 0, 'docker_rpc:' + args[0])
            require(out.tell() <= MAX_LOG and err.tell() <= MAX_LOG, 'docker_rpc_output')
            out.seek(0)
            return out.read(MAX_LOG+1)

    def inspect(self, cid, timeout):
        values=json.loads(self.rpc(['inspect',cid],timeout))
        require(len(values)==1 and values[0]['Id']==cid, 'container_identity')
        return values[0]

    def oom(self):
        fields=dict(line.split() for line in (self.cgroup/'memory.events').read_text().splitlines())
        return int(fields['oom_kill'])

    def verify(self, cid, timeout):
        head=subprocess.run(['git','-C',str(self.source),'rev-parse','HEAD'],
                            capture_output=True,text=True,check=True,timeout=timeout).stdout.strip()
        require(head==self.manifest['source_revision'], 'source_head')
        validate_manifest(self.manifest)
        for name,digest in json.loads(PINS.read_text())['reference_sources'].items():
            require(sha256((self.source/name).read_bytes())==digest, 'reference_source')
        value=self.inspect(cid,timeout)
        host,config=value['HostConfig'],value['Config']
        require(value['Image']==self.manifest['image_id'], 'container_image')
        require(config.get('Labels',{}).get('kadan.reference.run')==self.run_id, 'container_ownership_label')
        require(value['State']['Running'] is True and value['Path']=='/bin/sleep'
                and value['Args']==['infinity'], 'inert_container_required')
        require(host['RestartPolicy']['Name']=='no' and host['NetworkMode']=='none'
                and host['ReadonlyRootfs'] is True and host['Privileged'] is False
                and host['PidMode']=='' and host['IpcMode']=='private', 'container_isolation')
        require(host.get('Runtime')=='runc' and not host.get('DeviceRequests')
                and not host.get('Devices') and not host.get('DeviceCgroupRules'), 'container_devices')
        require(0 < host['Memory'] <= 40*1024**3 and host['MemorySwap']==host['Memory']
                and 0 < host['NanoCpus'] <= 2*10**9 and 0 < host['PidsLimit'] <= 256, 'container_limits')
        require(host.get('CapDrop')==['ALL'] and not host.get('CapAdd')
                and 'no-new-privileges' in host.get('SecurityOpt',[]), 'container_privileges')
        env=dict(item.split('=',1) for item in config['Env'])
        require(not any(env.get(k) for k in ('LD_PRELOAD','LD_LIBRARY_PATH','PYTHONHOME','PYTHONPATH')), 'container_import_env')
        require(env.get('CUDA_VISIBLE_DEVICES')=='' and env.get('NVIDIA_VISIBLE_DEVICES')=='void', 'container_gpu_env')
        mounts={(p['Source'],p['Destination'],p['RW'],p['Type']) for p in value['Mounts']}
        expected={(str(self.source),'/work',False,'bind'),
                  (str(self.snapshot),'/snapshot/'+self.manifest['snapshot'],False,'bind'),
                  (str(self.evidence),'/evidence',True,'bind')}
        require(mounts==expected, 'container_mounts')
        pid=value['State']['Pid']
        lines=Path(f'/proc/{pid}/cgroup').read_text().splitlines()
        require(len(lines)==1 and lines[0].startswith('0::/'), 'cgroup_v2_required')
        relative=lines[0][4:]
        require(cid in relative and '..' not in Path(relative).parts, 'owned_cgroup_path')
        self.cgroup=Path('/sys/fs/cgroup')/relative
        self.initial_oom=self.oom()
        procs=(self.cgroup/'cgroup.procs').read_text().split()
        require(procs==[str(pid)], 'unexpected_container_children')
        self.verified_id=cid
        return True

    def start(self, cid, timeout):
        require(cid==self.verified_id and self.child is None, 'stage_start_once')
        try:
            for name in ('stdout.raw','stderr.raw'):
                self.logs.append(ExclusiveOutput(self.evidence/name))
            command=['docker','exec','-w','/work',cid,'python','-B','-m',
                     'native.tests.reference_capture.capture_actual','--execute-approved-actual-reference',
                     '--snapshot-root','/snapshot/'+self.manifest['snapshot'],
                     '--manifest','/evidence/'+self.manifest_path.name,
                     '--manifest-sha256',self.manifest_digest,'--output-dir','/evidence/capture']
            self.child=subprocess.Popen(command,stdout=self.logs[0].fd,stderr=self.logs[1].fd,
                                        stdin=subprocess.DEVNULL,preexec_fn=log_limit)
        except BaseException:
            for log in self.logs:log.close()
            raise

    def poll(self, cid, timeout):
        require(cid==self.verified_id and self.child is not None, 'stage_identity')
        code=self.child.poll()
        for log in self.logs:
            if log.fd>=0:require(os.fstat(log.fd).st_size < MAX_LOG, 'raw_output_limit')
        if code is not None:
            # Preserve streams even when the child failed; fsync errors fail the stage.
            for log in self.logs:
                if log.fd>=0:log.finish()
        return code

    def validate_artifacts(self, cid, timeout):
        require(cid==self.verified_id and self.child.poll()==0, 'child_exit')
        folder=self.evidence/'capture'
        data=(folder/'reference.capture').read_bytes() if (folder/'reference.capture').stat().st_size==993316 else b''
        record=read(folder/'reference.capture')
        p=folder/'diagnostic.json'
        require(p.stat().st_size <= MAX_LOG, 'diagnostic_bound')
        report=json.loads(p.read_bytes())
        require(record['input']==248044 and record['vocab']==248320
                and report['status']=='complete' and report['zero_reservations'] is True
                and report['calls']['model']==1 and report['capture_sha256']==sha256(data)
                and report['manifest_sha256']==self.manifest_digest, 'reference_artifacts')
        return True

    def terminate(self, cid, signal, timeout):
        require(cid==self.verified_id and signal in ('TERM','KILL'), 'owned_signal')
        # Capture events before teardown, but never substitute this sample for a final one.
        self.rpc(['kill','--signal',signal,cid],timeout)

    def observe(self, cid, timeout):
        require(cid==self.verified_id, 'owned_observation')
        state=self.inspect(cid,timeout)['State']
        reaped=self.child is None or self.child.poll() is not None
        if not state['Running'] and not reaped:
            # This terminates only our Docker client, not proof of container cleanup.
            self.child.kill()
            try:self.child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:pass
            reaped=self.child.poll() is not None
        if reaped:
            for log in self.logs:
                if log.fd>=0:log.finish()
        # Require a final cgroup sample after the owned child has exited. On engines
        # that remove the cgroup immediately this intentionally returns uncertainty.
        procs=None
        if self.cgroup.exists():
            procs=(self.cgroup/'cgroup.procs').read_text().split()
            if not procs and reaped:self.final_oom=self.oom()-self.initial_oom
        gone=not state['Running'] and state['Pid']==0 and procs==[] and reaped
        unchanged=sha256(self.manifest_path.read_bytes())==self.manifest_digest
        return Observation(cid,state['Running'],len(procs) if procs is not None else None,
                           0,reaped,gone,gone,state['OOMKilled'],self.final_oom,unchanged)
